from typing import Optional, Tuple, Union
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import BitsAndBytesConfig, AutoConfig, AutoTokenizer
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from .ring_attn_utils import convert_ring_attn_params
from .utils import log_probs_from_logits, reset_position_ids, packed_sequence_to_position_tensor
from ..utils.utils import get_generation_cls

def entropy_from_logits(logits: torch.Tensor):
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy

class Actor(nn.Module):
    """
    Base class for Actor models in reinforcement learning.

    This class serves as a foundation for implementing various actor models, which are responsible for selecting actions based on the policy learned from the environment.

    Args:
        pretrain_or_model (nn.Module): A pretrained model or a new model instance to be used as the actor.
        use_flash_attention_2 (bool, optional): Whether to utilize Flash Attention 2.0 for improved performance. Defaults to False.
        bf16 (bool, optional): Enable bfloat16 precision for model computations. Defaults to True.
        load_in_4bit (bool, optional): Load the model in 4-bit precision. Defaults to False.
        lora_rank (int, optional): Rank for LoRA adaptation. Defaults to 0.
        lora_alpha (int, optional): Alpha parameter for LoRA. Defaults to 16.
        lora_dropout (float, optional): Dropout rate for LoRA layers. Defaults to 0.
        target_modules (list, optional): List of target modules for applying LoRA. Defaults to None.
        ds_config (dict, optional): Configuration for DeepSpeed, enabling model partitioning across multiple GPUs. Defaults to None.
        device_map (dict, optional): Device mapping for loading the model onto specific devices. Defaults to None.
        packing_samples (bool, optional): Whether to pack samples during training. Defaults to False.
    """

    def __init__(
        self,
        pretrain_or_model,
        use_flash_attention_2=False,
        bf16=True,
        load_in_4bit=False,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        ds_config=None,
        device_map=None,
        packing_samples=False,
        **kwargs,
    ) -> None:
        super().__init__()
        
        if isinstance(pretrain_or_model, str):
            attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"
            print(f"!!!! actor using ", attn_implementation)
            # Note: dschf is defined in function scope to avoid global effects
            # https://huggingface.co/docs/transformers/deepspeed#non-trainer-deepspeed-integration
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None

            if load_in_4bit:
                assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                nf4_config = None

            #There is no AutoModelForConditionalGeneration in transformers. We manually implement it.
            if "Intern" not in pretrain_or_model:
                config = AutoConfig.from_pretrained(pretrain_or_model)
                model_cls = get_generation_cls(config)
                self.model = model_cls.from_pretrained(
                    pretrain_or_model,
                    trust_remote_code=True,
                    attn_implementation=attn_implementation,
                    quantization_config=nf4_config,
                    torch_dtype=torch.bfloat16 if bf16 else "auto",
                    device_map=device_map,
                )
            else:
                print(f"!!!! [warning] Is {pretrain_or_model} InternVL?")
                from openrlhf.internvl import InternVLChatModel
                from openrlhf.internvl.train.constants import IMG_CONTEXT_TOKEN
                from transformers import AutoTokenizer
                self.model = InternVLChatModel.from_pretrained(
                    pretrain_or_model,
                    trust_remote_code=True,
                    attn_implementation=attn_implementation,
                    quantization_config=nf4_config,
                    torch_dtype=torch.bfloat16 if bf16 else "auto",
                    device_map=device_map,
                )
                tokenizer = AutoTokenizer.from_pretrained(pretrain_or_model, trust_remote_code=True)
                self.model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)


            # LoRA
            if lora_rank > 0:
                # https://github.com/huggingface/peft/issues/137
                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)

                if load_in_4bit:
                    for name, module in self.model.named_modules():
                        if isinstance(module, LoraLayer):
                            module = module.to(torch.bfloat16)
                        if "norm" in name:
                            module = module.to(torch.float32)
                        if "lm_head" in name or "embed_tokens" in name:
                            if hasattr(module, "weight"):
                                module = module.to(torch.bfloat16)

            # MoE - balancing loss
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                print("[MoE] set output_router_logits as True")
                self.model.config.output_router_logits = True

            # https://github.com/huggingface/transformers/issues/26877
            # Use `model.generate(use_cache=True)` instead.`
            self.model.config.use_cache = False

            # packing samples using Flash Attention 2
            self.packing_samples = packing_samples
        else:
            self.model = pretrain_or_model

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, **kwargs) -> Union[
        Tuple[torch.LongTensor, torch.LongTensor],
        Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor],
    ]:
        generate_args = {
            "input_ids": input_ids,
            "top_k": kwargs.get("top_k", None),
            "top_p": kwargs.get("top_p", None),
            "do_sample": kwargs.get("do_sample", True),
            "early_stopping": kwargs.get("num_beams", 1) > 1,
            "temperature": kwargs.get("temperature", 1),
            "use_cache": True,
            "num_beams": kwargs.get("num_beams", 1),
            "attention_mask": kwargs.get("attention_mask"),
            "eos_token_id": kwargs.get("eos_token_id"),
            "pad_token_id": kwargs.get("pad_token_id"),
            "min_new_tokens": kwargs.get("min_new_tokens", 1),
        }

        if kwargs.get("max_new_tokens", None):
            generate_args["max_new_tokens"] = kwargs.get("max_new_tokens")
        if kwargs.get("max_length", None):
            generate_args["max_length"] = kwargs.get("max_length")

        # Call generate
        sequences = self.model.generate(**generate_args)

        # Prepare mask tensor
        eos_token_id = generate_args["eos_token_id"]
        pad_token_id = generate_args["pad_token_id"]

        return self.process_sequences(sequences, input_ids.size(1), eos_token_id, pad_token_id)

    def process_sequences(self, sequences: torch.Tensor, input_len, eos_token_id, pad_token_id):
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        seq_length = attention_mask.size(1)

        # The following code is equivalent to:
        #
        # for i in range(attention_mask.size(0)):
        #     for t in reversed(range(seq_length)):
        #         if attention_mask[i][t] > 0.5:
        #             attention_mask[i][min(t + 1, seq_length - 1)] = True
        #             sequences[i][min(t + 1, seq_length - 1)] = eos_token_id
        #             break
        #
        eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)
        sequences.scatter_(dim=1, index=eos_indices, value=eos_token_id)

        # For Llama3 and Qwen2 models, there are some eos_tokens in the middle of the prompt.
        first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)
        mask = torch.arange(seq_length).unsqueeze(0).expand(sequences.size(0), -1).to(device=sequences.device)
        attention_mask = (mask >= first_token_indices) & (mask <= eos_indices).to(dtype=torch.long)

        # in RL, state_i (current token) + action_i (next token) -> state_i+1 (next token)
        state_seq = sequences[:, input_len - 1 : -1]
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        if action_mask.size(1)>0:
            action_mask[:, 0] = 1

        return sequences, attention_mask, action_mask

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: Optional[Union[int, list[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        ring_attn_group: Optional[dist.ProcessGroup] = None,
        packed_seq_lens: Optional[list[int]] = None,
        visual_inputs: Optional[dict] = None,
        with_entropy=False,
    ) -> torch.Tensor:
        """Returns action log probs"""
        if visual_inputs is None:
            visual_inputs = {}
        '''
        for k,v in visual_inputs.items():
            if v.dtype == torch.float32:
                visual_inputs[k] = v.to(self.model.get_input_embeddings().weight.dtype)
        '''
        cu_seqlens = None
        if not self.packing_samples:
            # https://github.com/OpenRLHF/OpenRLHF/issues/217
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        else:
            # convert attention_mask to position_ids
            if ring_attn_group is not None:
                sequences, attention_mask, position_ids = convert_ring_attn_params(
                    sequences, attention_mask, packed_seq_lens, ring_attn_group
                )
            else:
                # packed_input_ids = sequences
                # device = sequences.device
                # packed_attention_mask = torch.zeros((1, packed_input_ids.shape[1], packed_input_ids.shape[1]), dtype=torch.bool).to(device) # Initialize mask
                # start_index = 0
                # packed_position_ids = []
                # for seq_len in packed_seq_lens:
                #     # seq_len = packed_seq_lens[i].shape[0]
                #     packed_attention_mask[:, start_index:start_index + seq_len, start_index:start_index + seq_len] = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool)) # Lower triangular mask for each sequence block
                #     start_index += seq_len
                #     packed_position_ids.append(torch.arange(0, seq_len))
                # packed_position_ids = torch.cat(packed_position_ids, dim=0).unsqueeze(0).to(device)
                # cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(packed_seq_lens), dim=0))).int().to(device)

                # attention_mask = packed_attention_mask
                # position_ids = packed_position_ids
                
                position_ids = reset_position_ids(attention_mask)
                # position_ids = packed_sequence_to_position_tensor(packed_seq_lens, sequences.device)
                # print(position_ids, len(position_ids[0]))
            # explicitly ignore attention_mask for packing_samples
            attention_mask = None
        # ntoken_each = [(x==151655).to(float).sum().item() for x in sequences]
        # if sum(ntoken_each)==2377:
        #     print(f'===========\ntraining imagepad in sequences {ntoken_each} sum to {sum(ntoken_each)}\n========')
        #     print('!!!!!!!! 2377')
        #     print(visual_inputs['pixel_values'].shape)
        # import pdb; pdb.set_trace()
        # try:
        output = self.model(sequences, attention_mask=attention_mask, position_ids=position_ids, **visual_inputs)
        # except:
        #     print(f"seq imgpad", [(xx==151655).sum().item() for xx in sequences])
        #     print(f"imagegrd {visual_inputs['image_grid_thw']}, pixel shape", visual_inputs['pixel_values'].shape)
        #     breakpoint()
        # import pdb; pdb.set_trace()
        # https://github.com/OpenRLHF/OpenRLHF/pull/634
        output["logits"] = output["logits"].to(torch.float32)

        if num_actions is None:
            assert return_output
            return output

        log_probs = log_probs_from_logits(output["logits"][:, :-1, :], sequences[:, 1:])

        if not self.packing_samples:
            action_log_probs = log_probs[:, -num_actions:]
            action_entropy = None
            # action_logits = output["logits"][:, -num_actions-1:-1, :]
            # action_entropy = entropy_from_logits(action_logits)
        else:
            assert isinstance(num_actions, list) and len(num_actions) == len(packed_seq_lens)
            action_log_probs = []
            offset = 0
            for num_action, seq_len in zip(num_actions, packed_seq_lens):
                start, end = max(0, offset + seq_len - num_action - 1), offset + seq_len - 1
                action_log_probs.append(log_probs[:, start:end])
                offset += seq_len
            action_log_probs = torch.cat(action_log_probs, dim=1)
            action_entropy = None

        if return_output:
            if with_entropy:
                action_logits = output["logits"][:, -num_actions-1:-1, :]
                action_entropy = entropy_from_logits(action_logits)
                # pass 
            output['action_entropy'] = action_entropy
            return (action_log_probs, output)
        else:
            return action_log_probs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()
        
if __name__ == '__main__': 
    mp = "/home/ma-user/work/data_mllm/pretrain_models/Qwen2.5-VL/Qwen2.5-VL-3B-Instruct"
    actor = Actor(mp, True, device_map='cuda', packing_samples=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(mp, trust_remote_code=True, use_fast=True)
    ids = [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 151652, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151653, 2121, 6839, 304, 279, 7071, 11, 14137, 374, 279, 23033, 315, 12671, 506, 11, 356, 374, 264, 1459, 389, 279, 12671, 11, 323, 1459, 422, 374, 279, 81963, 315, 15580, 18040, 13, 1416, 11995, 254, 25411, 284, 220, 20, 15, 11616, 11, 1221, 279, 6629, 315, 11995, 254, 53572, 374, 12668, 32, 13, 220, 20, 15, 11616, 198, 33, 13, 220, 17, 20, 11616, 198, 34, 13, 220, 17, 15, 11616, 198, 35, 13, 220, 19, 15, 11616, 271, 5501, 2874, 3019, 553, 3019, 11, 323, 2182, 697, 1590, 4226, 2878, 1124, 79075, 46391, 151645, 198, 151644, 77091, 198, 1249, 11625, 279, 3491, 11, 582, 1184, 311, 23643, 279, 2661, 1995, 323, 990, 5888, 315, 25362, 323, 25941, 3019, 553, 3019, 382, 16, 13, 3070, 28301, 1437, 279, 2661, 1995, 25, 1019, 256, 481, 14137, 374, 279, 23033, 315, 12671, 506, 624, 256, 481, 356, 374, 264, 1459, 389, 279, 12671, 624, 256, 481, 422, 374, 279, 81963, 315, 15580, 18040, 624, 256, 481, 11995, 254, 25411, 284, 220, 20, 15, 11616, 382, 17, 13, 3070, 10253, 279, 3343, 315, 25941, 708, 4712, 124833, 125677, 44026, 428, 110210, 6944, 50202, 14567, 25737, 57859, 11798, 129842, 54514, 30433, 15361, 26, 75069, 144753, 146111, 20026, 122, 98428, 127034, 78651, 42508, 20170, 25325, 3165, 2707, 55195, 69495, 635, 68893, 81916, 109211, 89676, 68448, 137956, 116375, 96946, 5956, 55641, 69185, 17390, 20305, 256, 75753, 436, 62416, 230, 67778, 63741, 55914, 8311, 90762, 46570, 27925, 70764, 69, 24341, 3574, 33467, 63488, 4711, 4082, 84715, 89214, 151813, 38704, 116912, 17610, 60618, 83007, 21352, 12521, 25981, 136748, 26477, 77730, 52500, 48562, 11023, 32618, 69714, 41772, 10721, 30659, 4657, 4366, 56067, 145795, 38632, 65260, 18381, 46503, 47915, 114527, 145357, 108846, 83853, 88741, 126842, 15509, 33052, 21156, 79955, 85171, 57320, 9199, 11159, 147728, 25912, 76204, 89053, 13155, 29874, 19845, 87083, 35613, 74037, 9727, 98963, 75373, 138092, 61772, 126756, 51821, 18749, 96094, 125309, 127541, 39837, 698, 124547, 84009, 8784, 23529, 57836, 37553, 146414, 72930, 86089, 113, 238, 132589, 145139, 70966, 188, 145338, 147601, 66922, 111555, 30254, 93175, 76975, 25452, 3528, 70624, 96557, 171, 223, 119, 96808, 288, 148292, 79936, 139377, 18429, 57602, 18137, 224, 96, 92181, 150384, 10261, 67012, 15080, 24309, 26731, 23587, 63553, 51733, 118713, 11324, 41450, 128241, 227, 94974, 77745, 149790, 95205, 126987, 150945, 27592, 117559, 62238, 44594, 70567, 54720, 24644, 85901, 90585, 98371, 136075, 48476, 92655, 83818, 80476, 122890, 247, 455, 77820, 127570, 393, 94852, 90307, 76191, 143985, 44947, 31701, 37587, 132558, 124649, 143434, 137969, 86089, 113, 225, 115243, 78578, 145408, 43397, 1055, 32295, 88314, 113541, 146728, 86771, 128132, 93928, 21055, 7715, 40609, 86555, 70859, 24852, 35523, 87725, 11309, 38502, 44064, 80328, 26211, 97651, 12204, 87560, 3912, 143634, 46389, 96667, 89525, 52667, 50623, 78522, 62863, 147238, 150360, 17814, 38931, 24157, 59068, 93838, 143801, 119992, 59977, 91199, 143746, 13949, 242, 17168, 147266, 120597, 30738, 144522, 82125, 147868, 141345, 54389, 7903, 128132, 123431, 88899, 2345, 81213, 49624, 133209, 13468, 99039, 143710, 127270, 117704, 14045, 145371, 8773, 16238, 21704, 116, 147439, 142992, 74655, 343, 53486, 90304, 16192, 61323, 97562, 145051, 19992, 29540, 55186, 14398, 86089, 113, 238, 23178, 92054, 147840, 147260, 93873, 200, 140358, 29456, 121667, 254, 73448, 148367, 126176, 147380, 64044, 71905, 121107, 65846, 145676, 146917, 59109, 25053, 58198, 145419, 68972, 333, 755, 60432, 48412, 64444, 12310, 149444, 137, 250, 118608, 6988, 66450, 145803, 127390, 90041, 72554, 53225, 147726, 151864, 4082, 54873, 12398, 66917, 90945, 1859, 123536, 138003, 96224, 136680, 45004, 132432, 146632, 228, 149137, 85416, 54873, 131013, 86599, 63295, 79125, 69389, 76857, 87235, 95166, 133413, 68250, 95, 135709, 140068, 73837, 2159, 67284, 75984, 86658, 116374, 129118, 131918, 142316, 27757, 137414, 143363, 133330, 34559, 122133, 53625, 9031, 131186, 145264, 73746, 45446, 69147, 1551, 35231, 140782, 35529, 54050, 69673, 14125, 43453, 22988, 94672, 30197, 33188, 45397, 148607, 83415, 61052, 112107, 127985, 78926, 42531, 12605, 93134, 33198, 86, 31233, 143368, 50774, 149220, 13588, 25513, 22542, 9199, 19374, 150667, 76566, 149766, 134056, 86094, 43028, 134367, 75584, 79939, 63480, 17211, 62970, 18775, 139814, 888, 135572, 721, 99156, 100, 148171, 135132, 10622, 45454, 70857, 64501, 136981, 48808, 71129, 144192, 151652, 56585, 123827, 119, 3396, 144937, 47786, 79065, 90470, 144772, 128948, 37448, 145871, 62366, 124296, 252, 14657, 149199, 17871, 30547, 146222, 148559, 149496, 147274, 49779, 57100, 65630, 147408, 145773, 224, 124446, 5388, 43516, 122370, 151887, 94382, 14493, 33351, 271, 74677, 147578, 10727, 40614, 59262, 13871, 63676, 122127, 86656, 140873, 121562, 30712, 65976, 46964, 79278, 85534, 148896, 89372, 149007, 18851, 19271, 63517, 76139, 53628, 148600, 91801, 32090, 87233, 59172, 57968, 145296, 121528, 140678, 147849, 127979, 147002, 58, 124198, 105, 32596, 143477, 22227, 69260, 81293, 68101, 5158, 13272, 52310, 22448, 94601, 124049, 253, 123998, 99649, 253, 13223, 97469, 149004, 51974, 75407, 130493, 21509, 94309, 123221, 4314, 198, 133157, 139795, 13490, 87257, 53717, 98410, 64497, 47841, 98495, 67751, 14451, 50390, 64152, 31636, 198, 54474, 90881, 135564, 135175, 145944, 141848, 147491, 122739, 144167, 927, 57855, 8265, 32554, 41649, 90923, 75162, 64161, 41377, 128437, 144693, 138796, 79082, 144397, 47384, 67130, 15224, 254, 148108, 139189, 120069, 144656, 48440, 36161, 129909, 148103, 18754, 89611, 92, 32840, 70, 39848, 302, 37601, 88626, 133208, 147549, 87763, 61586, 143642, 92528, 96086, 43296, 146842, 136996, 149821, 55861, 43731, 137616, 141519, 77103, 150130, 29642, 52411, 140237, 98243, 134971, 127629, 40954, 34559, 145798, 103723, 61234, 150715, 147238, 80939, 9641, 41963, 151917, 88780, 75262, 147950, 125066, 57522, 148189, 11662, 143444, 148700, 4594, 139021, 124376, 147455, 92558, 94530, 34008, 31191, 146277, 37841, 3640, 42102, 93448, 55866, 44357, 136, 110, 9196, 54573, 72821, 77077, 53566, 81441, 58424, 20953, 148773, 93070, 3696, 89954, 37966, 80908, 123115, 124245, 99, 119964, 122, 142320, 1130, 97691, 117811, 88114, 65622, 18546, 133327, 152008, 88092, 8925, 66557, 24121, 30413, 84019, 21818, 54347, 67847, 138763, 59419, 48109, 26061, 151435, 55435, 150362, 119527, 69305, 50194, 36219, 79917, 134789, 131463, 73303, 138142, 138058, 30496, 130447, 122722, 143935, 67790, 15119, 146195, 147839, 42, 125918, 85046, 57832, 85095, 39927, 130001, 81363, 59565, 56469, 88577, 150744, 7138, 93547, 106626, 77122, 144300, 148709, 88997, 17308, 72435, 146945, 148628, 149215, 97216, 146506, 22543, 35023, 15374, 92181, 81896, 81441, 66327, 47, 83102, 96717, 121398, 85508, 50531, 8243, 113986, 55235, 137483, 139519, 146866, 60817, 80390, 143916, 2258, 133300, 62479, 126817, 151937, 138573, 145320, 22255, 16277, 16528, 151520, 72560, 148311, 45454, 33087, 73295, 138611, 123468, 143580, 37252, 148824, 132791, 30238, 36876, 19304, 96669, 32097, 133125, 47085, 146869, 43467, 148620, 131058, 54329, 11690, 8534, 149318, 59397, 87676, 33342, 120946, 56638, 150381, 113155, 30004, 91979, 91391, 112981, 134270, 140672, 124744, 229, 147980, 149004, 28443, 144481, 147166, 5667, 54237, 122306, 139681, 83262, 84509, 134660, 136540, 59068, 126987, 151187, 60569, 25639, 72522, 46901, 31464, 143197, 85097, 45387, 148386, 147073, 137926, 39895, 126431, 12250, 145148, 130824, 145818, 69328, 70199, 151850, 123235, 1097, 58643, 128226, 224, 90607, 57820, 137536, 96220, 127963, 149055, 79832, 60798, 28332, 4702, 15878, 150323, 86296, 19389, 136628, 46723, 138085, 93993, 89332, 13203, 16418, 118103, 32872, 61451, 33208, 123840, 235, 122329, 51288, 54662, 148712, 45148, 935, 66273, 28386, 61612, 78387, 123683, 32971, 147658, 124047, 124999, 234, 82976, 198, 20223, 146770, 146144, 44485, 151291, 70841, 935, 8583, 86105, 66, 108350, 126023, 64374, 53678, 27275, 73048, 29225, 94032, 26813, 66874, 123285, 147579, 36524, 95910, 93649, 37572, 97593, 148751, 40333, 91849, 38701, 88110, 80843, 39593, 22284, 124149, 236, 68760, 6692, 151241, 52672, 38045, 76191, 54189, 148113, 43448, 145090, 2233, 118874, 127729, 22370, 99007, 1444, 85132, 71221, 63191, 13377, 50729, 9060, 147888, 140772, 139978, 87972, 139220, 147151, 131383, 101825, 137993, 32582, 17845, 11323, 130760, 70305, 66015, 82027, 148623, 66052, 147546, 121752, 146554, 6862, 39999, 13547, 129899, 133814, 120140, 92723, 30171, 25969, 16511, 141470, 140535, 95869, 132105, 63820, 146269, 19113, 235, 144320, 145831, 41334, 80293, 44702, 42013, 147090, 147263, 151645]
    tmp = tokenizer.decode(ids)
    print(tmp)
    # texts = ["""<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\nIn the base 10 arithmetic problem $H M M T+G U T S=R O U N D$, each distinct letter represents a different digit, and leading zeroes are not allowed. What is the maximum possible value of $R O U N D$?<|im_end|>\n<|im_start|>assistant\nTo solve the problem \\( H M M T + G U T S = R O U N D \\), we need to maximize the value of \\( R O U N D \\). This involves strategically assigning the values to each letter such that the sum of the two numbers is maximized.\n\nFirst, let's consider the structure of the numbers:\n- \\( HMMT \\) is a 4-digit number.\n- \\( GUTS \\) is a 4-digit number.\n- \\( ROUND \\) is a 5-digit number.\n\nTo maximize \\( ROUND \\), we should maximize the largest possible digits for each position in the sum. Let's start with the largest digits for \\( R \\) and \\( O \\).\n\n### Step 1: Assigning the largest digits to \\( R \\) and \\( O \\)\nThe largest digit in base 10 is 9. To maximize \\( ROUND \\), we should place the largest possible digits for \\( R \\) and \\( O \\). Let's try \\( R = 9 \\) and \\( O = 9 \\).\n\n### Step 2: Assigning the remaining digits\nAfter assigning 9 to \\( R \\) and \\( O \\), we need to ensure that the remaining digits are distinct and do not conflict with the other constraints.\n\n### Step 3: Ensuring no leading zeroes\nSince leading zeroes are not allowed, we need to ensure that the smallest possible digit is used for the units place of the sum. This means we need to consider the carry from the previous columns.\n\n### Step 4: Assigning the remaining digits\nLet's assign the remaining digits \\( 8, 7, 6, 5, 4, 3, 2, 1 \\) to \\( H, M, M, T, G, U, T, S \\). We need to ensure that \\( HMMT \\) and \\( GUTS \\) are valid 4-digit numbers and that the sum is correct.\n\n### Step 5: Verifying the sum\nWe need to verify that the sum \\( HMMT + GUTS = 9999 \\) is correct and all digits are distinct.\n\nLet's try different combinations:\n- \\( HMMT = 9876 \\)\n- \\( GUTS = 9253 \\)\n\nCalculate the sum:\n\\[ 9876 + 9253 = 19129 \\]\n\nThis does not work because \\( ROUND \\) is not a 5-digit number.\n\n### Step 6: Adjusting the digits\nWe need to try different combinations while ensuring all digits are distinct. After several trials, we find that:\n\n- \\( HMMT = 9876 \\)\n- \\( GUTS = 9253 \\)\n- \\( ROUND = 19129 \\)\n\nThis combination works because:\n\\[ 9876 + 9253 = 19129 \\]\n\nAll digits are distinct, and the sum is correct.\n\n### Final Answer\nThe maximum possible value of \\( ROUND \\) is:\n\\[\n\\boxed{9999}\n\\]<|im_end|>"""
    #          ,
    #          """<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\nIn the base 10 arithmetic problem $H M M T+G U T S=R O U N D$, each distinct letter represents a different digit, and leading zeroes are not allowed. What is the maximum possible value of $R O U N D$?<|im_end|>\n<|im_start|>assistant\nTo find the maximum possible value of \\(R O U N D\\) in the base 10 arithmetic problem \\(H M M T + G U T S = R O U N D\\), we need to maximize the sum of the two 4-digit numbers \\(H M M T\\) and \\(G U T S\\). The maximum digits available are 9, 8, 7, 6, 5, 4, 3, 2, 1, and 0. The largest possible digits should be placed in the highest place values of each number to maximize the sum.\n\nWe start by assigning the largest possible digits to the highest place values of \\(H M M T\\) and \\(G U T S\\). Let's try \\(H = 9\\), \\(M = 8\\), \\(T = 9\\), \\(G = 8\\), \\(U = 7\\), and \\(S = 6\\). This gives us the numbers \\(H M M T = 9889\\) and \\(G U T S = 8796\\). Now we calculate their sum:\n\n\\[\n9889 + 8796 = 18685\n\\]\n\nWe need to check if this is the maximum possible value. If we try to replace any of the digits with a smaller value, the sum will be less than 18685. For example, if we try \\(H = 9\\), \\(M = 8\\), \\(T = 9\\), \\(G = 8\\), \\(U = 7\\), and \\(S = 5\\), we get:\n\n\\[\n9889 + 8756 = 18645\n\\]\n\nThis sum is less than 18685. If we try smaller values for \\(M, T, U, S\\), the sums will be even smaller. Therefore, the maximum possible value of \\(R O U N D\\) is when \\(H = 9\\), \\(M = 8\\), \\(T = 9\\), \\(G = 8\\), \\(U = 7\\), and \\(S = 6\\), giving us:\n\n\\[\n\\boxed{18685}\n\\]<|im_end|>"""
    #          ]
    # tensors = []
    # masks = []
    # for idx,text in enumerate(texts):
    #     # Tokenize the text, ensuring no special tokens are added
    #     token_ids = tokenizer(text, add_special_tokens=False)['input_ids'] # or tokenizer.encode(text, add_special_tokens=False)

    #     # Convert token IDs to a PyTorch tensor
    #     token_tensor = torch.tensor(token_ids).to('cuda')

    #     # Append the tensor to the list
    #     tensors.append(token_tensor)
    #     masks.append(torch.ones_like(token_tensor)*(idx+1))
    
    # # import pdb; pdb.set_trace()
    # tmp = actor.forward(torch.cat(tensors+[torch.tensor([0,0]).to('cuda')])[None], attention_mask=torch.cat(masks)[None], packed_seq_lens=[len(x) for x in tensors]+[1,1], return_output=True)
    # print(tmp)
    # tmp2 = actor.forward(torch.cat(tensors[:1])[None], attention_mask=torch.cat(masks[:1])[None], packed_seq_lens=[len(x) for x in tensors[:1]], return_output=True)
    # print(tmp2)
    # tmp3 = actor.forward(torch.cat(tensors[1:])[None], attention_mask=torch.cat(masks[1:])[None], packed_seq_lens=[len(x) for x in tensors[1:]], return_output=True)
    # print(tmp3)
