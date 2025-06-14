# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union

from trl import create_reference_model
import numpy as np
import torch.distributed as dist



import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    TrainingArguments,
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLProcessor,
    Qwen2_5_VLProcessor,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation

from trl.trainer.utils import generate_model_card, get_comet_experiment_url
import PIL.Image
from typing import List
import copy
from PIL import Image
import json


from vlm_modules.vlm_module import VLMBaseModule
if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb





class Qwen2VLSFTToolTrainer(Trainer):


    def __init__(
        self,
        vlm_module: VLMBaseModule,
        model: Union[str, PreTrainedModel],

        args: TrainingArguments = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        torch_dtype: str = "bfloat16",
        is_base_model: Optional[bool] = False,
        tools: Optional[list[str]] = None,
        weight:Optional[torch.Tensor]=None,
        freeze_vision_modules: Optional[bool] = False,
        use_final_answer: Optional[bool] = False,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        self.vlm_module = vlm_module


        
        model_init_kwargs =  {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if model_init_kwargs.get("torch_dtype") is None:
            model_init_kwargs["torch_dtype"] = torch_dtype
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            # model_init_kwargs["use_cache"] = True
            model = vlm_module.get_model(model_id,model_init_kwargs=model_init_kwargs)
            
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )
        #TODO: freeze vision modules
        if freeze_vision_modules:
            print("Freezing vision modules...")
            for n, p in model.named_parameters():
                if any(keyword in n for keyword in self.vlm_module.get_vision_modules_keywords()):
                    p.requires_grad = False



        if use_final_answer:
            self.final_answer = FinalAnswer()
            self.funcs = [self.final_answer.function]
        else:
            self.funcs = []
        # tools
        self.tools = [tools] if isinstance(tools, str) else tools



        self.funcs = self.vlm_module.tool_des_postprocess(self.funcs)

        

        # Processing class
        if processing_class is None:
            processing_class = vlm_module.get_processor(model_id)
        




        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
# = G in the GRPO paper


        self.weight = weight
       
        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True



        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
         # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False
        self.processing_class.tokenizer.padding_side = "left"


        

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_colu
        # mns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["message_list"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model,inputs,logits_to_keep, weight:Optional[torch.Tensor]=None):
        inputs.to(model.device)
        logits = model(**inputs).logits 
        input_ids = inputs.input_ids
         # (B, L, V)
        if weight is not None:
            answer = self.processing_class.tokenizer.encode("answer", add_special_tokens=False)[0]
            numbers = self.processing_class.tokenizer.encode("1234567890", add_special_tokens=False)


        # print(f'logits shape: {logits.shape}')
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        logits_to_keep = logits_to_keep[:, 1:]

        per_token_logps = []
        for seq_logits, seq_input_ids, seq_mask in zip(logits, input_ids, logits_to_keep):
            # Select only the tokens where mask is True
            masked_logits = seq_logits[seq_mask]  # (N, V) where N is number of True values
            masked_input_ids = seq_input_ids[seq_mask]  # (N,)

            # Calculate log probabilities for the selected tokens
            log_probs = masked_logits.log_softmax(dim=-1)  # (N, V)
            if weight is not None:
                answer_positions = (masked_input_ids == answer).nonzero().flatten()

                if len(answer_positions) == 1:
                    answer_zone = masked_input_ids[answer_positions[0]:answer_positions[0]+10]
                    # Find positions of number tokens in the answer zone
                    number_positions = []
                    for i, token_id in enumerate(answer_zone):
                        if token_id in numbers:
                            number_positions.append(answer_positions[0] + i)
                    
                    # If number positions found, apply weight to those positions
                    if number_positions:
                        weight_mask = torch.ones_like(log_probs)
                        for pos in number_positions:
                            weight_mask[pos] = weight
                        log_probs = log_probs * weight_mask
            token_log_prob = torch.gather(log_probs, dim=1, index=masked_input_ids.unsqueeze(1)).squeeze(1)  # (N,)
            per_token_logps.append(token_log_prob)
        max_length = max(len(logps) for logps in per_token_logps)
        padded_per_token_logps = []
        padding_masks = []
        for logps in per_token_logps:
            mask = torch.ones(max_length, device=logps.device)
            mask[len(logps):] = 0
            padding_masks.append(mask)
            
            # Pad the log probabilities with zeros
            padding = torch.zeros(max_length - len(logps), device=logps.device)
            # Concatenate original logps with padding to maintain gradient flow
            padded_logps = torch.cat([logps, padding])
            padded_per_token_logps.append(padded_logps)
        per_token_logps = torch.stack(padded_per_token_logps,dim=0)
        padding_masks = torch.stack(padding_masks,dim=0)
            
            
        
        return per_token_logps, padding_masks


    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:

        return inputs
    def has_unfinished_sample(self, this_peer_finished: bool,device:torch.device) -> bool:
        this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(device)
        dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)

        if this_peer_finished_flag.item() == 0.0:
            return False
        
        return True




    def create_assistant_response_mask(self,inputs, processor, if_use_weighted:bool=False):
        """
        Create a boolean mask for the assistant's responses based on the chat template format.
        """
        mask = torch.zeros_like(inputs, dtype=torch.bool)
        # weighted_mask = torch.zeros_like(inputs, dtype=torch.bool)
        # Get special token IDs
        im_start = processor.tokenizer.encode(self.vlm_module.get_im_start(), add_special_tokens=False)[0]
        im_end = processor.tokenizer.encode(self.vlm_module.get_im_end(), add_special_tokens=False)[0]
        assistant = processor.tokenizer.encode(self.vlm_module.get_assistant(), add_special_tokens=False)[0]
        # answer = processor.tokenizer.encode("answer", add_special_tokens=False)[0]


        
        # For each sequence in the batch
        for i in range(inputs.shape[0]):
            sequence = inputs[i]
            

            # Find all im_start positions
            im_start_positions = (sequence == im_start).nonzero().flatten()
            if True:
                pos = im_start_positions[-2]
                if pos + 1 < len(sequence) and sequence[pos + 1] == assistant:
                    next_end = sequence[pos:].eq(im_end).nonzero()
                    if len(next_end) > 0:
                        end_pos = pos + next_end[0].item()
                        # Mark the entire response (including the im_start and im_end tokens)
                        mask[i, pos:end_pos + 1] = True
                pos = im_start_positions[-4]
                if pos + 1 < len(sequence) and sequence[pos + 1] == assistant:
                    next_end = sequence[pos:].eq(im_end).nonzero()
                    if len(next_end) > 0:
                        end_pos = pos + next_end[0].item()
                        # Mark the entire response (including the im_start and im_end tokens)
                        mask[i, pos:end_pos + 1] = True
            else:
                for pos in im_start_positions:
                    # Check if the token after im_start is "assistant"
                    if pos + 1 < len(sequence) and sequence[pos + 1] == assistant:
                    # Find the next im_end
                        next_end = sequence[pos:].eq(im_end).nonzero()
                        if len(next_end) > 0:
                            end_pos = pos + next_end[0].item()
                            # Mark the entire response (including the im_start and im_end tokens)
                            mask[i, pos:end_pos + 1] = True
                            
                        # Debug print

            # if if_use_weighted:
            #     final_answer_positions = (sequence == answer).nonzero().flatten()
            #     if len(final_answer_positions) == 3:
            #         pos = final_answer_positions[-1]
            #         weighted_mask[i, pos+2] = True

        
        return mask
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

    
        device = self.accelerator.device
        message_lists = [x["message_list"] for x in inputs]

        # z2orcorrects = [x["z2orcorrect"] for x in inputs]


        # Handle both pre-loaded images and image paths

        messages = message_lists

        inputs = self.vlm_module.prepare_from_msg_2_vlm_inputs(self.processing_class,messages)



        # Generate completions






# Right pad and stack tensors

        
        padded_input_ids = inputs.input_ids


        all_logits_to_keep = self.create_assistant_response_mask(padded_input_ids, self.processing_class)
        # Concatenate prompt_mask with completion_mask for logit computation
        # attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)
        # pixel_values = prompt_inputs["pixel_values"].repeat(self.num_generations, 1)
        # image_grid_thw = prompt_inputs["image_grid_thw"].repeat_interleave(self.num_generations, dim=0)
        # self.accelerator.wait_for_everyone()
        # print(f"Process {self.accelerator.process_index}: Tensor shapes - pixel_values: {stacked_pixel_values.shape}, image_grid: {stacked_image_grid_thw.shape}")
        # print(f'starting to compute per_token_logps,local rank: {self.accelerator.process_index},now step: {self.state.global_step}')
      

        per_token_logps, completion_mask = self._get_per_token_logps(model, inputs,all_logits_to_keep,self.weight)
        # print(f'starting to compute ref_per_token_logps,local rank: {self.accelerator.process_index},now step: {self.state.global_step}')
            

  
        # Compute the KL divergence between the model and the reference model


        loss = -(per_token_logps * completion_mask).sum() / completion_mask.sum()

        # Log the metrics
        mode = "train"
        self._metrics[mode]["loss"].append(self.accelerator.gather_for_metrics(loss).mean().item())
                    
        return loss
    def eval_prediction_step(self,model,inputs):
        old_message_lists = [x["message_list"] for x in inputs]
        question_image_paths = [x["question_image_path"] for x in inputs]
        images = []
        for question_image_path in question_image_paths:
            if isinstance(question_image_path,str):
                img = PIL.Image.open(question_image_path)


            # Ensure minimum dimensions of 28 pixels
            w, h = img.size
            if w < 28 or h < 28:
                # Calculate new dimensions maintaining aspect ratio
                if w < h:
                    new_w = 28
                    new_h = int(h * (28/w))
                else:
                    new_h = 28
                    new_w = int(w * (28/h))
                img = img.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
            
            images.append(img)

        # Handle both pre-loaded images and image paths



        # Generate completions



        message_lists,answers,messages,images= self.sample_1_response(
        model, 
        self.processing_class, 
        old_message_lists,  # 直接用rank作为索引
        images,
        )


        issimple = inputs[0]["is_simple"]
        completions = answers

        def accuracy_reward(completions,answers):
            rewards = []
            for completion,answer in zip(completions,answers):
                try:
                    if int(completion) == answer:
                        rewards.append(1.0)
                    else:
                        rewards.append(0.0)
                except:
                    rewards.append(0.0)
            return rewards
        answers = [inputs[i]['solution'] for i in range(len(inputs))]
        rewards = accuracy_reward(completions,answers)
        rewards = torch.tensor(rewards,dtype=torch.float32,device=self.accelerator.device)
        accuracy_name = "complex" if not issimple else "simple"

        self._metrics["eval"][f"{accuracy_name}"].append(self.accelerator.gather_for_metrics(rewards).mean().item())


        if self.accelerator.is_main_process:
            checkpoint_folder = f"checkpoint-{self.state.global_step}"
            output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
            os.makedirs(output_dir, exist_ok=True)
            sample_json = os.path.join(output_dir, f"samples_{accuracy_name}.json")
            if not os.path.exists(sample_json):
                with open(sample_json, 'w', encoding='utf-8') as f:
                    json.dump(messages, f, indent=4) 
                os.makedirs(os.path.join(output_dir,f"images_{accuracy_name}"), exist_ok=True)
                for i,image in enumerate(images):
                    for j,img in enumerate(image):
                        img.save(os.path.join(output_dir,f"images_{accuracy_name}",f"{i}_{j}.png"))
        return torch.tensor(0.0,device=self.accelerator.device)
 
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.eval_prediction_step(model,inputs)
            loss = loss.mean().detach()
        return loss, None, None
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {}
        for mode in ['train','eval']:
            metrics[mode] = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()} 
            if mode == "eval":
                metrics[mode] = {f"eval_{key}": val for key, val in metrics[mode].items()}
            self._metrics[mode].clear()
        logs = {**logs, **metrics['train'],**metrics['eval']}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)


    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))