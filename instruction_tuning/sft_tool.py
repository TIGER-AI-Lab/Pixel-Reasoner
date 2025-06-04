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


import os
os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '2'
import debugpy
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# if os.getenv("LOCAL_RANK") == "0":
# # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 5680))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# if os.getenv("LOCAL_RANK") == "1":
#     debugpy.listen(("localhost", 5681))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# if os.getenv("LOCAL_RANK") == "2":
#     debugpy.listen(("localhost", 5682))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()   
# if os.getenv("LOCAL_RANK") == "3":
#     debugpy.listen(("localhost", 5683))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# if os.getenv("LOCAL_RANK") == "4":
#     debugpy.listen(("localhost", 5684))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# if os.getenv("LOCAL_RANK") == "5":
#     debugpy.listen(("localhost", 5685))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# if os.getenv("LOCAL_RANK") == "6":
#     debugpy.listen(("localhost", 5686))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# if os.getenv("LOCAL_RANK") == "7":
#     debugpy.listen(("localhost", 5687))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()



from transformers import TrainingArguments

from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize


import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import glob
from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify

# from open_r1.trainer.grpo_toolconfig import GRPOToolConfig

from sft_tooltrainer import Qwen2VLSFTToolTrainer
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments
import yaml
import json
import random
import math
from datasets import load_dataset
# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple,Optional,Union


SEED=42
def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        # print(111, 222, 333, 444, 555, 666, 777, 888, 999)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward


# ----------------------- Main Script -----------------------
@dataclass
class SFTScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """
    train_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the training dataset"},
    )
    eval_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the evaluation dataset"},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    reward_funcs_for_show: list[str] = field(
        default_factory=lambda: ["use_gd"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )
    use_base_model: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use the base model"},
    )
    tools: list[str] = field(
        default_factory=lambda: ["annotate_image"],
        metadata={"help": "tools"},
    )
    weight: Optional[float] = field(
        default=1.0,
        metadata={"help": "Weight for the number token"},
    )
    freeze_vision_modules: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to freeze the vision modules"},
    )
    use_final_answer: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use the final answer tool"},
    )
    datasetpath: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the training dataset"},
    )


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

class SFT_DATASET(Dataset):
    def __init__(self, json_path: str):
        self.data = json.load(open(json_path, 'r'))


    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]

def main(script_args, training_args, model_args):


    model_id = model_args.model_name_or_path

    if "Qwen" in model_id:
        from vlm_modules.qwen_module import Qwen2VLModule
        vlm_module = Qwen2VLModule()

    else:
        raise ValueError(f"Unsupported model: {model_id}")




    # datapath = os.environ.get('datapath', "/home/ma-user/work/haozhe/muze/all_IV_noweb_15video_processed.json")
    datapath = script_args.datasetpath
    print(f"!!!!!! DATA loading from {datapath}")
    train_dataset = SFT_DATASET(datapath)




    trainer_cls =  Qwen2VLSFTToolTrainer

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        vlm_module=vlm_module,
        model=model_args.model_name_or_path,

        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        torch_dtype=model_args.torch_dtype,
        is_base_model=script_args.use_base_model,
        tools=script_args.tools,

        freeze_vision_modules=script_args.freeze_vision_modules,
        use_final_answer=script_args.use_final_answer
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    # groundingdino_model = GroundingDINO(model_config_path="/home/ma-user/work/haozhe/muze/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py", model_checkpoint_path="/home/ma-user/work/haozhe/muze/GroundingDINO/groundingdino_swinb_cogcoor.pth",text_threshold=0.25,box_threshold=0.25)
    parser = TrlParser((SFTScriptArguments, TrainingArguments, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
