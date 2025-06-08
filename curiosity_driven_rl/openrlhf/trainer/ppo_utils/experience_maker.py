import os
import time
from abc import ABC
from copy import deepcopy, copy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Dict

import ray
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from datasets import interleave_datasets, load_dataset
from openrlhf.models.actor import Actor
from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean, unpacking_samples
from openrlhf.utils.logging_utils import init_logger
# from openrlhf.trainer.ppo_utils.data_processor import add_pixel_bounds
from qwen_vl_utils import smart_resize, process_vision_info, extract_vision_info, fetch_image

from collections import defaultdict

import datasets
import json
# pip install math-verify
from math_verify import parse, verify
import pickle as pkl
import re 
from PIL import Image
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)


logger = init_logger(__name__)

def to_rgb(pil_image: Image.Image) -> Image.Image:
      if pil_image.mode == 'RGBA':
          white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
          white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
          return white_background
      else:
          return pil_image.convert("RGB")

@register_tool("select_frames")
class SelectFrames(BaseTool):
    @property
    def description(self):
        return """
Select frames from a video.
""".strip()

    parameters = {
        "type": "object",
        "properties": {
            "target_frames": {
                "type": "array",
                "description": "List of frame indices to select from the video (no more than 8 frames in total).",
                "items": {
                    "type": "integer",
                    "description": "Frame index from 1 to 16."
                }
            }
        },
        "required": ["target_frames"]
    }

    def call(self, images, target_frames):
        return [images[tgt] for tgt in target_frames]
    
# @register_tool("crop_image")
# class CropImage(BaseTool):
#     @property
#     def description(self):
#         return """
# Zoom in on the image based on the bounding box coordinates. 
# """.strip()

#     parameters = {
#         "type": "object",
#         "properties": {
#             "bbox_2d": {
#                 "type": "array",
#                 "description": "normalized coordinates for bounding box of the region you want to zoom in. Values should be within [0.0,1.0].",
#                 "items": {
#                     "type": "number",
#                 }
#             },
#             "target_image":{
#                 "type": "number",
#                 "description": "The index of the image to crop. Index from 1 to the number of images. Choose 1 to operate on original image."
#             }
#         },
#         "required": ["bbox_2d", "target_image"]
#     }

#     def call(self, image, bbox_2d,padding=(0.1,0.1)):
#         """
#         Crop the image based on the bounding box coordinates.
#         """
#         img_x, img_y = image.size
#         padding_tr = (600.0/img_x,600.0/img_y)
#         padding = (min(padding[0],padding_tr[0]),min(padding[1],padding_tr[1]))

#         if bbox_2d[0] < 1 and bbox_2d[1] < 1 and bbox_2d[2] < 1 and bbox_2d[3] < 1:
#             normalized_bbox_2d = (float(bbox_2d[0])-padding[0], float(bbox_2d[1])-padding[1], float(bbox_2d[2])+padding[0], float(bbox_2d[3])+padding[1])
#         else:
#             normalized_bbox_2d = (float(bbox_2d[0])/img_x-padding[0], float(bbox_2d[1])/img_y-padding[1], float(bbox_2d[2])/img_x+padding[0], float(bbox_2d[3])/img_y+padding[1])
#         normalized_x1, normalized_y1, normalized_x2, normalized_y2 = normalized_bbox_2d
#         normalized_x1 =min(max(0, normalized_x1), 1)
#         normalized_y1 =min(max(0, normalized_y1), 1)
#         normalized_x2 =min(max(0, normalized_x2), 1)
#         normalized_y2 =min(max(0, normalized_y2), 1)
#         cropped_img = image.crop((int(normalized_x1*img_x), int(normalized_y1*img_y), int(normalized_x2*img_x), int(normalized_y2*img_y)))
#         w, h = cropped_img.size
#         assert w > 28 and h > 28, f"Cropped image is too small: {w}x{h}"


#         return cropped_img  

@register_tool("crop_image_normalized")
class CropImageNormalized(BaseTool):
    @property
    def description(self):
        return """
Zoom in on the image based on the bounding box coordinates. It is useful when the object or text in the image is too small to be seen.
""".strip()

    parameters = {
        "type": "object",
        "properties": {
            "bbox_2d": {
                "type": "array",
                "description": "coordinates for bounding box of the area you want to zoom in. Values should be within [0.0,1.0].",
                "items": {
                    "type": "number",
                }
            },
            "target_image":{
                "type": "number",
                "description": "The index of the image to crop. Index from 1 to the number of images. Choose 1 to operate on original image."
            }
        },
        "required": ["bbox_2d", "target_image"]
    }
    
    def call(self, image, bbox_2d,  padding=0.1):
        """
        Crop the image based on the bounding box coordinates.
        """
        img_x, img_y = image.size
        if bbox_2d[0] < 1 and bbox_2d[1] < 1 and bbox_2d[2] < 1 and bbox_2d[3] < 1:
            normalized_bbox_2d = (float(bbox_2d[0])-padding, float(bbox_2d[1])-padding, float(bbox_2d[2])+padding, float(bbox_2d[3])+padding)
        else:
            normalized_bbox_2d = (float(bbox_2d[0])/img_x-padding, float(bbox_2d[1])/img_y-padding, float(bbox_2d[2])/img_x+padding, float(bbox_2d[3])/img_y+padding)
        normalized_x1, normalized_y1, normalized_x2, normalized_y2 = normalized_bbox_2d
        normalized_x1 =min(max(0, normalized_x1), 1)
        normalized_y1 =min(max(0, normalized_y1), 1)
        normalized_x2 =min(max(0, normalized_x2), 1)
        normalized_y2 =min(max(0, normalized_y2), 1)
        cropped_img = image.crop((normalized_x1*img_x, normalized_y1*img_y, normalized_x2*img_x, normalized_y2*img_y))
        w, h = cropped_img.size
        assert w > 28 and h > 28, f"Cropped image is too small: {w}x{h}"



        return cropped_img 
    

def extract_qwen_query_and_response(input_text):
    # Split the input text by the assistant's start token
    parts = input_text.split("<|im_start|>assistant\n")
    
    # The first part contains the system and user messages
    user_part = parts[0]
    
    # The second part contains the assistant's response
    if len(parts)==1: assistant_response = ""
    else: assistant_response = "".join(parts[1:])
    
    # Extract the user query by splitting the user part
    user_query = user_part.split("<|im_start|>user\n")[1].split('<|im_end|>')[0].split('<|vision_end|>')[-1]
    
    # Return the user query and the assistant's response
    return user_query, assistant_response


def extract_dsmath_query_and_response(input_text):
    # Split the input text by the assistant's start token
    parts = input_text.split("Assistant:")
    
    # The first part contains the system and user messages
    user_part = parts[0]
    
    # The second part contains the assistant's response
    if len(parts)==1: assistant_response = ""
    else: assistant_response = parts[1]
    
    # Extract the user query by splitting the user part
    user_query = user_part.split("User:")[1].strip()
    
    # Return the user query and the assistant's response
    return user_query, assistant_response


def extract_dpsk_query_and_response(input_text):
    # Split the input text by the assistant's start token
    # print(input_text)
    parts = input_text.split("<｜Assistant｜>")
    
    # The first part contains the system and user messages
    if len(parts)==0:
        print('!!!! warning extraction', input_text)
    user_part = parts[0]
    
    # The second part contains the assistant's response
    if len(parts)==1: assistant_response = ""
    else: assistant_response = parts[1]
    
    # Extract the user query by splitting the user part
    user_query = user_part.split("<｜User｜>")[1]
    
    # Return the user query and the assistant's response
    return user_query, assistant_response

def extract_llama_query_and_response(input_text):
    # Split the input text by the assistant's start token
    parts = input_text.split("assistant<|end_header_id|>\n\n")
    
    # The first part contains the system and user messages
    user_part = parts[0]
    
    # The second part contains the assistant's response
    if len(parts)==1: assistant_response = ""
    else: assistant_response = parts[1]
    
    # Extract the user query by splitting the user part
    user_query = user_part.split("user<|end_header_id|>\n\n")[1].split('<|eot_id|><|start_header_id|>')[0]
    
    # Return the user query and the assistant's response
    return user_query, assistant_response

def extract_autocode_query_and_response(input_text):
    # print('!!!! example input', input_text)
    # Split the input text by the assistant's start token
    parts = input_text.split("Response:")
    
    # The first part contains the system and user messages
    user_part = parts[0]
    
    # The second part contains the assistant's response
    if len(parts)==1: assistant_response = ""
    else: assistant_response = parts[1]
    
    # Extract the user query by splitting the user part
    user_query = user_part.split("### Instruction:\n")[1].split('\n\n### ')[0]
    
    # Return the user query and the assistant's response
    return user_query, assistant_response
    
def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advantages: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    kl: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]
    kl: Optional[torch.Tensor] = None
    visual_inputs: Optional[dict] = field(default_factory=dict)
    validity: Optional[torch.Tensor] = None

    @torch.no_grad()
    def to_device(self, device: torch.device):
        self.sequences = to(self.sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        self.returns = to(self.returns, device)
        self.advantages = to(self.advantages, device)
        self.values = to(self.values, device)
        self.attention_mask = to(self.attention_mask, device)
        self.action_mask = to(self.action_mask, device)
        self.kl = to(self.kl, device)
        self.info = {key: to(value, device) if isinstance(value, torch.Tensor) else value for key, value in self.info.items()}
        if self.visual_inputs is not None:
            self.visual_inputs = {key: to(value, device) for key, value in self.visual_inputs.items()}
        if self.validity is not None: 
            self.validity = to(self.validity, device)
        return self

    def pin_memory(self):
        self.sequences = pin_memory(self.sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        self.returns = pin_memory(self.returns)
        self.advantages = pin_memory(self.advantages)
        self.values = pin_memory(self.values)
        self.attention_mask = pin_memory(self.attention_mask)
        self.action_mask = pin_memory(self.action_mask)
        self.kl = pin_memory(self.kl)
        self.info = {key: pin_memory(value) if isinstance(value, torch.Tensor) else value for key, value in self.info.items()}
        if self.visual_inputs is not None:
            self.visual_inputs = {key: pin_memory(value) for key, value in self.visual_inputs.items()}
        if self.validity is not None: 
            self.validity = pin_memory(self.validity)
        return self


@dataclass
class Samples:
    """Samples is a batch of data.
    There can be 2 formats to store the samples, batched or packed.
    The batched format means padding is applied to the sequences, while the packed format
    will concatenate the prompt and response without padding.

    Shapes of each tensor, when 2 shapes are shown, the first one is for batched format
        and the second one is for packed format:
    sequences: (B, S) or (1, total_length), the tokens of both prompt and response.
    attention_mask: (B, S) or (1, total_length), the attention mask for sequences.
    action_mask: (B, A) or None, the action (response) mask to show which part of the
        sequence is the response. When the samples are packed, this is None.
    num_actions: int or (B,), the number of actions (tokens) in the response.
        When the samples are not packed, we will use action_mask, so this is an int to
        show the size of action_mask. Otherwise, this is a tensor to show the number of
        actions for each sample.
    packed_seq_lens: None or (B,), the length of each sample in the packed samples.
    response_length: (B,), the number of tokens in the response.
    total_length: (B,), the total number of tokens in the sequences.
    prompts: the prompts used to generate responses
    visual_inputs: the visual input for vlm training
    """

    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    prompts: list[str]
    visual_inputs: Optional[Dict]
    na_each: list[int]
    round0_correctness: list 
    round1_correctness: list 
    round0_nwait: list[int]
    round1_nwait: list[int]
    questions: list[str]
    solutions: list[str]
    qids: list[str]
    round0_ALLTrue: list[float]
    round0_Easy: list[float]
    round0_Medium: list[float]
    round0_Hard: list[float]
    round0_ALLFalse: list[float]
    efficiency_label: list[float]
    shaped_rewards: list[float]
    uniformity: list[float]
    curiosity_bonus: list[float]
    penalty_bonus: list[float]
    # round0_saturation: list[float]

def get_raw(modelfamily, text):
    if modelfamily=='dpsk':
        user = text.split("<｜Assistant｜>")[0].split("<｜User｜>")[1]
        return user
    
class NaiveExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        data_processor,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: list[str] = None,
        reward_fn=None,
        modelfamily='qwen',
        gt_path=None
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.data_processor = data_processor
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn
        self.perf_stats = None
        self.advantage_estimator = strategy.args.advantage_estimator
        self.gt_path = gt_path
        self.modelfamily = modelfamily
        # custom reward func for reinforced finetuning
        self.custom_reward_func = None
        if remote_rm_url and remote_rm_url[0].endswith(".py"):
            print(f"Loading custom `reward_func(queries, prompts)` from {remote_rm_url[0]}")
            import importlib.util

            spec = importlib.util.spec_from_file_location("reward_func", remote_rm_url[0])
            reward_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reward_module)
            self.custom_reward_func = reward_module.reward_func

    # tokenizer
    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience_list(self, all_prompts: Union[str, List[str]], **generate_kwargs) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        pass 

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.
        """
        assert not getattr(self, "packing_samples", False)
        args = self.strategy.args
        self.actor.eval()
        # sample multiple response
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        samples_list = []
        for i in range(0, len(all_prompts), args.micro_rollout_batch_size):
            prompts = all_prompts[i : i + args.micro_rollout_batch_size]
            if self.data_processor is not None:
                inputs = self.data_processor(prompts, self.prompt_max_len, device="cuda")
                visual_inputs = {}
                for k,v in inputs.items():
                    if k not in ["input_ids", "attention_mask"]:
                        visual_inputs[k] = v
            else:
                inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
                visual_inputs = None

            sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
            samples = Samples(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                packed_seq_lens=None,
                response_length=action_mask.float().sum(dim=-1),
                total_length=attention_mask.float().sum(dim=-1),
                prompts=prompts,
                visual_inputs=visual_inputs,
            )
            samples_list.append(samples)
        return samples_list

    @torch.no_grad()
    def get_logprobs_and_logs(self, samples: Samples) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        pass

    @torch.no_grad()
    def handle_advantages(self, experiences: List[Experience], nsample=None) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.

        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        if nsample is None:
            nsample = args.n_samples_per_prompt
        args = self.strategy.args
        print(f"===> [verbose] handling advantages in NaiveEMaker handle_advantages()")
        do_longer = getattr(args, "format", "none") == 'longer'
        tmp = [experience.info["reward"] for experience in experiences]
        ns =  [experience.info["response_length"]  for experience in experiences]
        match = [experience.info["match"]  for experience in experiences]
        ns = np.array(ns).reshape((-1, nsample)) 
        match = np.array(match).reshape((-1, nsample)) 
        ns_diff = []
        for idx, match_i in enumerate(match): 
            # when there is no correct 
            if np.sum(match_i)==0: 
                ns_diff.append(np.zeros_like(ns[idx]))
            else: 
                mean_switch = np.sum(ns[idx] * match_i)/np.sum(match_i) # average length of correct response 
                len_adv = (ns[idx]-mean_switch)*(match_i>0.5) # positive values of longer 
                max_adv = abs(max(len_adv)) # right delta
                min_adv = abs(min(len_adv)) # left delta 
                # if min_adv<0: min_adv = -min_adv
                len_adv[len_adv>0] /= max_adv # normalized to [-1.0, 1.0]
                len_adv[len_adv<0] /= min_adv
                ns_diff.append(len_adv)
                # tmplist = []
        ns_diff = np.stack(ns_diff)
        bonus = np.clip(ns_diff * 1.0, -0.499, 0.499)
        num = len(experiences)
        bonus_flat = bonus.reshape((num, -1))
        for idx, exp in enumerate(experiences):
            exp.info["wait_bonus"] = bonus_flat[idx].tolist()
        print(f'!!!! [rbuffer] The estimator {args.advantage_estimator} is processing {len(experiences)} queries in a batch, each {len(tmp[0])} responses, longer={do_longer}')
        # reward shaping for RLOO
        if args.advantage_estimator in ["rloo","gloo","rloo_sft"]: # this operates in batch level
            rewards = torch.cat(tmp) 
            rewards = rewards.reshape(-1, nsample).to(device="cuda")  # (bsz,nsample) into groups
            
            if do_longer:
                bonus_tensor = torch.from_numpy(bonus).to(rewards.device).to(rewards.dtype)
                rewards += bonus_tensor
                # print('!!!! shaped reward', rewards.detach().cpu().numpy())
            else:
                print('!!!! [rbuffer] reward not using wait')
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (nsample - 1) # mean of others 
            rewards = rewards - baseline
            rewards = rewards.flatten().to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        elif args.advantage_estimator == "reinforce_baseline":
            # REINFORCE++-baseline removed the / std and K3 kl loss in GRPO.
            # `/ std` is not needed in RL variance reduction theory, and `k3 KL` has a larger variance than `k1 KL` under a categorical distribution.
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, nsample).to(device="cuda")
            rewards = rewards - rewards.mean(-1, keepdim=True)
            rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        elif args.advantage_estimator in ["group", "group_sft"]: # this operates in batch level
            rewards = torch.cat(tmp) 
            rewards = rewards.reshape(-1, nsample) # .to(device="cuda")  # (bsz,nsample) into groups
            raw_r = rewards.detach().numpy() # bsz,nsamples 
            mean_acc = np.tile(raw_r.mean(-1, keepdims=True), (1,nsample))
            solve_all = mean_acc>0.95
            solve_none = mean_acc<0.05
            easy = mean_acc>0.7
            hard = mean_acc<0.35
            medium = np.logical_not(np.logical_or(easy, hard))
            
            difficulty, solve_all, solve_none, easy, hard, medium = [x.reshape((len(experiences), -1)).astype(float) for x in [mean_acc, solve_all, solve_none, easy, hard, medium]]
            all_waits = []
            all_waits0 = []
            is_native = []
            t1_diff = []
            for iidx, exp in enumerate(experiences):
                exp.info['difficulty'] = difficulty[iidx].tolist()
                exp.info['solve_all'] = solve_all[iidx].tolist()
                exp.info['solve_none'] = solve_none[iidx].tolist()
                exp.info['easy'] = easy[iidx].tolist()
                exp.info['hard'] = hard[iidx].tolist()
                exp.info['medium'] = medium[iidx].tolist()
                all_waits.extend(exp.info['round1_nwait'])
                all_waits0.extend(exp.info['round0_nwait'])
                t1_cor = exp.info['round1_correctness']
                t0_cor = exp.info['round0_correctness']
                is_native.extend([float(x is None) for x in t1_cor])
                t1_diff.extend([-5.0 if x<0 else x-y for x,y in zip(t1_cor, t0_cor) ]) # x=-1 if no rethinking
                # print('!!!! [debug] solve status', exp.info['solve_all'], raw_r.mean(-1))
            reshaped_nwait_round1 = np.array(all_waits).reshape((len(rewards), -1))
            reshaped_nwait_round0 = np.array(all_waits0).reshape((len(rewards), -1))
            reshaped_is_native = np.array(is_native).reshape((len(rewards), -1))
            reshaped_t1_diff = np.array(t1_diff).reshape((len(rewards), -1))
            # all_waits = (np.logical_and(reshaped_nwait_round1>0, reshaped_nwait_round1<=2)).astype(float)
            # too_many_waits = (reshaped_nwait_round1>2).astype(float)
            # all_waits = torch.from_numpy(all_waits).to(rewards.device)
            baseline = rewards.sum(-1, keepdim=True) / (nsample) # mean of others 
            rewards = rewards - baseline
            
            # for iidx in range(len(rewards)):
            #     if reshaped_nwait_round1[iidx].sum()>0: 
            #         # if advantage>0.125, meaning this is informative positive example 
            #         # - if it has native wait or keep into a correct response, we praise it
            #         isnative = reshaped_is_native[iidx]>0.5
            #         ischeck = reshaped_t1_diff[iidx]==0.0
            #         notmanywaits = reshaped_nwait_round1[iidx]<4.0
            #         old = rewards[iidx].cpu().numpy()
            #         oldstr = str(old)
            #         # praise = torch.ones_like(rewards[iidx])
            #         # praiseflag = np.logical_and(notmanywaits,np.logical_or(isnative, ischeck))
            #         praiseflag = np.logical_and(notmanywaits,np.logical_or(isnative, ischeck))
            #         flag = False
            #         # print(f"[debug] native={isnative}, {reshaped_is_native[iidx]}, check={ischeck}, {reshaped_t1_diff[iidx]}, wait={notmanywaits}, {reshaped_nwait_round1[iidx]}")
            #         for ii,(rvalue,pflag,wflag,macc) in enumerate(zip(rewards[iidx], praiseflag, notmanywaits, baseline[iidx])):
                        ################
                        # if pflag and rvalue>0.: # correct and native wait
                        #     rewards[iidx][ii] = rvalue * 1.5
                        #     flag = True
                        # elif rvalue<0.0 and wflag:
                        #     rewards[iidx][ii] = rvalue * 0.5
                        #     flag = True 
                        #################
                        # if macc>0.98:
                        #     rewards[iidx][ii] = 1.0/8.0
                            
                    # scale_factor = (rewards[iidx]> 0.1 ).to(float) * praise 
                    # mask = 1.0+scale_factor # torch.tensor 
                    # - if a incorrect has a bit of rethinking try, we praise it 
                    # selector = torch.BoolTensor(notmanywaits).to(rewards.device)
                    # mask[selector] = 1.0 - 0.5*(rewards[iidx] < 0.0).to(float)
                    # new = rewards[iidx] * mask
                    # - if a incorrect has a bit of rethinking try, we praise it 
                    # scale_factor = (rewards[iidx]> 0.1 ).to(float) * all_waits[iidx] # (nsamples,) is positive and has wait? 
                    # if too_many_waits[iidx].sum()>0:
                    #     for kk, entry in enumerate(too_many_waits[iidx]):
                    #         if entry>0.5: 
                    #             scale_factor[kk] = -1.
                    #             print('!!!! [debug] too many waits supressed.')
                    # old = rewards[iidx].cpu().numpy()
                    # new = rewards[iidx] * (1.0+scale_factor) # positive examples with wait will get x2 advantages
                    # if flag: print(f'!!!! [debug] {oldstr} ->{rewards[iidx].cpu().numpy()}')
                    # rewards[iidx] = new
            
            if do_longer:
                print('!!!! length bonus', bonus)
                bonus_tensor = torch.from_numpy(bonus).to(rewards.device).to(rewards.dtype).reshape(rewards.shape)
                rewards  = (bonus_tensor+1)*rewards
            # else:
                # print('!!!! not using wait')
            rewards = rewards.flatten().to(device="cpu").chunk(len(experiences)) # num_exp, 
            return experiences, rewards
        # default rewards
        return experiences, tmp

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        if isinstance(values, list):
            # packing samples
            # TODO: this is slow...
            advantages = []
            returns = []
            for v, r in zip(values, rewards):
                adv, ret = self.get_advantages_and_returns(v.unsqueeze(0), r.unsqueeze(0), action_mask, gamma, lambd)
                advantages.append(adv.squeeze(0))
                returns.append(ret.squeeze(0))
            return advantages, returns

        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """

        if isinstance(rewards, list):
            # packing samples
            # TODO: this is slow...
            returns = []
            for r in rewards:
                ret = self.get_cumulative_returns(r.unsqueeze(0), action_mask, gamma)
                returns.append(ret.squeeze(0))
            return returns

        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns

def regularize_text(x):
    trigger = "Please reason step by step, and put your final answer within \\boxed{}."
    x = x.split(trigger)[0]
    return x.strip().replace(' ','')

def do_verify(nsol, b):
    res = 0.0 
    try:
        a = parse(nsol)
        if len(b)>1 and (b[1] in 'ABCDEFGHIJK'):
            res = float(nsol[len("\\boxed{"):].startswith(b[1]))
        else:
            # print(f"debug parsed: {a} from {nsol} and {b}")
            if len(a)==0: res = -1.0 
            else: res = float(verify(a, b))
    except: 
        print(f"!!!! [debug] {nsol} parsing exception")
        res = -1.0 
    return res 

ans_indicator = "answer is"
endstr = "Now everything looks fine. Solution finished."
def normalize_answer(answer):
    if answer is None: return answer
    if 'dfrac' in answer: answer = answer.replace("dfrac", "frac")
    # if '%' in answer: answer = answer.replace(r'\%',"").replace('%',"")
    if 'text' in answer: answer = answer.replace("\\text","")
    if "\\varnothing" in answer: answer = answer.replace("\\varnothing","\\emptyset")
    if "minutes" in answer: answer = answer.replace("minutes","")
    if "cm" in answer: answer = answer.replace("cm","")
    # if "^\\circ" in answer: answer = answer.replace("^\\circ","")
    # if "a.m." in answer: answer = answer.replace("a.m.","")
    return answer 

def handle_boxed(sol, gt, eostoken, format_type, requires_box=False):
    # print(sol)
    # print('!!!! debug', gt)
    norepeat = None
    usefmt = None
    res = 0.0
    # endstr and eos token
    index = sol.find(endstr)
    num_end = len(eostoken)
    
    if index>-1: 
        remains = sol[index+len(endstr)+num_end:]
        if len(remains)>0: 
            norepeat = False 
        else: norepeat = True 
    if not (norepeat is False):
        if format_type in ["confidence"]:
            if not ("<confidence>" in sol and "</confidence>" in sol):
                usefmt = False
            else: 
                count = sol.count("<confidence>")
                if count>5: usefmt = False 
                else: usefmt = True 
        elif format_type in ["wait"]:
            tmps = sol.lower()
            usefmt = False 
            if "wait" in tmps or "alternatively" in tmps:
                usefmt = True 
            
        # elif format_type in ["nocode"]:
        #     if "```python" in sol:
        #         usefmt = False 
        #     else: usefmt = True 
    
    if (norepeat is False): 
        pass # no need for correctness 
    else: 
        flag = True 
        gt = normalize_answer(gt)
        try:
            if "\\boxed" in gt: 
                b = parse(gt)
                # print('!!!! debug gt parse', gt, b)
            else:
                b = parse(f"\\boxed{{{gt}}}")
        except Exception as e:
            print(f"!!!! [debug] {gt} parsing exception")
            res = -1.0
            flag = False 
        if flag:
            if len(b)==0: res = -1.0 
            else: 
                if requires_box:
                    boxed_index = sol.rindex("boxed")
                    if boxed_index==-1: res = 0.0 
                    else:
                        nsol = '\\'+sol[boxed_index:]
                        res = do_verify(normalize_answer(nsol), b)
                else: 
                    flag = False 
                    
                    for indicator in ["\\boxed", "<answer>", "Answer:"]:
                        if indicator in sol:
                            if indicator == "<answer>":
                                found = re.search("<answer>(.*?)</answer>", sol)
                                if found:
                                    nsol = f"\\boxed{{{found.group(1)}}}"
                                else: continue 
                            elif indicator == "Answer:":
                                tmp = sol.split(indicator)
                                if len(tmp)>0: tmp = tmp[-1].strip() # .split(eostoken)[0].strip()
                                else: continue 
                                nsol = f"\\boxed{{{tmp}}}"
                            else: 
                                boxed_index = sol.rindex(indicator)
                                pred = sol[boxed_index:].strip()
                                nsol = pred
                            res = do_verify(normalize_answer(nsol), b)
                            if res > 0.99: 
                                flag = True 
                        if flag: 
                            break
                    # print("extracted sol", nsol)
                    if not flag:
                        nsol = sol 
                        res = do_verify(normalize_answer(nsol), b)
        
                    
    return norepeat, usefmt, res 

def rule_reward(sol, gt, eostoken, format_type, requires_box, *args):
    # valid = eos & boxed 
    error_info = None 
    valid = True 
    if eostoken not in sol and "<|endoftext|>" not in sol: 
        valid = False
        error_info = "No eos."
    elif requires_box and "boxed" not in sol:
        valid = False 
        error_info = "No valid boxed."
    elif sol.lower().count("wait")>5:
        valid = False
        error_info = "too many waits"
    ############ this is only for debugging 
    # if format_type=='wait':
    # tmps = sol.lower() 
    # if "wait" not in tmps and "alternatively" not in tmps:
    #     valid = False 
    
    # formats and correctness 
    norepeat = None
    usefmt = None
    res = 0.0
    # directly making no-boxed and no-eos as invalid seems unnecessary and harmful:
    # the model needs to understand these are unacceptable
    # if not valid: 
    #     pass 
    # else: 
    if isinstance(gt, list):
        gt = [xx.lower() for xx in gt]
        has_percent = None
        for xx in gt:
            if "%" in xx: 
                has_percent = xx.replace('%','')
                break
        if has_percent is not None and has_percent not in gt:
            gt.insert(0, has_percent)
    else:
        gt = [str(gt).lower()]
    tmpsol = sol.lower()
    # special_char = {'%'}
    for ans in gt:
        norepeat, usefmt, res = handle_boxed(tmpsol, ans, eostoken, format_type, requires_box=requires_box)
        if res>0.5:
            break
        else:
            # handle multi-span: troubles, the siege of krishnapur
            is_multispan = ',' in ans 
            
            if is_multispan:
                splits = [x.lower().strip() for x in ans[len("\\boxed{"):-1].split(',')]
                cnt = 0 # this could overestimate
                boxed_index = tmpsol.rfind("boxed")
                if boxed_index==-1: continue
                else:
                    nsol = '\\'+tmpsol[boxed_index:]
                    for sp in splits:
                        if sp in nsol:
                            cnt += 1
                    if cnt==len(splits):
                        res = 1.0
                        break 
    return valid, norepeat, usefmt, error_info, res 


def batch_rule_reward(sols, gts, eostoken, format_type, *args):
    rets = []
    for sol, gt in zip(sols,gts):
        rets.append(rule_reward(sol, gt, eostoken, format_type, *args))
    return rets


def find_last_code_block(text):
    # Define the regex pattern to match code blocks enclosed with ```python and ```
    pattern = r'```python(.*?)```'
    
    # Reverse the text and the pattern
    reversed_text = text[::-1]
    
    reversed_pattern = r'```(.*?)nohtyp```'
    
    # Search for the reversed pattern in the reversed text
    match = re.search(reversed_pattern, reversed_text, re.DOTALL)
    
    if match:
        # Extract the matched group, reverse it back to get the original code block
        reversed_code_block = match.group(1).strip()
        code_block = reversed_code_block[::-1]
        return code_block
    else:
        return None


def rule_reward_with_code(sol, gt, eostoken, format_type, executor):
    error_info = None 
    # valid = eos & boxed 
    valid = True 
    # formats and correctness 
    norepeat = None
    usefmt = None
    res = 0.0
    if eostoken not in sol: 
        valid = False
        return valid, norepeat, usefmt, error_info, res 
    if "```python" in sol:
        code = find_last_code_block(sol)
        if code is None: # no code found 
            valid = False 
            error_info = "No valid code block."
            return valid, norepeat, usefmt, error_info, res 
        pred, error_info = executor.apply(code)
        if error_info=='Done':
            try:
                b = parse(f"\\boxed{{{gt}}}")
                
                nsol = '\\boxed{'+pred+'}'
                a = parse(nsol)
                
                if len(a)==0: res = -1.0 
                else: res = float(verify(a, b))
            except:
                res = -1.0 
            error_info += f": {pred}"
        else: res = 0.0 
        # print(res, pred, error_info)
    else:
        if "boxed" not in sol:
             valid = False 
             return valid, norepeat, usefmt, "No valid boxed.", res 
        
        norepeat, usefmt, res = handle_boxed(sol, gt, eostoken, format_type)
    return valid, norepeat, usefmt, error_info, res 

def batch_rule_reward_with_code(sols, gts, eostoken, format_type, executor, requires_box=False):
    rets = []
    codes, code_i = [],[]
    # print('!!!! inside reward requires box', requires_box)
    for ii,(sol,gt) in enumerate(zip(sols, gts)):
        error_info = None 
        # valid = eos & boxed 
        valid = True 
        # formats and correctness 
        norepeat = None
        usefmt = None
        res = 0.0
        usecode = None 
        if eostoken not in sol: 
            valid = False
            ret = valid, norepeat, usefmt, error_info, usecode, res 
            rets.append(ret)
            # print('!!!! not valid: no eos', sol)
            continue
        if "```python" in sol:
            code = find_last_code_block(sol)
            usecode = True 
            if code is None: # no code found 
                valid = False 
                # print('!!!! not valid: no code', sol)
                error_info = "No valid code block."
                ret = valid, norepeat, usefmt, error_info, usecode, res 
                rets.append(ret)
                continue 
            codes.append(code)
            code_i.append(ii)
            ret = valid, norepeat, usefmt, error_info, usecode, res 
            rets.append(ret)
            continue 
            
        else:
            usecode = False 
            if requires_box and ("boxed" not in sol):
                valid = False 
                ret = valid, norepeat, usefmt, "No valid boxed.", usecode, res 
                rets.append(ret)
                continue 
            
            norepeat, usefmt, res = handle_boxed(sol, gt, eostoken, format_type, requires_box=requires_box)
            ret = valid, norepeat, usefmt, error_info, usecode, res 
            rets.append(ret)
            continue 
        
        if format_type in ['nocode']:
            if '```python' in sol: usefmt = False 
            else: usefmt = True
        
    #####
    if len(codes)>0:
        tmp = [executor.apply(c) for c in codes]
        preds, error_infos = list(zip(*tmp))
        for ii,code,pred,error_info in zip(code_i,codes,preds,error_infos):
            if error_info=='Done':
                flag = True 
                try:
                    gt = gts[ii]
                    b = parse(f"\\boxed{{{gt}}}")
                except:
                    res = -1.0 
                    flag = False 
                if flag: 
                    nsol = pred
                    res = do_verify(nsol, b)
                error_info += f": {pred}"
            else: res = 0.0 
            valid, norepeat, usefmt, _, usecode, _ = rets[ii]
            rets[ii] = valid, norepeat, usefmt, error_info, usecode, res 
    
    return rets
        
def prepare_target(prompt, eos_token):
    if "</think>" in prompt: 
        tmp = prompt.split("</think>")[0]+"</think>" 
        # print('!!!! prepare', [tmp])
        return tmp + eos_token
    else: return prompt 
    
replacewith = "<|vision_start|><|image_pad|><|vision_end|>"

def handle_placeholders(texts):
    newlist = []
    placeholder = "<image>"
    # placeholder2 = "<image1>"
    
    for m in texts:
        new = m 
        for k in ["<|vision_start|>","<|image_pad|>","<|vision_end|>"]:
            new = new.replace(k,"")
        # now new has no replacewith 
        if new.count(placeholder)>0:
            new = new.replace(placeholder, replacewith)
        else: 
            new = replacewith + new
        newlist.append(new)
    return newlist

tool_end = '</tool_call>'
tool_start = '<tool_call>'

def parse_last_tool(output_text):
    # print([output_text])
    return json.loads(output_text.split(tool_start)[-1].split(tool_end)[0])


def crop_image_normalized(image, bbox_2d,  padding=0.1):
    """
    Crop the image based on the bounding box coordinates.
    """
    img_x, img_y = image.size
    if bbox_2d[0] < 1 and bbox_2d[1] < 1 and bbox_2d[2] < 1 and bbox_2d[3] < 1:
        normalized_bbox_2d = (float(bbox_2d[0])-padding, float(bbox_2d[1])-padding, float(bbox_2d[2])+padding, float(bbox_2d[3])+padding)
    else:
        normalized_bbox_2d = (float(bbox_2d[0])/img_x-padding, float(bbox_2d[1])/img_y-padding, float(bbox_2d[2])/img_x+padding, float(bbox_2d[3])/img_y+padding)
    normalized_x1, normalized_y1, normalized_x2, normalized_y2 = normalized_bbox_2d
    normalized_x1 =min(max(0, normalized_x1), 1)
    normalized_y1 =min(max(0, normalized_y1), 1)
    normalized_x2 =min(max(0, normalized_x2), 1)
    normalized_y2 =min(max(0, normalized_y2), 1)
    cropped_img = image.crop((normalized_x1*img_x, normalized_y1*img_y, normalized_x2*img_x, normalized_y2*img_y))
    w, h = cropped_img.size
    assert w > 28 and h > 28, f"Cropped image is too small: {w}x{h}"
    return cropped_img 

do_controlled_rectify = True
def execute_tool(images, rawimages, args, toolname, is_video, function=None):
    if toolname=='select_frames':
        tgt = args['target_frames']
        if len(tgt)>8:
            
            message = f"You have selected {len(tgt)} frames in total. Think again which frames you need to check in details (no more than 8 frames)"
            # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            ##### controlled modification
            if do_controlled_rectify and np.random.uniform()<0.75:
                if np.random.uniform()<0.25:
                    tgt = tgt[:len(tgt)//2]
                elif np.random.uniform()<0.25/0.75:
                    tgt = tgt[-len(tgt)//2:]
                elif np.random.uniform()<0.25/0.5:
                    tgt = tgt[::2]
                else:
                    tgt = np.random.choice(tgt, size=len(tgt)//2, replace=False)
                    tgt = sorted(tgt)
                selected_frames = function(images[0], tgt)
                message = tgt
            else: 
                selected_frames = []
            # selected_frames = function(images[0], [x-1 for x in tgt][::2]) # video is always in the first item
        elif max(tgt)>len(images[0]):
            message = f"There are {len(images[0])} frames numbered in range [1,{len(images[0])}]. Your selection is out of range."
            selected_frames = []
        else:
            message = ""
            candidates = images[0]
            if not isinstance(candidates, list):
                candidates = [candidates]
            selected_frames = function(candidates, [x-1 for x in tgt]) # video is always in the first item
        return selected_frames, message
    else:
        tgt = args['target_image']
        if is_video:
            if len(images)==1: # there is only 
                # we default the candidate images into video frames 
                video_frames = images[0]
                index = tgt - 1 
                assert index<len(video_frames), f"Incorrect `target_image`. You can only select frames in the given video within [1,{len(video_frames)}]"
                image_to_crop = video_frames[index]
            else: # there are zoomed images after the video; images = [[video], img, img, img]
                cand_images = images[1:]
                index = tgt -1
                assert index<len(cand_images), f"Incorrect `target_image`. You can only select a previous frame within [1,{len(cand_images)}]"
                image_to_crop = cand_images[index]
        else:
            index =  tgt-1 
            assert index<len(images), f"Incorrect `target_image`. You can only select previous images within [1,{len(images)}]"
            
            if index<len(rawimages):
                tmp = rawimages[index]
            else:
                tmp = images[index]
            image_to_crop = tmp
        if function is None: function = crop_image_normalized
        cropped_image = function(image_to_crop, args['bbox_2d'])
    return cropped_image


def get_required_messages(messages):
    conversations_list = [json.loads(mm) if isinstance(mm,str) else mm for mm in messages]
    final = []
    for conversations in conversations_list:
        message_list = [Message(role="system", content=[ContentItem(text="You are a helpful assistant.")])]
        
        for entry in conversations:
            role = entry['role']
            
            content = entry['content']
            contlist = []
            for cont in content:
                if cont['type'] == 'text':
                    contlist.append(ContentItem(text="{Question}".format(Question=cont['text'])))
                elif cont['type'] in {'image','video'}:
                    key = cont['type']
                    contlist.append(ContentItem(image=cont[key] ) if key=='image' else ContentItem(video=cont[key] ))
            message_list.append(Message(role=role, content=contlist))
        final.append(message_list)
    return final

def get_prompt_from_messages(oldformat_messages, prompt_maker, tools, processor):
    messages = get_required_messages(oldformat_messages)
    if len(tools)>0:
        messages = [prompt_maker.preprocess_fncall_messages(
            messages=msg,
            functions=tools, 
            lang=None
        ) for msg in messages]

    messages = [[x.model_dump() for x in conversations] for conversations in messages]
    
    prompts = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    return prompts, messages
    
    
def create_action_mask_up_to_last_eos(
    sequences: torch.Tensor,
    eos_token_id: int,
    pad_token_id: int
) -> torch.Tensor:
    """
    Creates a mask that is True for tokens up to and including the last eos_token_id,
    excluding pad_token_id and tokens after the last eos_token_id.

    IMPORTANT ASSUMPTION: This function assumes every sequence in the batch
    contains at least one eos_token_id. Behavior is undefined if this is not met.

    Args:
        sequences: Tensor of token IDs (batch_size, sequence_length).
        eos_token_id: The ID of the end-of-sequence token.
        pad_token_id: The ID of the padding token.

    Returns:
        A boolean tensor mask of the same shape as sequences.
    """
    if sequences.ndim != 2:
        raise ValueError("sequences tensor must be 2D (batch_size, sequence_length)")

    device = sequences.device
    batch_size, seq_len = sequences.shape

    # 1. Find the index of the last occurrence of eos_token_id
    # Since we assume EOS always exists, argmax on the reversed sequence
    # will correctly identify the position relative to the end.
    is_eos = (sequences == eos_token_id)
    reversed_is_eos = torch.flip(is_eos, dims=[1])
    first_eos_in_reversed_idx = torch.argmax(reversed_is_eos.int(), dim=1)
    last_eos_idx = seq_len - 1 - first_eos_in_reversed_idx # Index in original sequence

    # 2. Create a mask based on position relative to the last EOS index
    col_indices = torch.arange(seq_len, device=device).unsqueeze(0) # Shape: (1, seq_len)
    # True for indices <= last_eos_idx. Shape: (batch_size, seq_len)
    position_mask = col_indices <= last_eos_idx.unsqueeze(1)

    # 3. Create a mask for non-padding tokens
    not_pad_mask = sequences.ne(pad_token_id)

    # 4. Combine the masks
    # True only if position is <= last_eos_idx AND token is not PAD
    action_mask = position_mask & not_pad_mask

    return action_mask

def create_assistant_response_mask(mask, inputs, processor):
    """
    Create a boolean mask for the assistant's responses based on the chat template format.
    """
    # mask = torch.zeros_like(inputs, dtype=torch.bool)
    # weighted_mask = torch.zeros_like(inputs, dtype=torch.bool)
    # Get special token IDs
    im_start = processor.tokenizer.encode("<|im_start|>", add_special_tokens=False)[0]
    im_end = processor.tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
    assistant = processor.tokenizer.encode("assistant", add_special_tokens=False)[0]
    # answer = processor.tokenizer.encode("answer", add_special_tokens=False)[0]
    # For each sequence in the batch
    for i in range(inputs.shape[0]):
        sequence = inputs[i]
        # Find all im_start positions
        im_start_positions = (sequence == im_start).nonzero().flatten()
        for pos in im_start_positions:
            # Check if the token after im_start is "assistant"
            if pos + 1 < len(sequence) and sequence[pos + 1] == assistant:
                # Find the next im_end
                
                next_end = sequence[pos:].eq(im_end).nonzero()
                if len(next_end) > 0:
                    end_pos = pos + next_end[0].item()
                    # print('debug', [processor.tokenizer.decode(sequence[pos:end_pos+1])])
                    # print('=====')
                    # Mark the entire response (including the im_start and im_end tokens)
                    mask[i, pos:end_pos + 1] = True
    
    return mask

# export MIN_PIXELS=50176
# export MAX_PIXELS=501760
DEFAULT_MIN_PIXELS = int(os.getenv("MIN_PIXELS", 256*28*28))
DEFAULT_MAX_PIXELS = int(os.getenv("MAX_PIXELS", 5120*28*28))
IMAGE_FACTOR = 28
print(f"emaker min max pixels", DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS)

def resize_cropped(image, min_pixels=None, max_pixels=None):
    image = to_rgb(image)
    width, height = image.size
    if min_pixels is None: min_pixels =  DEFAULT_MIN_PIXELS
    if max_pixels is None: max_pixels = DEFAULT_MAX_PIXELS
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=IMAGE_FACTOR,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    print('resize cropped min max', min_pixels, max_pixels)
    image = image.resize((resized_width, resized_height))
    return image 

def check_imagepad(processor, batch_texts, batch_images):
    visual_inputs = processor(
        text=batch_texts,
        images=batch_images,
        videos=None,
        padding=True,
        max_length=20000,
        add_special_tokens=False,
        truncation=False,
        return_tensors="pt",
    )
    input_ids = visual_inputs['input_ids']
    imgpad = processor.tokenizer.encode("<|image_pad|>")[0]
    ntokens = [(x==imgpad).to(float).sum().item() for x in input_ids]
    print(f"autoprocessor output says the images patches are {ntokens}")
    
class RemoteExperienceMaker(NaiveExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, packing_samples=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines
        self.packing_samples = packing_samples
        self.rule_reward_func = batch_rule_reward
        self.q2gt = dict() 
        self.q2r = defaultdict(list)
        for dp in self.gt_path:
            # dp = gt_path
            if dp is None: continue 
            print('!!!! adding gts for', dp)
            ext = dp.split('.')[-1]
            if ext in ["json", "jsonl", "csv"]:
                ext = ext.lower().strip(".")
                if ext == "jsonl":
                    ext = "json"
                data = datasets.load_dataset(ext, data_files=dp)
                self.qkey = 'question'
                self.gt_key = 'gt_answer'
                
            else:
                if dp.endswith('parquet'): data = load_dataset('parquet', data_files=dp)
                else: data = load_dataset(dp)
                # blending_datasets(dp, "1.0", self.strategy, )
                # data = datasets.load_dataset('parquet', data_dir=dp)
                self.qkey = 'question'
                self.gt_key = 'answer'
            self.qidkey = 'qid'
                
            full_list = []
            for k,v in data.items(): 
                full_list.extend(v.to_list())
            data = full_list
            
            # q2gt
            # q2gt = dict() 
            # do we need to regularize the question?
            for item in data: 
                self.q2gt[item[self.qidkey]] = item[self.gt_key]
                if 'responses' in item: 
                    self.q2r[item[self.qidkey]].extend(item['responses'])
        dataver = getattr(self.strategy.args, "data_version", "red")
        if 'use_response' in dataver:
            assert len(self.q2r)>0, "no q2responses for red mode."
        
        if self.custom_reward_func:
            self.custom_reward_func = ray.remote(self.custom_reward_func)
        self.parse_code = False 
        self.executor = None
        ###### old version
        # self.tools = [CropImageNormalized().function]
        ####### new version
        self.operations = dict(crop_image_normalized=CropImageNormalized(), select_frames=SelectFrames())
        notool = getattr(self.strategy.args, "system_prompt", "none")=="notool"
        
        self.tools = [] if notool else [self.operations[k].function for k in ['crop_image_normalized', 'select_frames']]
        print(f"!!!! [check] prompt notool={notool}")
        self.prompt_maker = NousFnCallPrompt()

    def separate_qa(self, queries):
        if self.modelfamily=='qwen':
            return list(zip(*[extract_qwen_query_and_response(qq) for qq in queries]))
        elif self.modelfamily=='llamasft':
            return list(zip(*[extract_llama_query_and_response(qq) for qq in queries]))
        elif self.modelfamily=='autocode':
            return list(zip(*[extract_autocode_query_and_response(qq) for qq in queries]))
        elif self.modelfamily=='dpsk':
            return list(zip(*[extract_dpsk_query_and_response(qq) for qq in queries]))
        elif self.modelfamily=='dsmath':
            return list(zip(*[extract_dsmath_query_and_response(qq) for qq in queries]))
        else:
            raise Exception('Not implemented')
        
    @torch.no_grad()
    def make_experience_list(self, all_prompts: Union[str, List[str]], is_eval=False, **generate_kwargs) -> List[Experience]:
        print("===> [verbose] remoteEMaker make_experience_list()")
        if self.strategy.args.perf:
            self.perf_stats = {
                "generate_time": 0,
                "actor_value_rm_time": 0,
                "wait_time": 0,
            }
        self.eval_step = generate_kwargs.get("eval_step", 0)
        args = self.strategy.args
        generate_kwargs['is_eval'] = is_eval
        data_version = getattr(args, "data_version", None)
        if ('use_response' in data_version) and not is_eval:
            samples_list = self.generate_samples(all_prompts, use_response=True, **generate_kwargs)
        else:
            samples_list = self.generate_samples(all_prompts, **generate_kwargs)
        
        experiences = []
        nsample = 1 if is_eval else args.n_samples_per_prompt
        print(f"===> [verbose] all synced. REMaker get_experience(): single experience is arranged as {args.micro_rollout_batch_size} qas, and nsample={nsample}")
        for batched_sample in tqdm(samples_list):
            # breakpoint()
            tmp = self.get_logprobs_and_logs(batched_sample, is_eval=is_eval, validity=None)
            experiences.append(tmp.to_device("cpu"))
        torch.distributed.barrier()
        print(f"===> [verbose] REMaker get_experience(): all samples done logp {len(samples_list)}")
        experiences, rewards = self.handle_advantages(experiences, nsample=nsample)
        
        # calculate return and advantages
        for experience, reward in zip(experiences, rewards):
            if experience.action_log_probs is None: continue 
            
            # experience = experience.to_device("cuda")
            # reward = reward.to(device="cuda") # tensor of shape (queries,)
            num_actions = experience.info["num_actions"] # list of shape (queries,)
            ###########
            # kl = [[x, x, x],
            #       [x, x, x]]
            # reward = [1.0, 0.0]
            # reward = [[x,x,x+1.0],
            #           [x,x,x+0.0]]
            reward = compute_reward(
                reward, # tensor of shape (queries,)
                self.kl_ctl.value,
                experience.kl, # list of tensor, each shape = (ntokens,)
                action_mask=experience.action_mask, # None
                num_actions=num_actions,
                reward_clip_range=args.reward_clip_range,
            ) # list of tensor, each shape (ntokens,)
            
            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                    generate_kwargs["lambd"],
                )
            elif self.advantage_estimator in ["reinforce", "rloo","rloo_sft","group","group_sft"]: # indeed not doing anything
                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                )
                experience.advantages = deepcopy(experience.returns)
                experience.info["return"] = [x.mean() for x in experience.advantages]
                
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            experience.kl = None
            del experience.info["num_actions"]
            # experience.to_device("cpu")
    
        
        if self.critic is not None:
            for experience in experiences:
                # send experience to critic
                experience_cpu = deepcopy(experience)
                experience_cpu.to_device("cpu")
                self._ref = self.critic.append.remote(experience_cpu)
        print("!!!! [rbuffer] rearranged as (bsz, nsample) to compute rewards")
        return experiences

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[Dict], **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        if self.vllm_engines is None:
            return super().generate_samples(all_prompts, **generate_kwargs)

        print("===> [verbose] remoteEMaker generate_samples() using generate_vllm()")
        samples = self._generate_vllm(all_prompts, **generate_kwargs)
        print(f"===> [verbose] remoteEMaker generate_samples() done with {len(samples)} samples each with args.micro_rollout_batch_size qas")
        # vLLM offload when colocate_all_models
        if self.strategy.args.vllm_enable_sleep:
            if torch.distributed.get_rank() == 0:
                refs = []
                for engine in self.vllm_engines:
                    refs.append(engine.sleep.remote())
                ray.get(refs)
        return samples

    def convenient_get_batch_rewards_from_queries(self, queries, potential_qids, no_question=False):
        if no_question: solutions = queries
        else:
            questions, solutions = self.separate_qa(queries)
        gts = [self.q2gt.get(q, None) for q in potential_qids]
        
        format_type = getattr(self.strategy.args, "format", None)
        sysprompt = getattr(self.strategy.args, "system_prompt", None)
        requires_box = False if self.parse_code or sysprompt=='dpsk' else True # sysprompt !='autocode'
        rets = self.rule_reward_func(solutions, gts, self.tokenizer.eos_token, format_type, self.executor, requires_box)
        return rets 
        
    @torch.no_grad()
    def get_logprobs_and_logs(self, batched_sample: Samples, is_eval=False, validity=None, ) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        args = self.strategy.args
        dataver = getattr(args, "data_version", "red")
        use_response = 'use_response' in dataver
        if self.actor: self.actor.eval()
        device = torch.cuda.current_device()

        # extract values from samples
        sequences = batched_sample.sequences
        attention_mask = batched_sample.attention_mask
        action_mask = batched_sample.action_mask
        num_actions = batched_sample.num_actions
        na_each = batched_sample.na_each 
        packed_seq_lens = batched_sample.packed_seq_lens
        visual_inputs = batched_sample.visual_inputs
        prompts = batched_sample.prompts
        round0_correctness = batched_sample.round0_correctness
        round1_correctness = batched_sample.round1_correctness
        round0_nwait = batched_sample.round0_nwait 
        round1_nwait = batched_sample.round1_nwait 
        questions = batched_sample.questions 
        solutions = batched_sample.solutions
        potential_qids = batched_sample.qids
        eff_labels = batched_sample.efficiency_label
        
        num_seq = len(sequences) # default to cpu device
        
        start = time.time()
        device = 'cuda'
        sequences_cpu, attention_mask_cpu = (
            sequences.to(device),
            attention_mask.to(device),
        )
        visual_inputs_cpu = None
        if visual_inputs is not None:
            visual_inputs_cpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in visual_inputs.items()}        
        # init log probs
        if self.initial_model is not None:
            base_action_log_probs_ref = self.initial_model.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens,visual_inputs=visual_inputs_cpu
            )

            if args.colocate_actor_ref or args.colocate_all_models:
                ray.get([base_action_log_probs_ref])
                ray.get([self.initial_model.empty_cache.remote()])
        else:
            base_action_log_probs_ref = ray.put(None)

        # values
        if self.critic is not None:
            value_ref = self.critic.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens, visual_inputs=visual_inputs_cpu
            )
            # avoid CUDA OOM when colocate models
            if args.colocate_critic_reward or args.colocate_all_models:
                ray.get([value_ref])
                ray.get([self.critic.empty_cache.remote()])
        else:
            value_ref = ray.put(None)

        # rewards
        r_refs = []
        
        if args.colocate_all_models and self.reward_model:
            ray.get(r_refs)
            ray.get([self.reward_model[0].empty_cache.remote()])

        actor_value_rm_time = time.time() - start

        # wait initial/critic/reward model done
        start = time.time()
        ref_values = ray.get([base_action_log_probs_ref, value_ref] + r_refs)
        wait_time = time.time() - start

        base_action_log_probs, value, rewards = ref_values[0], ref_values[1], ref_values[2:]
        acc_rewards = []
        norepeat_rewards = []
        usefmt_rewards = []
        raw_rewards = []
        initial_validity = validity
        validity = None
        error_infos = []
        use_codes = []
        # ns_in_correct = []
        exceptions = []
        eostoken = self.tokenizer.eos_token
        data_version = getattr(args, "data_version", None)
        force_wait = "force_append_wait" in data_version
        if not (self.reward_model or self.remote_rm_url): 
            
            rewards = []
            validity = []
            
            format_type = getattr(self.strategy.args, "format", None)
            sysprompt = getattr(self.strategy.args, "system_prompt", None)
            requires_box = False if self.parse_code or sysprompt in ['dpsk','notrigger'] else True # sysprompt !='autocode'
            print(f'requires_box={requires_box}')
            # num = len(questions)
            
            if use_response and not is_eval: 
                error_infos = [None for _ in range(num)]
                use_codes = [0.0 for _ in range(num)]
                validity = [1.0 for _ in range(num)]
                norepeat_rewards = [1.0 for _ in range(num)]
                usefmt_rewards = [1.0 for _ in range(num)]
                # ns_in_correct = [1.0 for _ in range(num)]
                round0_nwait = [0.0 for _ in range(num)]
                round1_nwait = [0.0 for _ in range(num)]
                raw_rewards = [1.0 for _ in range(num)] 
                exceptions = [0.0 for _ in range(num)]
            else: 
                for iidx,(ret0,ret1) in enumerate(zip(round0_correctness, round1_correctness)):
                    # print('!!!! solution', sol)
                    if ret1 is None: ret = ret0
                    else: ret = ret1
                    if self.parse_code:
                        valid, norepeat, usefmt, error_info, usecode, final_correct = ret
                    else: 
                        valid, norepeat, usefmt, error_info, final_correct = ret
                        usecode = False
                    
                    if initial_validity: 
                        valid = initial_validity[iidx] and valid
                    error_infos.append(error_info)
                    use_codes.append(usecode)
                    validity.append(1.0 if valid else 0.0 )
                    norepeat_rewards.append(norepeat) 
                    usefmt_rewards.append(usefmt)
                    # ns_in_correct.append(0.0)
                    raw_rewards.append(1.0 if final_correct>0 else 0.0)
                    exceptions.append(1.0 if final_correct<0 else 0.0)
                    
            # for valid, final_correct in zip(validity, raw_rewards):
            # for valid, final_correct, r0nw, r1nw, r1c, eff_flag in zip(validity, raw_rewards, round0_nwait, round1_nwait, round1_correctness, eff_labels):
            #     if valid>0.5:
            #         shaped_reward = 1.0 if final_correct>0.5 else 0.0 
            #     else:
            #         shaped_reward = -0.1
            #     ########### it seems not proper to use additive rewards
            #     if not is_eval:
            #         if eff_flag>0.5: # there is a solution without using tool 
            #             if final_correct>0.5 and r0nw>0:
            #                 shaped_reward = max(shaped_reward - 0.1*r0nw, 0.6)
                      
            #     print(f"===> [verbose] shaped_reward={shaped_reward}, final_correct={final_correct}, r0nw={r0nw}, r1nw={r1nw}, r1c={r1c}")
                
            #     rewards.append(shaped_reward)
            if is_eval:
                rewards = raw_rewards 
            else:
                rewards = batched_sample.shaped_rewards 
            
            # uniformity = batched_sample.uniformity
            print(f"===> [verbose] shaped_reward={rewards}")
            rewards = torch.FloatTensor(rewards) # a list of tensor, tensor shape = queries shape
        # print('!!!! debug rewards', rewards.shape)
        info = {
            "reward": rewards, # tensor of shape (queries)
            "response_length": batched_sample.response_length,
            "total_length": batched_sample.total_length,
            "num_actions": na_each,
            "validity": validity, 
            "norepeat": [0.0 if x is None else float(x) for x in norepeat_rewards],
            "usefmt": [0.0 if x is None else float(x) for x in usefmt_rewards],
            "match": [0.0 if x is None else float(x)  for x in raw_rewards],
            "use_codes": [0.0 if x is None else float(x) for x in use_codes],
            # "num_switch": [float(x) for x in ns_in_correct],
            "round0_nwait": [float(x) for x in round0_nwait],
            "round1_nwait": [float(x) for x in round1_nwait],
            "round0_correctness": [float(x[-1]) for x in round0_correctness],
            "round1_correctness": [-1.0 if x is None else float(x[-1]) for x in round1_correctness],
            "qids": potential_qids,
            # "round0_saturation": batched_sample.round0_saturation,
            "round0_ALLTrue": batched_sample.round0_ALLTrue,
            "round0_ALLFalse": batched_sample.round0_ALLFalse,
            "round0_Easy": batched_sample.round0_Easy,
            "round0_Hard": batched_sample.round0_Hard,
            "round0_Medium": batched_sample.round0_Medium,
            "uniformity": batched_sample.uniformity,
            "curiosity": batched_sample.curiosity_bonus,
            "penalty": batched_sample.penalty_bonus
        }
            
        if base_action_log_probs is not None:   
            base_action_log_probs = base_action_log_probs.to(device)
        if value is not None:
            value = value.to(device)
        
        # rewards = [r.to(device) for r in rewards]
        # r = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0]

        # avoid CUDA OOM when colocate models
        if args.colocate_critic_reward and self.reward_model:
            ray.get([self.reward_model[0].empty_cache.remote()])

        if args.colocate_actor_ref or args.colocate_all_models:
            torch.cuda.empty_cache()

        # log probs
        
        if is_eval or use_response:
            action_log_probs = None
        else:
            print(f"===> [verbose] remoteEMaker make_experience() processing {num_seq} qas for action_logprob")
            # when multiturn, num_actions will be 0
            # vidpad = self.data_processor.processor.tokenizer.encode("<|video_pad|>")[0]
            # imgpad = self.data_processor.processor.tokenizer.encode("<|image_pad|>")[0]
            with torch.no_grad():
                action_log_probs = self.actor(sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens,visual_inputs=visual_inputs_cpu)
            # print('finish forward', torch.distributed.get_rank())
            action_log_probs = action_log_probs.to('cpu')
        torch.distributed.barrier()
        if is_eval or use_response:
            kl = None
        elif self.initial_model is not None:
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=action_mask,
                use_kl_estimator_k3=args.use_kl_estimator_k3,
            )
        else:
            kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device='cpu')

        if is_eval or use_response:
            kl_mean_log = None 
            kl_mean = None
        else:
            if not self.packing_samples:
                kl_mean = masked_mean(kl.to(action_mask.device), action_mask, dim=-1)
                # print(kl.device, action_mask.device)
            else:
                # convert tensor into list of tensors so that it's easier to manipulate
                # within dataset.
                sequences = unpacking_samples(sequences, packed_seq_lens)
                attention_mask = None
                action_log_probs = unpacking_samples(action_log_probs, num_actions)
                if value is not None:
                    value = unpacking_samples(value, num_actions)

                kl = unpacking_samples(kl, num_actions)
                kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device)
                
            kl_mean_log = kl_mean.detach().cpu().numpy().tolist()
        
        info['kl'] =  kl_mean
        if self.strategy.args.perf:
            self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
            self.perf_stats["wait_time"] += wait_time

        experience = Experience(
            sequences,
            action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
            visual_inputs=visual_inputs,
            validity=validity
        )
        
        if self.actor: self.actor.train()  # reset model state
        # print('!!!! [debug] logging on', self.strategy.get_rank())
        if self.strategy.is_rank_0() or is_eval:
            log_file = self.strategy.args.ckpt_path + '/logs'
            import os 
            os.makedirs(log_file, exist_ok=True)
            log_file += '/sample.'
            
            if log_file:
                if is_eval: log_file += f'eval_iter{self.eval_step}_{self.strategy.get_rank()}.jsonl'
                else: log_file += 'jsonl'
                print(f'===> [verbose] actionlogp reward done for batch @rank{self.strategy.get_rank()}, written to log', log_file)
                with open(log_file,'a') as f:
                    
                    dump_info = dict()
                    for k,v in info.items():
                        if isinstance(v, torch.Tensor):
                            v = v.detach().cpu().numpy().tolist()
                        dump_info[k] = v
                    # print('debug', info['reward'])
                    dump_info['questions'] = questions
                    dump_info['solutions'] = solutions
                    gts = [self.q2gt.get(q, None) for q in dump_info['qids']]
                    dump_info['gts'] = gts 
                    
                    num = len(dump_info['qids']) # 96 
                    # print('!!!! debug ', dump_info)
                    for i in range(num):
                        entry = dict()
                        for k in ['solutions', 'gts', 'round0_correctness', 'round1_correctness','validity', 'reward', 'round1_nwait', 'round0_nwait',  'qids', 'questions', 'num_actions','curiosity','penalty']: # error_info, usefmt, use_codes
                            # if k=='sol': continue 
                            if k not in dump_info: continue 
                            
                            if len(dump_info[k])!=num:
                                raise Exception(f"dump-info key {k}: {len(dump_info[k])} should be {num}")
                            
                            v = dump_info[k][i]
                            
                            entry[k] = v
                        f.write(json.dumps(entry)+'\n')
        del sequences, sequences_cpu, action_log_probs, attention_mask, attention_mask_cpu, visual_inputs, visual_inputs_cpu       
        return experience

    def send_requests_to_vllms(self, rank, all_messages, llms, sampling_params):
        refs = []
        batch_size = (len(all_messages) + len(llms) - 1) // len(llms)
        print(f'!!!! [vllm] {len(all_messages)} messages, bsz_each={batch_size}=nqa, numllm={len(llms)}')
        for i, llm in enumerate(llms):
            messages = all_messages[i * batch_size : (i + 1) * batch_size]
            prompts = self.data_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts = self.data_processor.handle_placeholders(prompts)
            
            images = [self.data_processor.get_images_from_messages(m) for m in messages]
            # print('!!!! debug img type', type(images[0][0]))
            vllm_inputs = [{
                        "prompt": p,
                        "multi_modal_data":{"image": imgs}  
                        
            } for p, imgs in zip(prompts,images)]
            

            refs.append(
                llm.add_requests_vlm.remote(rank, sampling_params=sampling_params, vllm_vision_input=vllm_inputs)
            )
        return refs 
    
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
        action_mask[:, 0] = 1

        return sequences, attention_mask, action_mask
    
    def _generate_vllm(self, all_prompts: List[str], use_response=False, skip_generation=False, **kwargs) -> List[Samples]:
        from vllm import SamplingParams
        image_mode = 'RGB'
        image_size = (56, 56) # Example size: 400 pixels wide, 300 pixels high
        background_color = (255, 255, 255) # White color
        blank_image = Image.new(image_mode, image_size, background_color)
        raw_maxsize = 2000
        zoom_maxsize = 1000
        select_maxsize = 400
        eval_minpixel = 256 
        eval_maxpixel = 8000 # for tools, should be 8000
        
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            vllm_rank = rank % len(self.vllm_engines)
            llms = [self.vllm_engines[vllm_rank]]
            
        else:
            vllm_rank = rank
            llms = self.vllm_engines[rank::world_size]

        maxtoken = kwargs.get("max_new_tokens", 1024) # generate max len
        
        
        args = self.strategy.args
        maxtoken=getattr(args, "max_out_tokens", 2048)
        print(f"!!!! [warning] forcifully using maxtoken={maxtoken} for vllm")
        data_version = getattr(args, "data_version", None)
        do_wait = data_version == "append_wait"
        force_wait = "force_append_wait" in data_version 
        force_eval_wait = "force_append_wait_eval" == data_version
        force_all_wait = data_version == "force_append_wait_all"
        # print(f'!!!! debug replace_wait={do_wait} or {force_wait}')
        do_vlm = getattr(args, 'train_vlm', False)
        multires = getattr(args, "multires", "False")=="True"
        stop_tokens = ['<|im_end|>','<|eot_id|>','<|endoftext|>'] # ,'</tool_call>'
        
        print(f'===> [verbose] remoteEMaker _generate_vllm() handling whole batch of {len(all_prompts)} queries')
        skip_generation = use_response
        if use_response:
            
            if not do_vlm:
                pass 
                # questions = [extract_qwen_query_and_response(p)[0] for p in all_prompts]
                # raw_qlist = [regularize_text(q) for q in questions] # bug: should use qids instead
                
                # sources = [self.tokenizer.apply_chat_template([dict(role='user', content=prompt)], tokenize=False, add_generation_prompt=True).split("<think>")[0] for prompt in all_prompts for _ in range(args.n_samples_per_prompt)]
                # targets = [r+self.tokenizer.eos_token for q in raw_qlist for r in self.q2r[q][:args.n_samples_per_prompt]]
                
                # assert len(targets)==args.n_samples_per_prompt*len(all_prompts), f"{len(targets)}"
                # all_s = self.tokenize_fn(sources, maxtoken, padding=False)["input_ids"] # list of list of ids
                # inp_num_tokens = [len(x) for x in all_s]
                # out_num_tokens = [maxtoken-x for x in inp_num_tokens]
                
                # all_t = [self.tokenize_fn(t, nt, padding=False)["input_ids"] for t,nt in zip(targets, out_num_tokens)] # list of list of ids
                # for ttok,nt in zip(all_t, out_num_tokens):
                #     print(f'!!!! nt={nt}, realtok={len(ttok)}, valid={ttok[-1]==self.tokenizer.eos_token_id}')
                
                # print('!!!! peek targets', [targets[0][:100],'...',targets[0][-100:]])
                
                # all_outputs = [(a,b) for a,b in zip(all_s, all_t)]
                # print('!!!! num qas', len(all_outputs))
            else:
                all_outputs_offline = []
                all_inputs_offline = []
                for p in all_prompts:
                    chat = json.loads(p) 
                    
                    ###### will be a chat list like this
                    # chat = [dict(role='user', 
                    #          content=[dict(type='image', image=img),
                    #                   dict(type='text', text=q)
                    # ])]
                    # if sysp: chat.insert(0, dict(role='system', content=templates[system_prompt]))
                    # for entry in chat:
                    #     if entry['role']=='user': break
                    qid = chat[-1]['qid']
                    # rq = regularize_text(entry['content'][-1]['text']) 
                    
                    responses = self.q2r[qid][:args.n_samples_per_prompt]
                    # import pdb; pdb.set_trace()
                    cleaned_chat = []
                    for entry in chat:
                        if 'content' in entry:
                            cleaned_chat.append(entry)
                    inputs = self.data_processor(json.dumps(cleaned_chat), self.prompt_max_len, device="cpu")['input_ids'] # output will be a list 
                    
                    # rlist = [rsp+self.tokenizer.eos_token for rsp in responses]
                    for rsp in responses:
                        out = rsp+self.tokenizer.eos_token
                        
                        out_tokens = self.data_processor.processor(
                            text=out,
                            padding=False,
                            max_length=args.generate_max_len,
                            add_special_tokens=False,
                            truncation=True,
                            return_tensors='np',
                        )['input_ids'] # output will be a list 
                        all_outputs_offline.extend(out_tokens)
                        all_inputs_offline.extend(inputs.cpu().numpy().tolist()*len(out_tokens))
                        # print('!!!! [debug]', all_inputs_offline[-1])

        is_eval = kwargs['is_eval']
        max_imgnum = 16 # 24 if is_eval else 16
        maxpixel = os.environ.get("MAX_PIXELS", 5120*28*28)
        is_final_eval = maxpixel is not None
        if is_final_eval:
            maxpixel = int(maxpixel)
        
        img_maxtoken = 512 # int(maxpixel)//10//28//28 
        
        
        if is_eval: #  and args.n_samples_per_prompt==1:
            temperature = 0.0 # zero is greedy
            top_p = 1 
            top_k = -1 
        else:
            temperature=getattr(args, "temperature", 0.85)
            top_p=kwargs.get("top_p", 1.0)
            top_k=kwargs.get("top_k", 40)
            if is_eval:
                temperature = getattr(args, "val_temperature", 0.6)
                top_p = 0.95
        
        flag = False 
        all_messages = []
        all_raw_messages = []
        # all_conversations = []
        # all_images = []
        all_conversations = dict()
        all_images = dict()
        all_raw_images = dict()
        nsample = 1 if is_eval else args.n_samples_per_prompt
        
        potential_qids = []
        qids_expanded = []
        maxtokens = []
        for m in all_prompts: 
            info = json.loads(m)
            if 'qid' in info[-1]: 
                newm = json.dumps(info[:-1]) # we need to drop the qid entry 
                qid = info[-1]['qid']
                potential_qids.append(qid) 
                qids_expanded.extend([qid]*nsample)
                mtoken_list = [img_maxtoken for _ in range(nsample)]  if (is_eval or not multires) else [np.random.choice([128,256,512],size=1) for _ in range(nsample)]
                maxtokens.extend(mtoken_list)
            else: newm = m
            all_messages.extend([newm]*nsample)
            all_raw_messages.extend([m]*nsample)
            
        
        if is_eval or not skip_generation:
            
            sampling_params = SamplingParams(
                temperature=temperature, 
                top_p=top_p,
                top_k=top_k,
                max_tokens=maxtoken,
                min_tokens=kwargs.get("min_new_tokens", 1),
                skip_special_tokens=kwargs.get("skip_special_tokens", False),
                include_stop_str_in_output=False, # from True to False. 
                stop=stop_tokens,
            )
            print(f'!!!! [vllm] is_eval={is_eval}, sampling args', sampling_params)
            
            refs = []
            rearrange_indices = []
            
            batch_size = (len(all_messages) + len(llms) - 1) // len(llms)
            print(f'===> [verbose] to handle {len(all_messages)} qas, bsz={batch_size} qas for {len(llms)} vllm engine.')
            all_vllm_inputs = dict()
            all_uids = []
            all_video_flags = []
            for i, llm in enumerate(llms):
                messages = all_messages[i * batch_size : (i + 1) * batch_size]
                batch_qids = qids_expanded[i * batch_size : (i + 1) * batch_size]
                batch_mtokens = maxtokens[i * batch_size : (i + 1) * batch_size]
                batch_uids = [f"{qqid}-{xx}" for xx,qqid in zip(range(i * batch_size, i * batch_size+len(batch_qids)), batch_qids)]
                all_uids.extend(batch_uids)
                oldformat_messages = messages
                prompts, conversations = get_prompt_from_messages(oldformat_messages, self.prompt_maker, self.tools, self.data_processor.processor)
                
                conversations, images, has_video = self.data_processor.obtain_conv_images_from_conversations(conversations, 
                                                                                                  batch_min_pixels=[(eval_minpixel if is_eval else 4)*28*28  for x in batch_mtokens], #[x*28*28 for x in batch_mtokens], 
                                                                                                  batch_max_pixels=[(eval_maxpixel if is_eval else raw_maxsize)*28*28 for x in batch_mtokens],) #[x*10*28*28 for x in batch_mtokens])
                # print(f"image sizes = {images}")
                all_video_flags.extend(has_video)
                
                for uuid, conv, imglist, video_flag in zip(batch_uids, conversations, images, has_video):
                    all_conversations[uuid] = conv 
                    
                    if video_flag:
                        all_images[uuid] = [imglist] # video is represented as a list of images
                        rawimagelist = [imglist]
                    else: 
                        all_images[uuid] = imglist
                        # vinfo = copy(extract_vision_info(conv)[0]) # list of info of images
                        # # rawimagelist  = []
                        # if i==0: print('===> [verbose] previous max pixel', vinfo.get('max_pixels',None))
                        # # for vinfo in vinfolist:
                        # vinfo['min_pixels'] = 512*28*28 
                        # vinfo['max_pixels'] = 5120*28*28
                        # rawimagelist = [fetch_image(vinfo)]
                        rawimagelist = imglist
                        # breakpoint()
                    all_raw_images[uuid] = rawimagelist
                
                vllm_inputs = dict()
                for pp, imgs, uuid, video_flag in zip(prompts, images, batch_uids, has_video):
                    tmp = {
                            "prompt": pp,
                            "multi_modal_data": {"video": imgs} if video_flag else {"image": imgs} , # a trick to handle text-only queries  
                            # "mm_processor_kwargs": {
                            #     "min_pixels": int(os.getenv("MIN_PIXELS", 64 * 28 * 28)),
                            #     "max_pixels": int(os.getenv("MAX_PIXELS", 640 * 28 * 28)),
                            # },
                    }
                    if imgs is None:
                        raise Exception("images cannot be None")
                    # vllm_inputs.append(tmp)
                    vllm_inputs[uuid] = tmp 
                all_vllm_inputs.update(vllm_inputs)

                refs.append(
                    llm.add_requests_vlm.remote(rank, sampling_params=sampling_params, vllm_vision_input=[vllm_inputs[uuid] for uuid in batch_uids])
                )
            print(f'===> [verbose] {len(all_messages)} QA request submitted to {len(llms)} vllm engine.')
            if flag and rearrange_indices:
                print('!!!! debug rearr', rearrange_indices)
            ray.get(refs)

            # Make sure all requests are sent.
            torch.distributed.barrier()

            # Retrieve and combine results from all outputs
            all_output_refs = []
            for i, llm in enumerate(llms):
                all_output_refs.append(llm.get_responses.remote(rank))
            all_outputs = sum(ray.get(all_output_refs), [])
            if flag and rearrange_indices:
                print('!!!! output fetched', rearrange_indices)
                all_outputs = [all_outputs[i] for i in rearrange_indices]
        
        print(f"===> [verbose] decode and evaluate the initial round of responses")
        
        idx2uid = all_uids
        all_inputs_ = [list(old_out.prompt_token_ids) for old_out in all_outputs]
        all_outputs_ = [list(old_out.outputs[0].token_ids) for old_out in all_outputs] 
        solutions_round0 = self.tokenizer.batch_decode(all_outputs_, skip_special_tokens=False)
        questions = self.tokenizer.batch_decode(all_inputs_, skip_special_tokens=False)
        questions_cleaned = [x.replace("<|image_pad|>","").replace("<|video_pad|>","") for x in questions]
        all_qa_texts = [x+y for x,y in zip(questions_cleaned, solutions_round0)]
        
        num_toolcalls = [0] * len(all_qa_texts)
        num_toolfails = [0] * len(all_qa_texts)
        ####### add saturation calculation here
        tool_end = "</tool_call>"
        maxtool = getattr(args, "maxturn", 2)
        print(f"===> [verbose] doing multiturn of {maxtool} turns")
        niter = 0
        maxtool = maxtool - 2 # 0
        # multiturn_messages = [json.loads(line) for line in all_messages]
        all_flags =  [False for _ in range(len(qids_expanded))]
        final_error_flags = [False for _ in range(len(qids_expanded))]
        do_dump = (getattr(args, "training_mode", "train") in {'eval_only','train'}) and is_eval
        temp_error_flags = [False for _ in range(len(qids_expanded))]
        while True: 
            req_indexlist = []
            req_vllminputs = []
            req_qids = []
            
            
            print(f"========= niter {niter}")
            
            for out_idx, (qqid, out, qatext,fflag) in enumerate(zip(qids_expanded, all_outputs, all_qa_texts,all_flags)):
                if fflag: continue 
                uuid = idx2uid[out_idx]
                
                tfolder = self.strategy.args.ckpt_path + f'/logs/dumps_iter{self.eval_step}/{qqid}'
                if do_dump: os.makedirs(tfolder, exist_ok=True)
                targetpath = f"{tfolder}/iter{niter}.jpg"
                
                rsp = solutions_round0[out_idx].replace("<|im_end|>","")
                msg_this = [dict(role='assistant', content=[
                    dict(type='text', text=rsp),])]
                # rsp.endswith(tool_end) also fine
                last_string = rsp[-len(tool_end)-10:] if len(rsp)>len(tool_end)+10 else rsp
                require_tool = last_string.endswith(tool_end) # check whether tool trigger in out_ids 
                cur_tokens_in = len(all_outputs[out_idx].prompt_token_ids)
                cur_tokens_out = len(all_outputs[out_idx].outputs[0].token_ids)
                cur_tokens = cur_tokens_in + cur_tokens_out
                force_terminate = num_toolcalls[out_idx]>3 or len(all_images[uuid])>max_imgnum or cur_tokens>12*1024-200
                finish_flag = not require_tool or force_terminate # either it ends with im_end, either it exceeds max length 
                all_flags[out_idx] = finish_flag
                final_error_flags[out_idx] = finish_flag and temp_error_flags[out_idx]
                num_toolcalls[out_idx] += 1 if require_tool else 0
                num_toolfails[out_idx] += 1 if force_terminate else 0
                
                if out_idx == 0:
                    print(f"++++ [debug] qqid={qqid}, iter={niter}, A={msg_this}, require_tool={require_tool}")
                    # if niter==2:
                if finish_flag: 
                    all_conversations[uuid] = all_conversations[uuid] + msg_this
                    if do_dump:
                        json.dump([all_conversations[uuid],
                               all_vllm_inputs[uuid]['prompt'],
                               str(all_images[uuid])
                               ], open(f"{tfolder}/conv.json",'w'))
                    continue 
                # msg_this = []
                imagelist = all_images[uuid]
                rawimagelist = all_raw_images[uuid]
                video_flag = all_video_flags[out_idx]
                error_flag = False 
                temp_error_flags[out_idx] = False
                try:
                    tool_params = parse_last_tool(qatext) 
                    tool_name = tool_params['name']
                    tool_args = tool_params['arguments']
                    # if tool_name in {'crop_image_normalized','crop_image'}:
                    raw_result = execute_tool(imagelist, rawimagelist, tool_args, tool_name, is_video=video_flag, function=self.operations[tool_name].call)
                    if tool_name=='select_frames': 
                        if not video_flag:
                            selected_frames = []
                            info = "Execution error:\nYou attempted to select frames from an **image**, but this operation is only desgined for analyzing videos. Think again.\n"
                        selected_frames, info = raw_result 
                        if not isinstance(info, str): # info is the replacement 
                            # assert "" in msg_this[-1]['content'][0]['text']
                            oldtext = msg_this[-1]['content'][0]['text']
                            newtext = oldtext.replace(str([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]), str(info))
                            msg_this[-1]['content'][0]['text'] = newtext 
                        
                        if is_eval:
                            added = selected_frames
                        else:
                            added = [resize_cropped(ff, min_pixels=(256 if is_eval else 4)*28*28, max_pixels=(5120 if is_eval else select_maxsize)*28*28) for ff in selected_frames]
                        
                        if len(selected_frames)==0:
                            msg_this.append(
                                dict(role='user', content=[
                                    dict(type='text', text=f"\n{info}"),
                                ] )
                            )
                        else:    
                            msg_this.append(
                            dict(role='user', content=[
                                dict(type='text', text="\nHere are the selected frames (Frame Size: {}x{}, Numbered {} to {}):".format(added[0].size[0], added[0].size[1], len(imagelist), len(selected_frames)+len(imagelist)-1)),
                            ] + [dict(type='image', image=targetpath) for _ in range(len(selected_frames))])
                            )
                        
                    else:
                        mtoken = maxtokens[out_idx]
                        proc_img = resize_cropped(raw_result, min_pixels=(256 if is_eval else 4)*28*28, max_pixels=(5120 if is_eval else zoom_maxsize)*28*28)
                        if do_dump:
                            proc_img.save(targetpath)
                            print('dumped', targetpath)
                        
                        added = [proc_img]
                        msg_this.append(
                            dict(role='user', content=[
                                dict(type='text', text="\nHere is the cropped image (Image Size: {}x{}):".format(proc_img.size[0], proc_img.size[1])),
                                dict(type='image', image=targetpath)
                            ])
                        )
                    
                except Exception as e:
                    print('!!!!!!! warning')
                    print(e)
                    # print('tool args', tool_args, tool_name)
                    # breakpoint()
                    error_info = str(e)
                    msg_this.append(
                        dict(role='user', content=[
                            dict(type='text', text=f"\nExecution error:\n{error_info}\n")
                        ])
                    )
                    # breakpoint()
                    num_toolfails[out_idx] += 1
                    error_flag = True 
                    temp_error_flags[out_idx] = True
                    added = []
                    
                new_images = all_images[uuid] + added
                if len(new_images)>max_imgnum: # the returned is exceeding the max image num
                    all_flags[out_idx] = True
                    if error_flag: final_error_flags[out_idx] = True
                    continue
                tconv, _,_ = self.data_processor.obtain_conv_images_from_conversations([msg_this], no_image=True)
                all_conversations[uuid] = all_conversations[uuid] + tconv[0]
                all_images[uuid] = all_images[uuid] + added
                prompts = self.data_processor.processor.apply_chat_template([all_conversations[uuid]], tokenize=False, add_generation_prompt=True)[0]
                
                all_vllm_inputs[uuid]['prompt'] = prompts
                
                if video_flag:
                    # for videos, they have been added in the first round
                    all_vllm_inputs[uuid]["multi_modal_data"]['image'] = all_images[uuid][1:]
                else:
                    all_vllm_inputs[uuid]["multi_modal_data"]['image'] = all_images[uuid]
                
                ############# 
                req_vllminputs.append(all_vllm_inputs[uuid])
                req_qids.append(qqid)
                req_indexlist.append(out_idx)
                
            # ====> termination 
            if len(req_vllminputs)==0:
                print(f"===> [verbose] all queries already finish at iter {niter}")
                break 
            batch_size = (len(req_vllminputs) + len(llms) - 1) // len(llms)
            print(f'===> [vllm] tool-call@iter{niter} requests {len(req_vllminputs)}/{len(all_flags)} messages, bsz_each={batch_size}=nqa, numllm={len(llms)}')
            
            reqs = []
            
            for i, llm in enumerate(llms):
                vllm_inputs = req_vllminputs[i * batch_size : (i + 1) * batch_size] # dict(prompt, multimodal_data)
                tmp_params = copy(sampling_params)
                if not is_eval:
                    tmp_params.temperature = 0.9 
                reqs.append(
                    llm.add_requests_vlm.remote(rank, sampling_params=tmp_params, vllm_vision_input=vllm_inputs)
                )
            print(f"===> [verbose] submit rethinking requests {len(req_vllminputs)}")
            ray.get(reqs)
            
            niter += 1
            
            new_output_refs = []
            for i, llm in enumerate(llms):
                new_output_refs.append(llm.get_responses.remote(rank))
            new_outputs = sum(ray.get(new_output_refs), [])
            
            print(f"===> [verbose] decode the new tool-executed responses ")
            new_tokens_in = [list(out.prompt_token_ids) for out in new_outputs]
            new_tokens_out = [list(out.outputs[0].token_ids) for out in new_outputs]
            new_texts_in = self.tokenizer.batch_decode(new_tokens_in, skip_special_tokens=False)
            new_texts_out = self.tokenizer.batch_decode(new_tokens_out, skip_special_tokens=False)
            # print(f"peek iter {niter}", new_texts_in[0])
            new_texts_in = [x.replace("<|image_pad|>","").replace("<|video_pad|>","") for x in new_texts_in]
            new_texts = [x+y for x,y in zip(new_texts_in, new_texts_out)]
            
            new_idx = 0
            # for old_idx, flag in enumerate(all_flags):
            for new_idx, old_idx in enumerate(req_indexlist):
                # if flag: continue 
                all_outputs[old_idx] = new_outputs[new_idx]
                # all_vllm_inputs[old_idx] = req_vllminputs[new_idx]
                all_qa_texts[old_idx] = new_texts[new_idx]
                solutions_round0[old_idx] = new_texts_out[new_idx]
            
        # peek the responses 
        torch.distributed.barrier()
        
        rets_round1 = self.convenient_get_batch_rewards_from_queries(all_qa_texts, qids_expanded)
        difficulty_labels = []
        total = 0
        # nsample = args.n_samples_per_prompt
        efficiency_labels = []
        uniformity = []
        shaped_rewards = []
        curiosity_bonus = []
        penalty_bonus = []
        for idx in range(0, len(rets_round1), nsample):
            correctness = [x[-1] for x in rets_round1[idx:idx+nsample]]
            group_score = np.mean(correctness)
            ntoolcalls = num_toolcalls[idx:idx+nsample]
            videoflags = all_video_flags[idx:idx+nsample]
            
            has_correct_without_tool = False 
            for mres, ncall in zip(correctness, ntoolcalls):
                if mres>0.5 and ncall<0.1: 
                    has_correct_without_tool = True 
                    break 
            rapr = np.mean([ncall>0. for ncall in ntoolcalls])
            efficiency_labels.extend([float(has_correct_without_tool)]*nsample)
            this_rewards = []
            discount = 1.0
            this_cur = []
            this_pen = []
            for iidx, (mres, ncall, isvideo) in enumerate(zip(correctness, ntoolcalls,videoflags)):
                this_r = float(mres)
                final_is_error_vo = final_error_flags[iidx]
                if this_r>0.5 and final_is_error_vo: # there is a failure for visual operations but the model does not fix it
                    this_r = 0.0 
                # incentivize select_frames 
                # curiosity: if rapr==0.5, fair
                curiosity = 0.0 
                penalty = 0.0
                # if isvideo and ncall>0.1: # for video
                #     curiosity = max(0.3 - rapr, 0.0) / 1.5
                #     curiosity = curiosity / rapr * 0.25 # the maximum curiosity is 0.25
                #     penalty = - 0.05*(ncall-1)
                #     # bonus = max(0.0, discount*(curiosity + penalty)) # curiosity; failure penalty
                #     bonus = discount*(curiosity + penalty)
                #     this_r += bonus 
                # elif ncall>0.1: # for image only when it's correct?
                #     curiosity = max(0.3 - rapr, 0.0) / 2.0
                #     curiosity = curiosity / rapr * 0.25 # the maximum curiosity is 0.25
                #     penalty = - 0.05*(ncall-1)
                #     # bonus = max(0.0, discount*(curiosity + penalty))
                #     bonus = discount*(curiosity + penalty)
                #     this_r += bonus
                #################
                # if has_correct_without_tool: # then we can encourage correct tool
                #     if mres>0.5 and ncall>0.1: # discourage unnecessary tool
                #         this_r = max(0.7, 1.0-ncall*0.06)
                # else: # there is no correct without tool
                #     if mres>0.5: # encourage efficient tool
                #         if ncall<1.1: this_r = 1.06
                #         else: this_r = max(0.7, 1.0-(ncall-1)*0.06)
                #############
                this_rewards.append(this_r)
                this_cur.append(curiosity)
                this_pen.append(penalty)
            sum_rewards = sum(this_rewards)
            mean_rewards = np.mean(this_rewards)
            is_uniform = False
            
            if np.mean([abs(x-mean_rewards) for x in this_rewards])<0.01:
                is_uniform = True
                      
            shaped_rewards.extend(this_rewards)  
            curiosity_bonus.extend(this_cur)
            penalty_bonus.extend(this_pen)
            uniformity.extend([float(is_uniform)]*nsample)
            if group_score<1./8.:
                difficulty_labels.extend([0]*nsample)
            elif group_score<3./8.: 
                difficulty_labels.extend([1]*nsample)
            elif group_score<6./8.: 
                difficulty_labels.extend([2]*nsample)
            elif group_score<1.: 
                difficulty_labels.extend([3]*nsample)
            else: 
                difficulty_labels.extend([4]*nsample)
            total += 1
    
        match_results = [x[-1] for x in rets_round1]
        print(f"===> [verbose] multiturn responses toolrate={np.mean(num_toolcalls)}, acc={np.mean(match_results)}, match_results={match_results}")
        samples_list = []
        # groupsize = args.micro_rollout_batch_size
        device = 'cpu'
        groupsize = args.micro_rollout_batch_size
        # astart = self.data_processor.processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        # aend = self.data_processor.processor.tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
        imgpad = self.data_processor.processor.tokenizer.encode("<|image_pad|>", add_special_tokens=False)[0]
        videopad = self.data_processor.processor.tokenizer.encode("<|video_pad|>", add_special_tokens=False)[0]
        pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
        # print(f"!!!! assistant start", astart)
        print(f"===> [verbose] vllm generated {len(all_outputs)} outputs arranged in mrbsz={args.micro_rollout_batch_size}")
        for i in range(0, len(all_outputs), args.micro_rollout_batch_size):
            # prompts = all_messages[i : i + self.strategy.args.micro_rollout_batch_size]
            raw_prompts = all_raw_messages[i : i + self.strategy.args.micro_rollout_batch_size]
            batch_toolcalls = num_toolcalls[i : i + self.strategy.args.micro_rollout_batch_size]
            batch_toolfails = num_toolfails[i : i + self.strategy.args.micro_rollout_batch_size]
            batch_correctness = rets_round1[i : i + self.strategy.args.micro_rollout_batch_size]
            batch_q = questions_cleaned[i : i + self.strategy.args.micro_rollout_batch_size]
            
            batch_flags = all_video_flags[i : i + self.strategy.args.micro_rollout_batch_size]
            batch_qids = qids_expanded[i : i + self.strategy.args.micro_rollout_batch_size]
            batch_uuids = all_uids[i : i + self.strategy.args.micro_rollout_batch_size]
            batch_images = [all_images[uuid] for uuid in batch_uuids]
            batch_conv = [all_conversations[uuid] for uuid in batch_uuids]
            batch_s = []
            diff_labels = difficulty_labels[i : i + self.strategy.args.micro_rollout_batch_size]
            eff_labels = efficiency_labels[i : i + self.strategy.args.micro_rollout_batch_size]
            max_input_len, max_output_len = 0, 0
            
            batch_texts = self.data_processor.processor.apply_chat_template(
                    batch_conv, tokenize=False, add_generation_prompt=False
            )
            batch_texts = [x.strip() for x in batch_texts]
            # _, batch_images = self.data_processor.obtain_conv_images_from_conversations(batch_conv)
            
            ###################
            assert args.micro_rollout_batch_size==1 or is_eval, "mix of video and image only support mrbsz==1"
            video_flag = batch_flags[0]
            padded = False 
            if video_flag:
                this_visual = batch_images[0]
                video = this_visual[0] # list of images 
                imagelist = this_visual[1:] # list of images
                if len(imagelist)==0:
                    this_imglist = [blank_image]
                    new_text = batch_texts[0]+"<|vision_start|><|image_pad|><|vision_end|>"
                    padded = True 
                else:
                    this_imglist = imagelist
                    new_text = batch_texts[0]
                
                visual_inputs = self.data_processor.processor(
                    text=[new_text],
                    images=[this_imglist],
                    videos=[video],
                    padding=True,
                    max_length=20000,
                    add_special_tokens=False,
                    truncation=False,
                    return_tensors="pt",
                )
            else:
                video = [blank_image]
                new_text = batch_texts[0]+"<|vision_start|><|video_pad|><|vision_end|>"
                padded = True 
                
                visual_inputs = self.data_processor.processor(
                    text=[new_text],
                    images=batch_images,
                    videos=[video], # [[blank_image]],
                    padding=True,
                    max_length=20000,
                    add_special_tokens=False,
                    truncation=False,
                    return_tensors="pt",
                )
                # breakpoint()
            
            
            seqlist = visual_inputs['input_ids']
            visual_inputs.pop('input_ids')
            visual_inputs.pop('attention_mask')
            
            expected_imgpad_list = [(gg[-1]*gg[-2]//4).item() for gg in visual_inputs.get('image_grid_thw',[])]
            expected_vidpad_list = [(gg[0]*gg[1]*gg[2]//4).item() for gg in visual_inputs.get('video_grid_thw',[])]
            # breakpoint()
            batch_segment_lengths = []
            q_tokens, a_tokens = [],[]
            for seq, single_conv in zip(seqlist,batch_conv):
                segments = []
                segment = []
                tmp_texts = []
                info = []
                mlist = []
                for entry in single_conv:
                    # current += 1
                    if entry['role'] == 'assistant':
                        segments.append([segment,entry])
                        segment = []
                    else:
                        segment.append(entry)
                
                first_round = True
                
                for q,a in segments: 
                    
                    qtext = self.data_processor.processor.apply_chat_template(
                            q, tokenize=False, add_generation_prompt=True 
                    )
                    
                    if not first_round:
                        start = "<|im_start|>user"
                        qtext = start+qtext.split(start)[-1]
                        mlist.append(qtext.replace("<|image_pad|>","").replace("<|video_pad|>",""))
                        
                    first_round = False 
                    
                    # note: in the second round, a system may be added, which is incorrect
                    has_image = 0
                    has_video = 0
                    for mm in q:
                        if mm['role']=='user' and isinstance(mm['content'],list):
                            for item in mm['content']:
                                if 'image' in item:
                                    has_image +=1
                                elif 'video' in item:
                                    has_video += 1
                                    
                            # if has_image>0: break 
                    
                    atext = a['content'][0]['text']
                    mlist.append(atext)
                    info.append(([qtext,  atext+"<|im_end|>"], has_image, has_video))
                    tmp_texts.extend(info[-1][0])
                
                batch_s.append("".join(mlist))
                encodings = self.data_processor.processor.tokenizer(tmp_texts, add_special_tokens=False)
                lengths = [len(encoding) for encoding in encodings["input_ids"]]
                
                assert len(info)==len(lengths)//2, f"qa length, {len(info)} vs {len(lengths)}"
                # img_idx = 0
                segment_lengths = []
                pos = 0
                initial_q = None
                full_a = []
                imgstart,vidstart = 0,0
                nmax = len(lengths)//2
                niter = 0
                for idx,minf in zip(range(0, len(lengths), 2), info):
                    qlen, alen = lengths[idx:idx+2]
                    alen += 1 # plus the eostoken
                    has_image,has_video = minf[-2:]
                    npad = 0
                    if has_image>0: # image
                        npad = sum(expected_imgpad_list[imgstart:imgstart+has_image]) 
                        imgstart += has_image
                        qlen = qlen - has_image + npad # there is extra image_pad tokens
                        
                    if has_video>0:
                        # expected_vidpad_list[0]
                        npad = expected_vidpad_list[0] # sum(expected_imgpad_list[imgstart:imgstart+has_image]) 
                        vidstart += has_video
                        qlen = qlen - has_video + npad 
                    segment_lengths.extend([qlen, alen]) # for selected frames, there will be a list of frames returned
                    q_ids, a_ids = seq[pos:pos+qlen], seq[pos+qlen:pos+qlen+alen]
                    if nmax-1==niter:
                        a_ids = seq[pos+qlen:] # sometimes the answer is padded with image 
                        
                    if idx>0:
                        full_a.extend([q_ids, a_ids])
                        
                    else:
                        initial_q = q_ids
                        full_a = [a_ids]

                    pos += qlen + alen
                    niter += 1
                    
                batch_segment_lengths.append(segment_lengths)
                
                q_tokens.append(initial_q)
                a_tokens.append(torch.cat(full_a))
                
                # pos = 0
                # for ntok in segment_lengths:
                #     print(f"{self.data_processor.processor.decode(seq[pos:pos+ntok])}")
                #     print('------')
                #     pos += ntok
                # breakpoint()
                max_input_len = max(max_input_len, len(q_tokens[-1]))
                max_output_len = max(max_output_len, len(a_tokens[-1]))
                # npad = seq.eq(pad_token_id).to(float).sum().item()
                
            ######## left and right padding 
            sequences = torch.ones((len(q_tokens), max_input_len+max_output_len), dtype=torch.long) * pad_token_id
            for idx, (qtoken, atoken) in enumerate(zip(q_tokens, a_tokens)):
                sequences[idx][max_input_len-len(qtoken):max_input_len] = qtoken 
                sequences[idx][max_input_len:max_input_len+len(atoken)] = atoken 
                
            #################
            # sequences, attention_mask, action_mask = self.actor.process_sequences(
            #     sequences, max_input_len, eos_token_id, pad_token_id
            # )
            # attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
            attention_mask = sequences.ne(pad_token_id).to(dtype=torch.long)
            sequences = sequences.to(device)
            ntokens_img = [(x==imgpad).to(float).sum().item() for x in sequences]
            ntokens_vid = [(x==videopad).to(float).sum().item() for x in sequences]
            ntokens = [x+y for x,y in zip(ntokens_img, ntokens_vid)]
            
            print("should sum to the same", expected_vidpad_list+expected_imgpad_list, sum(expected_vidpad_list+expected_imgpad_list), ntokens)
            
            attention_mask = attention_mask.to(device)
            # action_mask = sequences.ne(eos_token_id) & sequences.ne(pad_token_id)
            action_mask = create_action_mask_up_to_last_eos(sequences, eos_token_id, pad_token_id)
            
            for idx, slength in enumerate(batch_segment_lengths):
                pos = 0
                for iidx, slen in enumerate(slength):
                    pos += slen
                    if iidx%2==0: 
                        action_mask[idx][pos-slen:pos] = False
            
            action_mask = action_mask[:, max_input_len-1:-1] # this is to align with action log probs
            action_mask = action_mask.to(device)
            na_each = [x.sum().item() for x in action_mask]

            samples_list.append(
                Samples(
                    sequences=sequences,
                    attention_mask=attention_mask,
                    action_mask=action_mask,
                    num_actions=max_output_len,
                    na_each=na_each, 
                    packed_seq_lens=None,
                    response_length=action_mask.float().sum(dim=-1),
                    total_length=attention_mask.float().sum(dim=-1),
                    prompts=raw_prompts,
                    visual_inputs=visual_inputs,
                    round0_nwait=batch_toolcalls,
                    round0_correctness=batch_correctness, # be careful here because each entry is a tuple: valid, norepeat, usefmt, error_info, usecode, final_correct
                    round1_nwait=batch_toolfails,
                    round1_correctness=batch_correctness, # be careful here because each entry is a tuple: valid, norepeat, usefmt, error_info, usecode, final_correct
                    questions=batch_q,
                    solutions=batch_s,
                    qids=batch_qids,
                    round0_ALLTrue=[float(x==4) for x in diff_labels],
                    round0_Easy=[float(x==3) for x in diff_labels],
                    round0_Medium=[float(x==2) for x in diff_labels],
                    round0_Hard=[float(x==1) for x in diff_labels],
                    round0_ALLFalse=[float(x==0) for x in diff_labels],
                    efficiency_label=eff_labels,
                    shaped_rewards=shaped_rewards[i : i + self.strategy.args.micro_rollout_batch_size],
                    uniformity=uniformity[i : i + self.strategy.args.micro_rollout_batch_size],
                    curiosity_bonus=curiosity_bonus[i : i + self.strategy.args.micro_rollout_batch_size],
                    penalty_bonus=penalty_bonus[i : i + self.strategy.args.micro_rollout_batch_size],
                )
            )
            
        return samples_list

    def flush(self):
        "Ensure all experience has been send to critic"
        if self.critic is not None:
            ray.get(self._ref)
            self._ref = None


if __name__ == "__main__":
    sol = "To determine the third step taken if you have suffered from the signs of COVID-19, let's follow the flowchart step by step:\n\n1. **Step 1**: Start isolating.\n2. **Step 2**: Book a test.\n3. **Step 3**: Share contacts via NHS Test and Trace.\n\nThe flowchart clearly indicates that if you have symptoms, you should start by isolating, then book a test, and finally share your contacts with NHS Test and Trace.\n\nTherefore, the third step taken if you have suffered from the signs of COVID-19 is:\n\n\\boxed{Share contacts via NHS Test and Trace}"
    gt = ["\\boxed{share contacts}"]
    print(rule_reward(sol, gt, "<|im_end|>", "none", requires_box=True))
    # outputs = '<|im_start|>system\nYou are a helpful assistant.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"type": "function", "function": {"name": "crop_image_normalized", "description": "Zoom in on the image based on the bounding box coordinates. It is useful when the object or text in the image is too small to be seen.", "parameters": {"type": "object", "properties": {"bbox_2d": {"type": "array", "description": "coordinates for bounding box of the area you want to zoom in. minimum value is 0 and maximum value is 1.", "items": {"type": "number"}}, "target_image": {"type": "number", "description": "The index of the image to crop. Index from 1 to the number of images. Choose 1 to operate on original image."}}, "required": ["bbox_2d", "target_image"]}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><image>\nWhat animal is depicted on the Woolshops Shopping Centre sign?\nchoices:\nA: Lion  \nB: Sheep  \nC: Tiger  \nD: Elephant\n\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\nThe image shows a street with cobblestone pavements, shops, and a signboard for Woolshops Shopping Centre on the left side. The signboard has an animal depicted on it.\n\nTo identify the animal on the sign, I will zoom in on the Woolshops Shopping Centre sign.\n\n<tool_call>\n{"name": "crop_image_normalized", "arguments": {"bbox_2d": [0.1, 0.2, 0.3, 0.4], "target_image":1}}\n</tool_call>'
    # print(parse_last_tool(outputs))
    ################
    # from transformers import AutoProcessor
    # path = "/home/ma-user/work/haozhe/muze/VLM-R1/src/open-r1-multimodal/output/SFT_Qwen_Z2ORCORRECT"
    # processor = AutoProcessor.from_pretrained(path)
    # tools = [CropImageNormalized().function]
    # prompt_maker = NousFnCallPrompt()
    # messages = ["[{\"role\": \"user\", \"content\": [{\"type\": \"image\", \"image\": \"/home/ma-user/work/haozhe/muze/modelartsdata/sa1brl/5394/1.jpg\"}, {\"type\": \"text\", \"text\": \"<image>\\nWhat animal is depicted on the Woolshops Shopping Centre sign?\\nchoices:\\nA: Lion  \\nB: Sheep  \\nC: Tiger  \\nD: Elephant\\n\\nPlease reason step by step, and put your final answer within \\\\boxed{}.\"}]}]"]*5
    # messages = get_required_messages(messages)
    # # print(messages[0])
    # messages = [prompt_maker.preprocess_fncall_messages(
    #                 messages=msg,
    #                 functions=tools, 
    #                 lang=None
    #             ) for msg in messages]
    # # print('=========')
    # toolappended_messages = [[x.model_dump() for x in conversations] for conversations in messages]
    # returns = processor.apply_chat_template(toolappended_messages, tokenize=False, add_generation_prompt=True)
    # for ii in returns:
    #     print(ii)
    #     print('======')