# /*
#  * Original Copyright Haozhe Wang in 2025
#  *
#  * Licensed under the Apache License, Version 2.0 (the "License");
#  */

import os
import os.path
from abc import ABC
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.distributed
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from openrlhf.models import Actor, GPTLMLoss, PolicyLoss, SFTLoss, ValueLoss
from openrlhf.models.utils import masked_mean
from openrlhf.utils.distributed_sampler import DistributedSampler
from openrlhf.models.utils import log_probs_from_logits

from .ppo_utils import AdaptiveKLController, Experience, FixedKLController, NaiveExperienceMaker, NaiveReplayBuffer, DATA_PROCESSOR_MAP
import random 
import copy
import numpy as np
from collections import defaultdict
import json



def read_jsonl(filepath):
    """
    Reads a JSON Lines (jsonl) file and returns a list of dictionaries.

    Args:
        filepath (str): The path to the jsonl file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a line
            from the jsonl file. Returns an empty list if the file is empty
            or if an error occurs.
    """
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON on line: {line.strip()}")
                    # Optionally, you might want to log the error or handle it differently.

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return data

class Evaluator(ABC):
    """
    Trainer for Proximal Policy Optimization (PPO) algorithm.

    Args:
        strategy (Strategy): The training strategy to use.
        actor (Actor): The actor model in the PPO algorithm.
        critic (nn.Module): The critic model in the PPO algorithm.
        reward_model (nn.Module): The reward model for calculating rewards in the RLHF setup.
        initial_model (Actor): The initial model for reference logits to limit actor updates in RLHF.
        ema_model (Actor): The exponential moving average model for stable training.
        actor_optim (Optimizer): The optimizer for the actor model.
        critic_optim (Optimizer): The optimizer for the critic model.
        actor_scheduler (Scheduler): The learning rate scheduler for the actor.
        critic_scheduler (Scheduler): The learning rate scheduler for the critic.
        ema_beta (float, defaults to 0.992): EMA decay rate for model stability.
        init_kl_coef (float, defaults to 0.001): Initial coefficient for KL divergence.
        kl_target (float, optional): Target value for KL divergence.
        kl_horizon (int, defaults to 10000): Horizon for KL annealing.
        ptx_coef (float, defaults to 0): Coefficient for supervised loss from pre-trained data.
        micro_train_batch_size (int, defaults to 8): Micro-batch size for actor training.
        buffer_limit (int, defaults to 0): Maximum size of the replay buffer.
        buffer_cpu_offload (bool, defaults to True): If True, offloads replay buffer to CPU.
        eps_clip (float, defaults to 0.2): Clipping coefficient for policy loss.
        value_clip (float, defaults to 0.2): Clipping coefficient for value function loss.
        micro_rollout_batch_size (int, defaults to 8): Micro-batch size for generating rollouts.
        gradient_checkpointing (bool, defaults to False): If True, enables gradient checkpointing.
        max_epochs (int, defaults to 1): Number of epochs to train.
        max_norm (float, defaults to 1.0): Maximum gradient norm for gradient clipping.
        tokenizer (Callable, optional): Tokenizer for input data.
        prompt_max_len (int, defaults to 128): Maximum length for prompts.
        dataloader_pin_memory (bool, defaults to True): If True, pins memory in the data loader.
        remote_rm_url (str, optional): URL for remote reward model API.
        reward_fn (Callable, optional): Custom reward function for computing rewards.
        save_hf_ckpt (bool): Whether to save huggingface-format model weight.
        disable_ds_ckpt (bool): Whether not to save deepspeed-format model weight. (Deepspeed model weight is used for training recovery)
        **generate_kwargs: Additional arguments for model generation.
    """

    def __init__(
        self,
        strategy,
        ema_beta: float = 0.992,
        init_kl_coef: float = 0.001,
        kl_target: float = None,
        kl_horizon: int = 10000,
        ptx_coef: float = 0,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        eps_clip: float = 0.2,
        value_clip: float = 0.2,
        micro_rollout_batch_size: int = 8,
        gradient_checkpointing: bool = False,
        max_epochs: int = 1,
        max_norm: float = 1.0,
        processor: Optional[Callable[[Any], Dict]] = None,
        tokenizer: Optional[Callable[[Any], Dict]] = None,
        prompt_max_len: int = 128,
        dataloader_pin_memory: bool = True,
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        save_hf_ckpt: bool = False,
        disable_ds_ckpt: bool = False,
        **generate_kwargs,
    ) -> None:
        # assert (
        #     not isinstance(reward_model, List) or len(reward_model) == 1 or reward_fn is not None
        # ), "reward_fn must be specified if using multiple reward models"

        super().__init__()
        self.strategy = strategy
        
        strategy.setup_distributed()
        self.args = strategy.args
        self.rloo_sft = self.args.advantage_estimator.lower() in ['rloo_sft', 'group_sft']
        self.save_hf_ckpt = save_hf_ckpt
        self.disable_ds_ckpt = disable_ds_ckpt
        self.micro_rollout_batch_size = micro_rollout_batch_size
        self.max_epochs = max_epochs
        self.tokenizer = tokenizer
        self.processor = processor
        self.data_processor = None
        # for vlm critic model, not provice processor.
        if self.args.train_vlm and processor is not None:
            self.data_processor = DATA_PROCESSOR_MAP[type(processor)](processor)
            self.tokenizer = self.data_processor.tokenizer

        self.generate_kwargs = generate_kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.max_norm = max_norm
        self.ptx_coef = ptx_coef
        self.micro_train_batch_size = micro_train_batch_size
        self.kl_target = kl_target
        self.prompt_max_len = prompt_max_len
        self.ema_beta = ema_beta
        self.gradient_checkpointing = gradient_checkpointing
        self.reward_fn = reward_fn

        args = self.args
        self.max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
        
        packing_samples = getattr(self.args, "packing_samples", False)
        self.replay_buffer = NaiveReplayBuffer(
            micro_train_batch_size, self.data_processor, buffer_limit, buffer_cpu_offload, packing_samples,
            drop_maxlen=self.args.drop_maxlen, 
            maxlen=self.args.generate_max_len + prompt_max_len,
        )

        self.iter = 0 
        self.eval_step = 0
        self.best = -1

    def eval_unit(self, args, ep, global_step, dataloader):
        keys = ['reward', 'response_length', 'validity','match','usefmt','round1_nwait']
        infos = {k:[] for k in keys}
        print("!!!! eval loader size", len(dataloader), 'step', global_step)
        batchsize = dataloader.batch_sampler.batch_size
        for idx, rand_prompts in enumerate(dataloader):
            if batchsize>len(rand_prompts): 
                current_len = len(rand_prompts)
                needed = batchsize - current_len
                repeat_indices = np.arange(needed) % current_len
                # repeat_indices = repeat_indices.to(rand_prompts.device)
                additional = [rand_prompts[ii] for ii in repeat_indices]
                rand_prompts = rand_prompts + additional
            else: needed = 0
            print(f"!!!! ========== eval progress {idx}/{len(dataloader)} ==========")
            
            exp_list = self.get_explist_from_prompts(args, ep, rand_prompts, is_eval=True, eval_step=global_step)
            
            for i, experience in enumerate(exp_list):
                self.replay_buffer.append_split(experience, is_eval=True)
            
        
        for item in self.replay_buffer.eval_items:
            info = item.info
            for k in keys:
                infos[k].append(info[k])
        out_lens = infos['response_length']
        
        for k,vlist in infos.items():
            infos[k] = np.mean(vlist)
        infos['generation_exceed_rate'] = np.mean([x>args.generate_max_len-1 for x in out_lens])
        
        torch.distributed.barrier()
        gather_info = self.strategy.all_reduce(infos) # mean 
        
        return gather_info
            
            
        
    def get_eval_result_from_disk(self):
        args = self.strategy.args
        from glob import glob 
        # os.makedirs(args.ckpt_path, exist_ok=True)
        # os.makedirs(f'{args.ckpt_path}/logs', exist_ok=True)
        tmp = f'{args.ckpt_path}/logs/sample.eval_iter{self.eval_step}*.jsonl'
        files = glob(tmp)
        print(f'!!!! [eval] reading from disk {len(files)} files', tmp, )
        
        datalist = [read_jsonl(file) for file in files]
        results_each = defaultdict(list)
        q2results = defaultdict(list)
        for info in datalist:
            for x in info:
                qid = x['qids']
                res = x.get('match')
                if res is None:
                    r0_res = x['round0_correctness']
                    res = r0_res 
                    
                q2results[qid].append(res>0.5)
        # We compute query-wise mean acc, and then average them
        # this is a trick to handle the drop_last=False issue
        for qid, vlist in q2results.items():
            bench = qid.split('-')[0]
            macc = np.mean(vlist)
            results_each[bench].append(macc)
        all_results = []
        dump_info = []
        modelpath = args.pretrain
        for k in results_each.keys():
            nc = np.sum(results_each[k])
            num  = len(results_each[k]) 
            dump_info.append(dict(benchname=k, pass1=nc/num, ncorrect=nc, ntotal=num, modelpath=modelpath))
            print(f'!!!! [eval] from disk bench={k}, acc={np.mean(results_each[k])}={nc}/{num}')
            all_results.extend(results_each[k])
            results_each[k] = np.mean(results_each[k])
        
        json.dump(dump_info, open(f'{args.ckpt_path}/logs/metrics_iter{self.eval_step}.json', 'w'))
        acc = np.mean(all_results)
        return acc, results_each
        
    def fill_replay_buffer(self, buffer, num_expected):
        # Ensure every item in buffer appears at least once
        for item in buffer[:num_expected]:
            self.replay_buffer.append_split(item)
        
        # Fill the remaining slots with random choices from buffer
        remaining_slots = num_expected - len(buffer)
        if remaining_slots>0:
            for _ in range(remaining_slots):
                item = random.choice(buffer)
                self.replay_buffer.append_split(item)
        print(f'!!!! rbuffersize after filling: {len(self.replay_buffer)} should be {num_expected} x nsamples_per_query', )
        # assert len(self.replay_buffer)==num_expected
        
    def get_explist_from_prompts(self, args, ep, all_prompts, append=False, is_eval=False, force_noprefix=False, eval_step=None):
        autocode = getattr(args, "prefix_generation", None)
        requires_group = getattr(args, "advantage_estimator", None) in ['']
        # print('!!!! requires group', requires_group)
        generate_kwargs = copy.copy(self.generate_kwargs)
        generate_kwargs['requires_group'] = requires_group
        if force_noprefix:
            pass
        elif autocode=='autocode':
            if ep==0:
                new_prompts = [x+prefix for x in all_prompts for prefix in self.prefixes[:2]]
                all_prompts = new_prompts
                generate_kwargs['prefix_lengths'] = [plen for x in all_prompts for plen in self.prefix_lengths[:2]]
            else:
                new_prompts = [x+prefix for x in all_prompts for prefix in self.prefixes[2:3]]
                all_prompts = new_prompts
                generate_kwargs['prefix_lengths'] = [plen for x in all_prompts for plen in self.prefix_lengths[2:3]]
        elif autocode=='autocode1':
            # if ep==0:
            new_prompts = [x+prefix for x in all_prompts for prefix in self.prefixes[:2]]
            all_prompts = new_prompts
            generate_kwargs['prefix_lengths'] = [plen for x in all_prompts for plen in self.prefix_lengths[:2]]
        # else:
        #     new_prompts = [x+prefix for x in all_prompts for prefix in self.prefixes[2:3]]
        #     all_prompts = new_prompts
        #     generate_kwargs['prefix_lengths'] = [plen for x in all_prompts for plen in self.prefix_lengths[2:3]]
        elif autocode=='autocode2':
            # if ep==0:
            new_prompts = [x+prefix for x in all_prompts for prefix in self.prefixes[:3]]
            all_prompts = new_prompts
            generate_kwargs['prefix_lengths'] = [plen for x in all_prompts for plen in self.prefix_lengths[:3]]
        elif autocode=='autocode_continue':
            # if ep==0:
            new_prompts = [x+prefix for x in all_prompts for prefix in self.prefixes[3:5]]
            all_prompts = new_prompts
            generate_kwargs['prefix_lengths'] = [plen for x in all_prompts for plen in self.prefix_lengths[3:5]]
        elif append and autocode=="autocode_append":
            new_prompts = [x+prefix for x in all_prompts for prefix in self.prefixes[5:6]]
            all_prompts = new_prompts
            generate_kwargs['prefix_lengths'] = [plen for x in all_prompts for plen in self.prefix_lengths[5:6]]
            
        return self.experience_maker.make_experience_list(all_prompts, is_eval=is_eval, eval_step=eval_step, **generate_kwargs)
        
            
    def evaluate(
        self,
        args,
        eval_data
    ) -> None:
        
        tmp = eval_data
        eval_bsz = args.micro_rollout_batch_size 
        eval_dataloader = self.strategy.setup_dataloader(
            tmp,
            eval_bsz, # should larger than world size?
            True,
            True,
            drop_last=False
            )
        print(f'!!!! eval dataloader size', len(eval_dataloader), 'eval_bsz', eval_bsz)
        self.eval_dataloader = eval_dataloader
        if len(eval_data)==0 or len(eval_dataloader)==0: print('!!!! no eval data, eval_data should be larger than num_vllm * micro_bsz', len(eval_data), len(eval_dataloader))
        else: print(f'!!!! eval data {len(eval_data)} eval dataloader', len(eval_dataloader), args.micro_rollout_batch_size)
        info = self.eval_unit(args, 0, self.eval_step, eval_dataloader)
        eval_result = info['match']
        torch.distributed.barrier()
        result2, bench_results = self.get_eval_result_from_disk()
        print(f'!!!! [eval] finish with step {self.eval_step} rank {self.strategy.get_rank()} gathered eval stats', info, 'from disk:', result2)
        
        self.eval_step += 1
        # info['match_overall'] = result2
        for k,v in bench_results.items():
            info[f'match_{k}'] = v
        info['match_overall'] = result2
        eval_save = self.best<=result2 #  and args.rollout_batch_size>16
        if eval_save:
            self.best = result2
            print(f"!!!! [eval] saving with average score {self.best}")

        del eval_dataloader
        self.replay_buffer.eval_items.clear()
        
