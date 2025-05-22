# /*
#  * Modified by Haozhe Wang in 2025
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

class PPOTrainer(ABC):
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
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        ema_model: Actor,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        actor_scheduler,
        critic_scheduler,
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
        remote_rm_url: str = None,
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        save_hf_ckpt: bool = False,
        disable_ds_ckpt: bool = False,
        **generate_kwargs,
    ) -> None:
        assert (
            not isinstance(reward_model, List) or len(reward_model) == 1 or reward_fn is not None
        ), "reward_fn must be specified if using multiple reward models"

        super().__init__()
        self.strategy = strategy
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

        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.ema_model = ema_model
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.actor_scheduler = actor_scheduler
        self.critic_scheduler = critic_scheduler

        sft_only = getattr(self.strategy.args, "loss_version", "none") == "sft_only"
        if sft_only:
            self.actor_loss_fn = SFTLoss(eps_clip, rloo_sft=self.rloo_sft)
        else: self.actor_loss_fn = PolicyLoss(eps_clip, rloo_sft=self.rloo_sft)
        self.critic_loss_fn = ValueLoss(value_clip)
        self.ptx_loss_fn = GPTLMLoss()

        self.freezing_actor_steps = getattr(self.args, "freezing_actor_steps", -1)

        # Mixtral 8x7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(init_kl_coef, kl_target, kl_horizon)
        else:
            self.kl_ctl = FixedKLController(init_kl_coef)
        print(f'!!!! using kl coef (init={init_kl_coef}, target={kl_target}), sft coef = {ptx_coef}, bsz = {micro_train_batch_size}')
        args = self.args
        self.max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
        
        self.experience_maker = NaiveExperienceMaker(
            actor,
            critic,
            reward_model,
            initial_model,
            tokenizer,
            self.data_processor,
            prompt_max_len,
            self.kl_ctl,
            strategy,
            remote_rm_url,
            reward_fn,
        )
        packing_samples = getattr(self.args, "packing_samples", False)
        self.replay_buffer = NaiveReplayBuffer(
            micro_train_batch_size, self.data_processor, buffer_limit, buffer_cpu_offload, packing_samples,
            drop_maxlen=self.args.drop_maxlen, 
            maxlen=self.args.generate_max_len + prompt_max_len,
            train_batch_size=self.args.train_batch_size,
            use_pos=("sft" in getattr(self.args, "advantage_estimator", "None")) and (getattr(self.args, "aux_loss_coef", 0.0)>0.0)
        )

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/epoch")
            wandb.define_metric("eval/*", step_metric="eval/epoch", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)
            
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
                # results_each[bench].append(x['match']>0.5)
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
        
                
    def train_unit(self, args, pbar, steps, ep, eval_data=None): # for an episode
        print(f"===> [verbose] trainer.train_unit() @Epoch={ep}")
        total_consumed = 0 
        num_real = 0 
        num_rounds = len(self.prompts_dataloader)
        target_prefix_generation_rounds = 2 
        freq = num_rounds//target_prefix_generation_rounds
        is_debug = args.rollout_batch_size < 16
        eval_only = getattr(args, "training_mode", "train") == 'eval_only'
        no_eval =  getattr(args, "training_mode", "train") == 'no_eval'
        small_eval = getattr(args, "training_mode", "train") == 'small_eval'
        assert (eval_only and eval_data) or not eval_only, "eval_only mode should have eval_data"
        savepath = None
        for idx, rand_prompts in enumerate(self.prompts_dataloader):
            num_expected = len(rand_prompts)
            
            eval_save = False
            if eval_data and not no_eval and ((args.eval_steps>0 and (idx+1)%args.eval_steps==0) or args.eval_steps<=0):
                print(f'!!!! doing evaluation @Step{steps}', eval_data)
                if eval_only:
                    tmp = eval_data 
                elif small_eval:
                    tmp = eval_data[:512]
                else: tmp = eval_data
                # make sure eval_bsz*nsample%num_vllm == 0
                
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
                info = self.eval_unit(args, ep, self.eval_step, eval_dataloader)
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
                    print(f"!!!! [eval] saving {savepath} with average score {self.best}")
    
                client_states = {"consumed_samples": steps * args.rollout_batch_size}
                self.save_logs_and_checkpoints(args, steps, pbar, info, client_states, is_eval=True, eval_save=eval_save)
                del eval_dataloader
                self.replay_buffer.eval_items.clear()
                if eval_only: 
                    print('!!!! [eval] exiting')
                    break 
                if savepath is not None and eval_save and self.strategy.is_rank_0():
                    newpath = f"{savepath}_evalbest"
                    os.rename(savepath, newpath)
                    print(f"!!!! [eval] renaming {savepath}->{newpath}")
            
            print(f"===> [rbuffer] {len(rand_prompts)} queries @Epoch{ep}-RBufferNo{idx}(Total={num_rounds} for full Epoch)")
            exp_list = self.experience_maker.make_experience_list(rand_prompts, is_eval=False, eval_step=None, **self.generate_kwargs)
            print(f"===> [rbuffer] @Epoch{ep}-RBufferNo{idx}(Total={num_rounds} for full Epoch) done experiences, split to rbuffer items")
            
            for i, experience in enumerate(exp_list): # for a replaybuffer batch
                self.replay_buffer.append_split(experience)
                num_real += 1 
                
            total_consumed += num_expected
            
            torch.cuda.empty_cache()
            if args.advantage_estimator in ['grpo','gloo'] or getattr(args, "buffer_norm", 1)==0:
                print('!!!! [rbuffer] not using buffer norm')
            else: self.replay_buffer.normalize("advantages", self.strategy)
            
            print('!!!! [rbuffer] done.')
            print('===> [verbose] waiting for all actors')
            torch.distributed.barrier()
            status = self.ppo_train(steps)
            self.replay_buffer.clear()
            torch.cuda.empty_cache()

            if "kl" in status:
                self.kl_ctl.update(status["kl"], args.rollout_batch_size * args.n_samples_per_prompt)
            
            status['num_real_samples'] = num_real
            num_real = 0 
            pbar.set_postfix(status)

            # logs/checkpoints
            client_states = {"consumed_samples": steps * args.rollout_batch_size}
            self.save_logs_and_checkpoints(args, steps, pbar, status, client_states)
            tag = f"global_step{steps}" 
            if (steps +1)%2 == 0:
                savepath = self._save_checkpoint(args, tag, client_states)
            pbar.update()
            steps = steps + 1
        return steps 
            
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
        print(f"===> [verbose] trainer.get_explist_from_prompts() @Epoch={ep}")
        autocode = getattr(args, "prefix_generation", None)
        requires_group = getattr(args, "advantage_estimator", None) in ['']
        generate_kwargs = copy.copy(self.generate_kwargs)
        generate_kwargs['requires_group'] = requires_group
        return self.experience_maker.make_experience_list(all_prompts, is_eval=is_eval, eval_step=eval_step, **generate_kwargs)
        
            
    def fit(
        self,
        args,
        prompts_dataloader,
        pretrain_dataloader,
        consumed_samples=0,
        num_update_steps_per_episodes=1,
        eval_data=None
    ) -> None:
        num_rollouts_per_episodes = (
            num_update_steps_per_episodes
            * args.train_batch_size
            // args.max_epochs
            // args.rollout_batch_size
            // args.n_samples_per_prompt
        ) # num replay buffer 

        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_rollouts_per_episodes  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        self.prompts_dataloader = prompts_dataloader
        self.pretrain_dataloader = pretrain_dataloader

        # Restore step and start_epoch
        steps = consumed_samples // args.rollout_batch_size + 1
        print(f'===> [config] one episode = rbuffersize ({args.rollout_batch_size} queries) * num sampler sync (rbuffer clear frequency) {num_rollouts_per_episodes}\n', 
              f"""num_grad_steps_in_episode ({num_update_steps_per_episodes})
            * train_bsz ({args.train_batch_size}) QAs
            // {args.max_epochs} epoch
            // {args.rollout_batch_size} queries in rbuffer
            // {args.n_samples_per_prompt} qa_per_query"""
            )
        start_episode = consumed_samples // args.rollout_batch_size // num_rollouts_per_episodes
        consumed_samples = consumed_samples % (num_rollouts_per_episodes * args.rollout_batch_size)
        eval_only = getattr(args, "training_mode", "train") == 'eval_only'
        if eval_only: 
            print('!!!! [eval] eval only mode')
            rg = range(1)
        else: rg = range(start_episode, args.num_episodes)
        for episode in rg:
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(
                    episode, consumed_samples=0 if episode > start_episode else consumed_samples
                )
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                disable=not self.strategy.is_rank_0(),
            )

            steps = self.train_unit(args, pbar, steps, episode, eval_data)
        
        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    def ppo_train(self, global_steps=0):
        torch.cuda.empty_cache()
        do_filter = not (getattr(self.strategy.args, "filter", "False") == "False")
        do_ssr = getattr(self.strategy.args, "filter", "False") == "SSR"
        # if do_filter:
        rbuffer_status = self.replay_buffer.active_sampling(do_ssr=do_ssr)
        rbuffer_status = self.strategy.all_reduce(rbuffer_status)
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        print(f"training nsteps={len(dataloader)}=buffersize{len(self.replay_buffer)}/microsize{self.replay_buffer.sample_batch_size}\nSearch for gradient_accumulation_steps, the real global step should be <nsteps>/<gradaccumsteps>")
        device = torch.cuda.current_device()

        status_list = []
        status_mean = {}
        
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            for experience in pbar:
                experience.to_device(device)
                status = self.training_step(experience, global_steps)

                # for DP
                # weighted mean for kl
                if "kl" in status:
                    status["kl"] *= status["response_length"]
                    status = self.strategy.all_reduce(status)
                    status["kl"] /= status["response_length"]
                    

                short_status = {}

                if "policy_loss" in status:
                    short_status = {
                        "pg": status["policy_loss"],
                        "rm": status["reward"],
                        "validity": status["validity"],
                        # "ret": status["return"],
                        "glen": status["response_length"],
                        # "tlen": status["total_length"],
                        "kl": status["kl"],
                        "act_lr": status["actor_lr"],
                        
                    }

                if "critic_loss" in status:
                    short_status["cri"] = status["critic_loss"]
                    short_status["vals"] = status["values"]
                    short_status["cri_lr"] = status["critic_lr"]

                if "ptx_loss" in status:
                    short_status["ptx"] = status["ptx_loss"]

                status_list.append(status)
                # pbar.set_postfix(short_status)

        if status_list:
            status_mean = status_list[0]
            # print('!!! [debug] status', status_mean)
            cnt = defaultdict(int)
            for m in status_list[1:]:
                for k, v in m.items():
                    
                    status_mean[k] += v
                    if k in ['weighted_pos_logp', 'weighted_neg_logp']:
                        cnt[k] += 1.0 if v!=0. else 0.0 
                    
            for k in status_mean.keys():
                if k in ['weighted_pos_logp', 'weighted_neg_logp']:
                    status_mean[k] /= (cnt[k]+1e-4)
                else: status_mean[k] /= len(status_list)
        torch.cuda.empty_cache()
        status_mean.update(rbuffer_status)
        return status_mean

    def training_step(self, experience: Experience, global_steps, **kwargs) -> Dict[str, float]:
        status = {}
        kwargs['global_steps'] = global_steps
        if global_steps > self.freezing_actor_steps:
            status = self.training_step_actor(experience, **kwargs)
        if self.critic is not None:
            status.update(self.training_step_critic(experience))
        return status

    def training_step_actor(self, experience: Experience, **kwargs) -> Dict[str, float]:
        # print('!!!! start training')
                
        self.actor.train()
        validity = experience.validity
        
        packing = getattr(self.strategy.args, "packing_samples", False)
        
        diffs = experience.info['difficulty']
        waits = None
        if packing:
            sequences = torch.cat(experience.sequences, dim=0).unsqueeze(0)
            old_action_log_probs = torch.cat(experience.action_log_probs, dim=0).unsqueeze(0)
            advantages = torch.cat(experience.advantages, dim=0).unsqueeze(0)
            num_actions = [v.numel() for v in experience.advantages]
            rewards = experience.info['reward']
            raw_rewards = []
            for ritem, na in zip(rewards, num_actions):
                raw_rewards.extend([ritem]*na)
            raw_rewards = torch.stack(raw_rewards).to(sequences.device).unsqueeze(0)
            packed_seq_lens = [s.numel() for s in experience.sequences]
            attention_mask = torch.cat(
                [torch.full_like(s, i + 1) for i, s in enumerate(experience.sequences)], dim=0
            ).unsqueeze(0)
            visual_inputs = experience.visual_inputs
            if validity is not None:
                validity = torch.cat([torch.ones_like(adv)*vv for vv,adv in zip(validity, experience.advantages)], dim=0).unsqueeze(0)
                # print('!!!! should match', validity.shape, advantages.shape)
        else:
            sequences = experience.sequences
            old_action_log_probs = experience.action_log_probs
            advantages = experience.advantages
            num_actions = experience.action_mask.size(1)
            packed_seq_lens = None
            attention_mask = experience.attention_mask
            visual_inputs = experience.visual_inputs
            if validity is not None:
                assert len(validity)==len(sequences), f"{len(validity)}, {len(sequences)}"
                # validity = torch.FloatTensor(validity).unsqueeze(-1).expand_as(advantages).to(sequences.device)
                validity = torch.FloatTensor(validity).unsqueeze(-1).repeat(1,num_actions).to(sequences.device)
            
            # rewards = experience.info['reward'] # tensor of (bsz,)
            waits = experience.info['round1_nwait']
                

        # actor loss
        action_log_probs, output = self.actor(
            sequences, # left padded
            num_actions,
            attention_mask=attention_mask,
            return_output=True,
            packed_seq_lens=packed_seq_lens,
            visual_inputs=visual_inputs
        )
        action_entropy = output['action_entropy']
        # loss function
        self.iter += 1
        kl_penalty_coef = getattr(self.strategy.args, "kl_penalty_coef", 0.0) # == "sft_only"
        
        actor_loss_dict = self.actor_loss_fn(
            action_log_probs,
            old_action_log_probs,
            advantages,
            action_mask=experience.action_mask,
            kl_coef=self.kl_ctl.value,
            validity=validity,
            return_dict=True,
            raw_rewards=waits,
            action_entropy=action_entropy
        )
        
        actor_loss = actor_loss_dict.get('actor_loss', 0.0)
        aux_loss = actor_loss_dict.get("sft_loss", 0.0)
        entropy_loss = -actor_loss_dict.get("allneg_entropy", 0.0)
        
        print('!!!! [training] step', kwargs['global_steps'], f'with {len(experience.sequences)} sample, shape={sequences.shape}')
        sft_only = getattr(self.strategy.args, "loss_version", "none") == "sft_only"
        skip = False 
        if sft_only: loss = aux_loss
        else: 
            advlist = [x[-1].item() for x in experience.advantages]
            nonzero = np.mean([x!=0 for x in advlist])
            print(f'!!!! [training] adv nonzero ratio', nonzero, 'kl penalty ratio', kl_penalty_coef)
            loss = actor_loss + aux_loss * self.args.aux_loss_coef + kl_penalty_coef*actor_loss_dict.get("kl_penalty",0.0) + self.args.entropy_loss_coef * entropy_loss
            print('!!!! [training] iter', self.iter, f'actorloss={actor_loss.item()}, sftloss={aux_loss if isinstance(aux_loss,float) else aux_loss.item()}, final={loss.item()}, reward={experience.info["reward"]}, adv={advlist}, waits={experience.info["round1_nwait"]}, val={experience.validity}')
            
        if not skip: 
            self.strategy.backward(loss, self.actor, self.actor_optim)
        del action_entropy
        # ptx loss
        ptx_loss = None
        if self.pretrain_dataloader is not None:
            data = next(self.pretrain_dataloader)
            inputs = data[1].squeeze(1).to(torch.cuda.current_device())
            attention_mask = data[2].squeeze(1).to(torch.cuda.current_device())
            label = torch.where(
                attention_mask.bool(),
                inputs,
                self.ptx_loss_fn.IGNORE_INDEX,
            )

            output = self.actor(inputs, attention_mask=attention_mask, return_output=True)
            ptx_log_probs = output["logits"]

            # loss function
            ptx_loss = self.ptx_loss_fn(ptx_log_probs, label)
            # mixtral
            if self.aux_loss:
                aux_loss = output.aux_loss
            else:
                aux_loss = 0
            loss = ptx_loss + aux_loss * self.args.aux_loss_coef
            self.strategy.backward(self.ptx_coef * loss, self.actor, self.actor_optim)
        
        # status
        if sft_only:
            status = {"effective_loss": loss.item(),"actor_lr": self.actor_scheduler.get_last_lr()[0],}
        else:
            status = {"policy_loss": actor_loss.item(), "actor_lr": self.actor_scheduler.get_last_lr()[0], "validity":",".join([str(round(x.max().item(),1)) for x in validity]),
                  "adv_nonzero": nonzero, "effective_loss": loss.item()}
    
        status.update(actor_loss_dict)
        if ptx_loss is not None:
            status["ptx_loss"] = ptx_loss.item()
        print(f"===> [verbose] iter {self.iter} collecting status")
        for k, v in experience.info.items():
            # print(k,v)
            if k == "kl":
                out_tokens = experience.info["response_length"] # list 
                nomin = sum([x*y for x,y in zip(v,out_tokens)])
                denom = sum(out_tokens)
                status[k] = nomin/denom
            elif k in {'num_actions', 'round1_diff'}: continue
            elif k=='round1_correctness': 
                # continue
                r1c = experience.info['round1_correctness']
                r0c = experience.info['round0_correctness']
                tmp = dict(round1_average_improvement=0.0, round1_negative_rate=0.0, round1_positive_rate=0.0, round1_percentage=0.0)
                final_results = []
                diff = []
                for aa,bb in zip(r0c, r1c):
                    if bb<0: # not forced rethinking
                        final_results.append(aa)
                        continue 
                    else: 
                        final_results.append(bb)
                        diff.append(bb-aa)
                valid_vlist = diff 
                tmp['round0_correctness'] = np.mean(final_results)
                if len(valid_vlist)>0:
                    tmp['round1_average_improvement'] = np.mean(valid_vlist)
                    tmp['round1_negative_rate'] = np.mean([x<0 for x in valid_vlist])
                    tmp['round1_positive_rate'] = np.mean([x>0 for x in valid_vlist])
                    tmp['round1_percentage'] = len(valid_vlist)/len(r1c)
                # print(f"!!!! [debug] round1 correctness {tmp}")
                status.update(tmp)
            else:
                if isinstance(v[0], str): continue 
                status[k] = v.mean().item() if isinstance(v, torch.Tensor) else np.mean(v)
        
        num_exceed = np.mean([x>=self.strategy.args.generate_max_len-1 for x in experience.info['response_length']])
        status['generation_exceed_rate'] = num_exceed
        
        if skip: return status 
        
        self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")
        
        if self.ema_model:
            self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cuda")

        del experience.info, experience.sequences, experience.action_log_probs, experience.advantages, experience, loss
        # print(status)
        return status

    def training_step_critic(self, experience: Experience) -> Dict[str, float]:
        self.critic.train()

        # TODO: this is a bad indicator to say that data is packed...
        if isinstance(experience.sequences, list):
            sequences = torch.cat(experience.sequences, dim=0).unsqueeze(0)
            old_values = torch.cat(experience.values, dim=0).unsqueeze(0)
            returns = torch.cat(experience.returns, dim=0).unsqueeze(0)
            num_actions = [v.numel() for v in experience.advantages]
            packed_seq_lens = [s.numel() for s in experience.sequences]
            attention_mask = torch.cat(
                [torch.full_like(s, i + 1) for i, s in enumerate(experience.sequences)], dim=0
            ).unsqueeze(0)
            visual_inputs = experience.visual_inputs
        else:
            sequences = experience.sequences
            old_values = experience.values
            returns = experience.returns
            num_actions = experience.action_mask.size(1)
            packed_seq_lens = None
            attention_mask = experience.attention_mask
            visual_inputs = experience.visual_inputs

        # critic loss
        values, output = self.critic(
            sequences,
            num_actions=num_actions,
            attention_mask=attention_mask,
            return_output=True,
            packed_seq_lens=packed_seq_lens,
            visual_inputs=visual_inputs,
        )
        # loss function
        critic_loss = self.critic_loss_fn(
            values,
            old_values,
            returns,
            action_mask=experience.action_mask,
        )
        # mixtral
        if self.aux_loss:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0
        loss = critic_loss + aux_loss * self.args.aux_loss_coef
        self.strategy.backward(loss, self.critic, self.critic_optim)
        self.strategy.optimizer_step(self.critic_optim, self.critic, self.critic_scheduler, name="critic")

        # status
        status = {
            "critic_loss": critic_loss.item(),
            "values": masked_mean(values, experience.action_mask).item(),
            "critic_lr": self.critic_scheduler.get_last_lr()[0],
        }
        return status

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}, is_eval=False, eval_save=False):
        if global_step % args.logging_steps == 0 or is_eval:
            # wandb
            tagname = 'eval' if is_eval else 'train'
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {
                    f"{tagname}/{k}": v
                    for k, v in {
                        **logs_dict,
                        "global_step": global_step,
                    }.items()
                }
                if self.experience_maker.perf_stats is not None and not is_eval:
                    logs.update({f"perf/experience_maker/{k}": v for k, v in self.experience_maker.perf_stats.items()})
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)
                if self.experience_maker.perf_stats is not None:
                    for k, v in self.experience_maker.perf_stats.items():
                        self._tensorboard.add_scalar(f"perf/experience_maker/{k}", v, global_step)

        
        # if eval_save:
        #     if self.strategy.is_rank_0(): 
        #         print(f'!!!! [eval] step {self.eval_step} saving ', self.best)
                
        # if (args.save_steps>0 and global_step % args.save_steps == 0):
        #     tag = f"global_step{global_step}" 
        #     self._save_checkpoint(args, tag, client_states)
            
        # if eval_save:
        #     if self.strategy.is_rank_0(): 
        #         save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
        #         os.rename(save_path, save_path+f'_evalbest')
        #         print(f'!!!! [eval] step {self.eval_step} saving {self.best} with name', save_path+f'_evalbest')
                
                


    # def _save_checkpoint(self, args, tag, client_states):
    #     print('!!!! [saving] inside trainer save_checkpoint')
    #     if not self.disable_ds_ckpt:
    #         self.strategy.save_ckpt(
    #             self.actor.model,
    #             os.path.join(args.ckpt_path, "_actor"),
    #             tag,
    #             args.max_ckpt_num,
    #             args.max_ckpt_mem,
    #             client_states,
    #         )
    #         if self.critic is not None:
    #             self.strategy.save_ckpt(
    #                 self.critic, os.path.join(args.ckpt_path, "_critic"), tag, args.max_ckpt_num, args.max_ckpt_mem
    #             )

    #     if self.save_hf_ckpt:
    #         save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
    #         self.strategy.save_model(self.actor, self.processor or self.tokenizer, save_path)
