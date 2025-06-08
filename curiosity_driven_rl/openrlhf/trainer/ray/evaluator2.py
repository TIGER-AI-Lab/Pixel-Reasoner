import itertools
import math
import os
import socket
from typing import Callable, Dict, List

import deepspeed
import ray
import torch
import torch.distributed
from transformers.trainer import get_scheduler

from openrlhf.datasets import PromptDataset
from openrlhf.models import Actor
from openrlhf.trainer import PPOTrainer, Evaluator
from openrlhf.trainer.ppo_utils import Experience, RemoteExperienceMaker
from openrlhf.utils import blending_datasets, get_tokenizer, get_vl_processor
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.distributed_util import init_process_group

from .launcher import BasePPORole
from .utils import get_physical_gpu_id
import json
import shutil


class Evaluator2(Evaluator):
    def __init__(
        self,
        *args,
        vllm_engines: List = None,
        remote_rm_url: List[str] = None,
        critic_train_remote: bool = False,
        **kwargs,
    ):
        """PPOTrainer for ray.

        Args:
            vllm_engines (List, optional): vllm engines for text generation, if not specified, generate text by actor model directly. Defaults to None.
            critic_train_remote (bool, optional): whether this actor should triger corresponding critic model training. Defaults to False.
        """
        super().__init__(*args, **kwargs)
        # self.remote_rm_url = remote_rm_url
        self.vllm_engines = vllm_engines
        # self.critic_train_remote = critic_train_remote
        args = self.strategy.args
        train_data = getattr(args, 'prompt_data',None)
        eval_data = getattr(args, "eval_data",None)
        self.gt_path = [train_data, eval_data]
        print('!!!! gts', self.gt_path)
        self.modelfamily = kwargs.get('modelfamily', 'qwen')
        self.experience_maker = RemoteExperienceMaker(
            None,
            None,
            None,
            None,
            self.tokenizer,
            self.data_processor,
            self.prompt_max_len,
            0.0, # self.kl_ctl,
            self.strategy,
            None,
            self.reward_fn,
            vllm_engines=self.vllm_engines,
            packing_samples=self.strategy.args.packing_samples,
            gt_path=self.gt_path, 
            modelfamily=self.modelfamily
        )

        backend = getattr(self.strategy.args, "vllm_sync_backend", "nccl")
        self.use_cuda_ipc = False
        # if backend == "nccl" and self.strategy.args.colocate_all_models:
        #     self.use_cuda_ipc = True

        # Create torch group with deepspeed rank 0 and all vllm ranks
        # to update vllm engine's weights after each training stage.
        #
        # Say we have 3 vllm engines and eache of them has 4 GPUs,
        # then the torch group is:
        # [    0,      1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12]
        # |ds rank 0 |  engine-0  |  engine-1  |   engine-2   |
        #
        # For ZeRO-1/2:
        #   1. Broadcast parameters from rank 0 to all vllm engines
        # For ZeRO-3:
        #   1. AllGather paramters to rank 0
        #   2. Broadcast parameters from rank 0 to all vllm engines
        if self.vllm_engines is not None and not self.use_cuda_ipc and torch.distributed.get_rank() == 0:
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]

            vllm_num_engines, vllm_tensor_parallel_size = (
                self.strategy.args.vllm_num_engines,
                self.strategy.args.vllm_tensor_parallel_size,
            )
            world_size = vllm_num_engines * vllm_tensor_parallel_size + 1

            use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
            group_name = "openrlhf"
            refs = [
                engine.init_process_group.remote(
                    master_address,
                    master_port,
                    i * vllm_tensor_parallel_size + 1,
                    world_size,
                    group_name,
                    backend=backend,
                    use_ray=use_ray,
                )
                for i, engine in enumerate(self.vllm_engines)
            ]
            if use_ray:
                import ray.util.collective as collective

                collective.init_collective_group(world_size=world_size, rank=0, backend=backend, group_name=group_name)
                self._model_update_group = group_name
            else:
                self._model_update_group = init_process_group(
                    backend=backend,
                    init_method=f"tcp://{master_address}:{master_port}",
                    world_size=world_size,
                    rank=0,
                    group_name=group_name,
                )

            ray.get(refs)

        torch.distributed.barrier()

    def evaluate(self, global_steps):
        # 1. ensure all experience makers done
        self.experience_maker.flush()
        torch.distributed.barrier()
        status = {}
        strategy = self.strategy
        args = self.strategy.args
        eval_dp = getattr(args, "eval_data", None)
        assert eval_dp, f"args.eval_data: {eval_dp} is invalid"
        # if eval_dp: 
        eval_data = blending_datasets(
            eval_dp,
            args.prompt_data_probs,
            strategy,
            args.seed,
            max_count=args.max_samples,
            return_eval=False,
            train_split=args.prompt_split,
        )
        
        self.eval_data = PromptDataset(
            eval_data, self.tokenizer, strategy, input_template=args.input_template, is_eval=True, processor=self.processor
        )
        print('!!!!! eval data', len(eval_data), eval_dp)
        print(self.eval_data)
        status = super().evaluate(
            args, self.eval_data
        )
        torch.cuda.empty_cache()
        torch.distributed.barrier()

        return status

    def training_step(self, experience: Experience, global_steps, **kwargs) -> Dict[str, float]:
        return self.training_step_actor(experience, global_steps=global_steps, **kwargs)

    def _broadcast_to_vllm(self):
        use_prefix_cache = getattr(self.strategy.args, "enable_prefix_caching", False)
        cache_reset_refs = []
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            # clear prefix cache
            for engine in self.vllm_engines:
                cache_reset_refs.append(engine.reset_prefix_cache.remote())

        torch.cuda.empty_cache()
        model = self.actor.model.module
        count, num_params = 0, len(list(model.named_parameters()))
        for name, param in model.named_parameters():
            count += 1  # empty_cache at last param

            # broadcast
            if not self.use_cuda_ipc:
                use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
                # Fire all vllm engines for broadcast
                if torch.distributed.get_rank() == 0:
                    shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                    refs = [
                        engine.update_weight.remote(
                            name, dtype=param.dtype, shape=shape, empty_cache=count == num_params
                        )
                        for engine in self.vllm_engines
                    ]

                # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
                with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                    if torch.distributed.get_rank() == 0:
                        if use_ray:
                            import ray.util.collective as collective

                            collective.broadcast(param.data, 0, group_name=self._model_update_group)
                        else:
                            torch.distributed.broadcast(param.data, 0, group=self._model_update_group)
                        ray.get(refs)
            # CUDA IPC
            else:
                from torch.multiprocessing.reductions import reduce_tensor

                # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
                with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                    weight = param.data.clone()
                    ipc_handle = reduce_tensor(weight)

                    ipc_handle = {get_physical_gpu_id(): ipc_handle}
                    ipc_handle_list = [None] * torch.distributed.get_world_size()
                    torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

                    if torch.distributed.get_rank() == 0:
                        ipc_handles = {}
                        for d in ipc_handle_list:
                            ipc_handles.update(d)

                        shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                        refs = [
                            engine.update_weight_cuda_ipc.remote(
                                name,
                                dtype=param.dtype,
                                shape=shape,
                                ipc_handles=ipc_handles,
                                empty_cache=count == num_params,
                            )
                            for engine in self.vllm_engines
                        ]
                        ray.get(refs)
                    torch.distributed.barrier()
                    torch.cuda.synchronize()

        if cache_reset_refs:
            ray.get(cache_reset_refs)
        torch.cuda.empty_cache()
        torch.distributed.barrier()

    def _save_checkpoint(self, args, tag, client_states):
        save_path = None
        print('!!!! [saving] inside actor save_checkpoint')
        # call remote critic
        if not self.disable_ds_ckpt:
            if self.critic_train_remote:
                ref = self.critic.save_checkpoint.remote(tag)
            self.strategy.save_ckpt(
                self.actor.model,
                os.path.join(args.ckpt_path, "_actor"),
                tag,
                args.max_ckpt_num,
                args.max_ckpt_mem,
                client_states,
            )
        if self.save_hf_ckpt:
            save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
            self.strategy.save_model(
                self.ema_model if args.enable_ema else self.actor,
                self.processor or self.tokenizer,
                save_path,
            )
            max_num = args.max_ckpt_num
            if self.strategy.is_rank_0():
                while True:
                    save_dir = args.ckpt_path 
                    subdirs = sorted(
                        [
                            (os.path.join(save_dir, d), os.path.getmtime(os.path.join(save_dir, d)))
                            for d in os.listdir(save_dir)
                            if d.endswith('hf') and os.path.isdir(os.path.join(save_dir, d)) 
                        ],
                        key=lambda x: x[1],
                    ) # only take folders that ends with hf
                    
                    if len(subdirs) >= max_num: # or total_size > MAX_SIZE:
                        oldest_dir = subdirs[0][0]
                        if os.path.exists(oldest_dir):
                            shutil.rmtree(oldest_dir)
                            print(f"Deleted oldest ckpt {oldest_dir}")
                    else:
                        break
        # wait
        if not self.disable_ds_ckpt:
            if self.critic_train_remote:
                ray.get(ref)
                
        return save_path
        
