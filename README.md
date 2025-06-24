# Pixel Reasoner: Incentivizing Pixel-Space Reasoning with Curiosity-Driven Reinforcement Learning

<a target="_blank" href="https://arxiv.org/abs/2505.15966">
<img style="height:22pt" src="https://img.shields.io/badge/-Paper-red?style=flat&logo=arxiv"></a>
<a target="_blank" href="#">
<img style="height:22pt" src="https://img.shields.io/badge/-Code-green?style=flat&logo=github"></a>

<a target="_blank" href="https://tiger-ai-lab.github.io/Pixel-Reasoner/">
<img style="height:22pt" src="https://img.shields.io/badge/-🌐%20Website-blue?style=flat"></a>

<a target="_blank" href="https://huggingface.co/TIGER-Lab/PixelReasoner-RL-v1">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Models-red?style=flat"></a>

<a target="_blank" href="https://huggingface.co/collections/TIGER-Lab/pixel-reasoner-682fe96ea946d10dda60d24e">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Dataset-blue?style=flat"></a>

<a target="_blank" href="https://huggingface.co/spaces/TIGER-Lab/Pixel-Reasoner">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Demo-yellow?style=flat"></a>

<br>
<span>
<b>Authors:</b> Alex Su<sup>*</sup>
<a class="name" target="_blank" href="https://HaozheH3.github.io">Haozhe Wang<sup>*</sup><sup>&dagger;</sup></a>, 
<a class="name" target="_blank" href="https://cs.uwaterloo.ca/~w2ren/">Weiming Ren</a>, 
<a class="name" target="_blank" href="https://cse.hkust.edu.hk/~flin/">Fangzhen Lin</a>,
<a class="name" target="_blank" href="https://wenhuchen.github.io/">Wenhu Chen<sup>&Dagger;</sup></a>
<br>
<sup>*</sup>Equal Contribution. 
<sup>&dagger;</sup>Project Lead. 
<sup>&Dagger;</sup>Correspondence.
</span>



## 🔥News
- [2025/5/25] We made really fun demos! You can now play with the [**online demo**](https://huggingface.co/spaces/TIGER-Lab/Pixel-Reasoner). Have fun!
- [2025/5/22] We released models-v1. Now actively working on data and code release.

## Features
- Supports Instruction Tuning with multi-turn trajectories
- Supports RL with multi-turn trajectories
- Supports RL with mixed video and image data
- Supports Inference/Evaluation with vLLM 
  
## Overview
![overview](./assets/teaser.png)

<details><summary>Abstract</summary> 
Chain-of-thought reasoning has significantly improved the performance of Large Language Models (LLMs) across various domains. However, this reasoning process has been confined exclusively to textual space, limiting its effectiveness in visually intensive tasks. To address this limitation, we introduce the concept of reasoning in the pixel-space. Within this novel framework, Vision-Language Models (VLMs) are equipped with a suite of visual reasoning operations, such as zoom-in and select-frame. These operations enable VLMs to directly inspect, interrogate, and infer from visual evidences, thereby enhancing reasoning fidelity for visual tasks.
Cultivating such pixel-space reasoning capabilities in VLMs presents notable challenges, including the model's initially imbalanced competence and its reluctance to adopt the newly introduced pixel-space operations. We address these challenges through a two-phase  training approach. The first phase employs instruction tuning on synthesized reasoning traces to familiarize the model with the novel visual operations. Following this, a reinforcement learning (RL) phase leverages a curiosity-driven reward scheme to balance exploration between pixel-space reasoning and textual reasoning. With these visual operations, VLMs can interact with complex visual inputs, such as information-rich images or videos to proactively gather necessary information. We demonstrate that this approach significantly improves VLM performance across diverse visual reasoning benchmarks. Our 7B model, \model, achieves 84\% on V* bench, 74\% on TallyQA-Complex, and 84\% on InfographicsVQA, marking the highest accuracy achieved by any open-source model to date. These results highlight the importance of pixel-space reasoning and the effectiveness of our framework.
</details>

### Models
Please check the [TIGER-Lab/PixelReasoner-RL-v1](https://huggingface.co/TIGER-Lab/PixelReasoner-RL-v1) and [TIGER-Lab/PixelReasoner-WarmStart](https://huggingface.co/TIGER-Lab/PixelReasoner-WarmStart)

## 🚀Quick Start
We proposed two-staged post-training. The instruction tuning is adapted from Open-R1. The Curiosity-Driven RL is adapted from VL-Rethinker.

### DATA

- Instruction Tuning: [TIGER-Lab/PixelReasoner-SFT-Data](https://huggingface.co/datasets/TIGER-Lab/PixelReasoner-SFT-Data)

- RL Queries: [TIGER-Lab/PixelReasoner-RL-Data](https://huggingface.co/datasets/TIGER-Lab/PixelReasoner-RL-Data)

- Evaluation: [HF Collections](https://huggingface.co/collections/JasperHaozhe/evaldata-pixelreasoner-6846868533a23e71a3055fe9)

### Running Instruction Tuning

Follow these steps to start the instruction tuning process:

1. **Installation**
   - Navigate to the `instruction_tuning` folder
   - Follow the detailed setup guide in [installation instructions](instruction_tuning/install/install.md)

2. **Configuration**
   -configure model and data path in sft.sh

3. **Launch Training**
   ```bash
   bash sft.sh
   ```

### Running Curiosity-Driven RL
First prepare data. Run the following will get the training data prepared under `curiosity_driven_rl/data` folder. 
```
dataname=PixelReasoner-RL-Data
export hf_user=TIGER-Lab
cd onestep_evaluation
bash prepare.sh ${dataname}
```

Then download model [TIGER-Lab/PixelReasoner-RL-v1](https://huggingface.co/TIGER-Lab/PixelReasoner-RL-v1).

Under `curiosity_driven_rl` folder, install the environment following [the installation instructions](curiosity_driven_rl/installation.md).

Set the data path, model path, wandb keys (if you want to use it) in `curiosity_driven_rl/scripts/train_vlm_multi.sh`.

Run the following for multinode training (e.g., 4x8 actor + 4x8 vllm).
```bash
cd curiosity_driven_rl

export temperature=1.0
export trainver="dataname"
export testver="testdataname"
export filter=True # filtering zero advantages
export algo=group # default for grpo
export lr=10
export MAX_PIXELS=4014080 # =[max_image_token]x28x28
export sys=vcot # system prompt version
export mode=train # [no_eval, eval_only, train]
export policy=/path/to/policy
export rbuffer=512 # replay buffer size
export bsz=256 # global train batch size
export evalsteps=4 
export nactor=4 # 4x8 for actor if with multinode
export nvllm=32 # 4x8 for sampling if with multinode
export tp=1 # vllm tp, 1 for 7B
export repeat=1 # data repeat
export nepoch=3 # data epoch
export logp_bsz=1 # must be 1
export maxlen=10000 # generate_max_len
export tagname=Train

bash ./scripts/train_vlm_multi.sh
```

Run the following for single-node training.
```bash
cd curiosity_driven_rl

export temperature=1.0
export trainver="dataname"
export testver="testdataname"
export filter=True # filtering zero advantages
export algo=group # default for grpo
export lr=10
export MAX_PIXELS=4014080 # =[max_image_token]x28x28
export sys=vcot # system prompt version
export mode=train # [no_eval, eval_only, train]
export policy=/path/to/policy
export rbuffer=512 # replay buffer size
export bsz=256 # global train batch size
export evalsteps=1
export mbsz=2 
export tp=1 # vllm tp, 1 for 7B
export repeat=1 # data repeat
export nepoch=3 # data epoch
export logp_bsz=1 # must be 1
export maxlen=10000 # generate_max_len
export tagname=Train

bash ./scripts/train_vlm_single.sh
```

**Note**: the number of prompts into vLLM inference is controlled by `--eval_batch_size_pergpu` during evaluation, and `args.rollout_batch_size // strategy.world_size` during training. Must set `logp_bsz=1` or `--micro_rollout_batch_size=1` for computing logprobs because model.generate() suffers from feature mismatch when batchsize > 1.

### Evaluation
Evaluation data can be found in [the HF Collection](https://huggingface.co/collections/JasperHaozhe/evaldata-pixelreasoner-6846868533a23e71a3055fe9).

#### Image-Based Benchmarks
Let's take the vstar evaluation as an example. The HF data path is `JasperHaozhe/VStar-EvalData-PixelReasoner`.

**1. Prepare Data**
```
dataname=VStar-EvalData-PixelReasoner
cd onestep_evaluation
bash prepare.sh ${dataname}
```
The bash script will download from HF, process the image paths, and move the data to `curiosity_driven_rl/data`. 

Check the folder `curiosity_driven_rl/data`, you will know the downloaded parquet file is named as `vstar.parquet`. 

**2. Inference and Evaluation**

Install the openrlhf according to `curiosity_driven_rl/installation.md`.

Under `curiosity_driven_rl` folder. Set `benchmark=vstar`, `working_dir`, `policypath`, and `savefolder`,`tagname` for saving evaluation results. Run the following.
```
benchmark=vstar
export working_dir="/path/to/curiosity_driven_rl"
export policy="/path/to/policy"
export savefolder=tooleval
export nvj_path="/path/to/nvidia/nvjitlink/lib" # in case the system cannot fiind the nvjit library
############
export sys=vcot # define the system prompt
export MIN_PIXELS=401408
export MAX_PIXELS=4014080 # define the image resolution
export eval_bsz=64 # vllm will processes this many queries 
export tagname=eval_vstar_bestmodel
export testdata="${working_dir}/data/${benchmark}.parquet"
export num_vllm=8
export num_gpus=8
bash ${working_dir}/scripts/eval_vlm_new.sh
```
#### Video-Based Benchmarks
For the MVBench, we extracted the frames from videos and construct the eval data that fits into our evaluation. The data is available in `JasperHaozhe/MVBench-EvalData-PixelReasoner`

**1. Prepare Data**
```
dataname=MVBench-EvalData-PixelReasoner
cd onestep_evaluation
bash prepare.sh ${dataname}
```
**2. Inference and Evaluation**

Under `curiosity_driven_rl` folder. Set `benchmark=mvbench`, `working_dir`, `policypath`, and `savefolder`,`tagname` for saving evaluation results. Run the following.
```
benchmark=mvbench
export working_dir="/path/to/curiosity_driven_rl"
export policy="/path/to/policy"
export savefolder=tooleval
export nvj_path="/path/to/nvidia/nvjitlink/lib" # in case the system cannot fiind the nvjit library
############
export sys=vcot # define the system prompt
export MIN_PIXELS=401408
export MAX_PIXELS=4014080 # define the image resolution
export eval_bsz=64 # vllm will processes this many queries 
export tagname=eval_vstar_bestmodel
export testdata="${working_dir}/data/${benchmark}.parquet"
export num_vllm=8
export num_gpus=8
bash ${working_dir}/scripts/eval_vlm_new.sh
```

#### Inference of Qwen2.5-VL-Instruct
Set `sys=notool` for textual reasoning with Qwen2.5-VL-Instruct . `sys=vcot` can trigger zero-shot use of visual operations but may also induce unexpected behaviors.

## Possible Exceptions
**1. Exceeding Model Context Length**

If you do sampling (e.g., during training) and set a larger `MAX_PIXELS`, you could encounter the following:
```
ValueError: The prompt (total length 10819) is too long to fit into the model (context length 10240). Make sure that `max_model_len` is no smaller than the number of text tokens plus multimodal tokens. For image inputs, the number of image tokens depends on the number of images, and possibly their aspect ratios as well.
```

This stems from:
1. max model length is too short, try adjust `generate_max_len + prompt_max_len`
2. too many image tokens, which means there could be too many images or the image resolution is too large and takes up many image tokens. 
To address this problem during training, you could set smaller `MAX_PIXELS`, or you could set the max number images during training, via `max_imgnum` in `curiosity_driven_rl/openrlhf/trainer/ppo_utils/experience_maker.py`.

**2. transformers and vLLM version mismatch**
```
Input and cos/sin must have the same dtype, got torch.float32 and torch.bfloat32.
```
Try reinstall transformers: `pip install --force-reinstall git+https://github.com/huggingface/transformers.git@9985d06add07a4cc691dc54a7e34f54205c04d4` and update vLLM. 

**3. logp_bsz=1**
```
Exception: dump-info key solutions: {} should be {}
```
Must set `logp_bsz=1` or `--micro_rollout_batch_size=1` for computing logprobs because model.generate() suffers from feature mismatch when batchsize > 1.

**4. Reproduction Mismath**

Correct values of `MAX_PIXELS` and `MIN_PIXELS` are crucial for reproducing the results. 
1. make sure the env variables are correctly set
2. make sure the vLLM engines correctly read the env variables

When using ray parallelism, chances are that the env variables are only set on one machine but not all machines. To fix this, add ray env variables as follows:
```
RUNTIME_ENV_JSON="{\"env_vars\": {\"MAX_PIXELS\": \"$MAX_PIXELS\", \"MIN_PIXELS\": \"$MIN_PIXELS\"}}"

ray job submit --address="http://127.0.0.1:8265" \
--runtime-env-json="$RUNTIME_ENV_JSON" \
```

Thanks [@LiqiangJing](https://github.com/LiqiangJing) for feedback!

## Contact
Contact Haozhe (jasper.whz@outlook.com) for direct solution of any bugs in RL.

Contact Muze for SFT.



## Citation
If you find this work useful, please give us a free cite:
```bibtex
@article{pixelreasoner,
      title={Pixel Reasoner: Incentivizing Pixel-Space Reasoning with Curiosity-Driven Reinforcement Learning},
      author = {Su, Alex and Wang, Haozhe and Ren, Weiming and Lin, Fangzhen and Chen, Wenhu},
      journal={arXiv preprint arXiv:2505.15966},
      year={2025}
}
```
