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


## Overview
![overview](./assets/teaser.png)

<details><summary>Abstract</summary> 
Chain-of-thought reasoning has significantly improved the performance of Large Language Models (LLMs) across various domains. However, this reasoning process has been confined exclusively to textual space, limiting its effectiveness in visually intensive tasks. To address this limitation, we introduce the concept of reasoning in the pixel-space. Within this novel framework, Vision-Language Models (VLMs) are equipped with a suite of visual reasoning operations, such as zoom-in and select-frame. These operations enable VLMs to directly inspect, interrogate, and infer from visual evidences, thereby enhancing reasoning fidelity for visual tasks.
Cultivating such pixel-space reasoning capabilities in VLMs presents notable challenges, including the model's initially imbalanced competence and its reluctance to adopt the newly introduced pixel-space operations. We address these challenges through a two-phase  training approach. The first phase employs instruction tuning on synthesized reasoning traces to familiarize the model with the novel visual operations. Following this, a reinforcement learning (RL) phase leverages a curiosity-driven reward scheme to balance exploration between pixel-space reasoning and textual reasoning. With these visual operations, VLMs can interact with complex visual inputs, such as information-rich images or videos to proactively gather necessary information. We demonstrate that this approach significantly improves VLM performance across diverse visual reasoning benchmarks. Our 7B model, \model, achieves 84\% on V* bench, 74\% on TallyQA-Complex, and 84\% on InfographicsVQA, marking the highest accuracy achieved by any open-source model to date. These results highlight the importance of pixel-space reasoning and the effectiveness of our framework.
</details>

## Release Progress
- [x] models.
- [x] data.
- [x] inference and evaluation code.
- [x] instruction-tuning code.
- [x] RL code. 

### Models
Please check the [TIGER-Lab/PixelReasoner-RL-v1](https://huggingface.co/TIGER-Lab/PixelReasoner-RL-v1) and [TIGER-Lab/PixelReasoner-WarmStart](https://huggingface.co/TIGER-Lab/PixelReasoner-WarmStart)

## 🚀Quick Start
We proposed two-staged post-training. The instruction tuning is adapted from Open-R1. The Curiosity-Driven RL is adapted from VL-Rethinker.

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
Under `curiosity_driven_rl` folder, check out [the installation instructions](curiosity_driven_rl/installation.md).

Run the following.
```bash
cd curiosity_driven_rl
bash ./scripts/train_vlm_multi.sh
```

### One-Step Evaluation
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
