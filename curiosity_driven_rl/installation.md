### Installation

```bash
cd VL-Rethinker
conda create -n rethinker python=3.10
pip install -e .[vllm]
pip install flash_attn --no-build-isolation
```

Note: vLLM >=0.7.2 is recommended.

Note: If you will use multi-node training, downgrade DeepSpeed to 0.15.0.
    reference: https://github.com/OpenRLHF/OpenRLHF/issues/776#issuecomment-2694472824

### Workarounds
At the time of this project, some bugs still linger around using flash-attn and vLLM for Qwen2.5-VL. The following are  solutions from the community:
1. to fix flash-attn issues
    ```
    export LD_LIBRARY_PATH=/path/to/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
    ```
    reference: https://github.com/pytorch/pytorch/issues/111469#issuecomment-1869208750


2. to fix qwen-vl preprocessor issues: modify preprocessor_config.json
    
    reference:
    - https://github.com/huggingface/transformers/issues/36193#issuecomment-2661278628
    - https://github.com/huggingface/transformers/issues/36246
