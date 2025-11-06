


### Dependencies Installation
```bash
# Install transformers from source


# Run setup script
cd instruction_tuning/install
pip install -r requirements.txt
pip install git+ssh://git@github.com/huggingface/transformers.git@89d27fa6fff206c0153e9670ae09e2766eb75cdf

pip install wandb==0.18.3
pip install tensorboardx
pip install torch==2.6.0 torchvision==0.21.0
pip install flash-attn==2.7.4.post1

# Install Qwen related packages
pip install git+ssh://git@github.com/cjakfskvnad/Qwen-Agent.git
pip install qwen-vl-utils

```