set -x


find_interface() {
  local ip_output=$(ip addr show | head -n 10) # Limit to first 10 lines
  local selected_interface=""

  # Debug output (can be removed in final version)
  # echo "--- First 10 lines of ip addr show output: ---"
  # echo "$ip_output"
  # echo "--- End of ip addr show output ---"

  while IFS= read -r line; do
    # Debug output (can be removed in final version)
    # echo "Processing line: $line"

    if [[ "$line" =~ ^[0-9]+:\ ([^:]+):\ \<.*UP.*\> ]]; then
      local interface_name="${BASH_REMATCH[1]}"
      # Debug output (can be removed in final version)
      # echo "  Interface found: $interface_name"
      local interface_up=true
      local is_loopback=false

      if [[ "$interface_name" == "lo" ]]; then
        is_loopback=true
        # Debug output (can be removed in final version)
        # echo "  Interface '$interface_name' is loopback. Skipping."
      fi

      if $is_loopback; then
        continue # Skip loopback interface
      fi

      # Look for inet lines within this interface block
      while IFS= read -r subnet_line; do
        # Debug output (can be removed in final version)
        # echo "  Processing subnet line: $subnet_line"
        if [[ "$subnet_line" =~ inet\ ([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)/([0-9]+)\ .*scope\ ([^ ]+) ]]; then
          local ip_address="${BASH_REMATCH[1]}"
          local scope="${BASH_REMATCH[3]}"
          # Debug output (can be removed in final version)
          # echo "    Found inet line: IP Address: $ip_address, Scope: $scope"

          # Exclude loopback IPs and docker0/bridge related IPs by IP range
          if [[ "$ip_address" =~ ^127\. ]]; then
            # Debug output (can be removed in final version)
            # echo "      IP '$ip_address' is loopback. Skipping."
            continue # Skip 127.0.0.0/8 loopback IPs (although 'lo' should already be skipped)
          elif [[ "$ip_address" =~ ^169\.254\. ]]; then
            # Debug output (can be removed in final version)
            # echo "      IP '$ip_address' is link-local (169.254.x.x). Skipping."
            continue # Skip 169.254.0.0/16 link-local IPs (like docker0 often has)
          fi

          local is_private_ip=false
          if [[ "$ip_address" =~ ^10\.([0-9]{1,3}\.){2}[0-9]{1,3}$ ]] ||
             [[ "$ip_address" =~ ^172\.(1[6-9]|2[0-9]|3[0-1])\.([0-9]{1,3}\.){1}[0-9]{1,3}$ ]] ||
             [[ "$ip_address" =~ ^192\.168\.([0-9]{1,3}\.){1}[0-9]{1,3}$ ]]; then
            is_private_ip=true
            # Debug output (can be removed in final version)
            # echo "      IP '$ip_address' is a private IP."
          # else
            # Debug output (can be removed in final version)
            # echo "      IP '$ip_address' is NOT a private IP."
          fi

          if $is_private_ip || [[ "$scope" == "global" ]]; then # Consider private or global scope interfaces
            selected_interface="$interface_name"
            # Debug output (can be removed in final version)
            # echo "      Interface '$interface_name' with IP '$ip_address' and scope '$scope' is selected."
            # echo "export GLOO_SOCKET_IFNAME=$selected_interface"
            # exit 0 # Exit immediately after finding the first suitable interface for debugging (removed for function)
            break 2 # Found a suitable interface! Break out of both inner and outer loops
          # else
            # Debug output (can be removed in final version)
            # echo "      Interface '$interface_name' with IP '$ip_address' and scope '$scope' is NOT suitable (not private or global)."
          fi
        fi
      done < <(echo "$ip_output" | sed -n "/$interface_name: /,/^[0-9]\+:/p" | sed '$d' ) # Extract lines belonging to current interface block
      if [[ -n "$selected_interface" ]]; then # Check if selected_interface is not empty, if so, interface found and loops broken.
          # Debug output (can be removed in final version)
          # echo "      Selected interface '$selected_interface' already found. Breaking outer loop."
          break # Already found and assigned an interface, break outer loop as well.
      fi
    # else
      # Debug output (can be removed in final version)
      # echo "  Line does not match interface pattern."
    fi
  done < <(echo "$ip_output")

  if [[ -n "$selected_interface" ]]; then
    echo "$selected_interface"
  else
    echo "" # Return empty string if no interface is found, so export GLOO_SOCKET_IFNAME=  (empty)
    # echo "No suitable network interface could be automatically identified for GLOO_SOCKET_IFNAME." # No longer print error message to stderr in function context
    # return 1 # Optionally, you could return a non-zero exit code if you need to check for failure.
  fi
}

MULTINODE_FLAG=True
if [ -v MULTINODE_FLAG ]; then 
    # Define a string
    
    # Set the IFS (Internal Field Separator) to space
    IFS=','

    WORLD_SIZE=${MA_NUM_HOSTS:-"1"}
    export RAY_MASTER_NODE_ADDRESS=${myvar[(($WORLD_SIZE-1))]}
    export RAY_MASTER_NODE_PORT=$(shuf -n 1 -i 30000-40000)

    NODE_RANK=""
    GPUS_PER_NODE=""

else 
    RAY_MASTER_NODE_ADDRESS="0.0.0.0"
    RAY_MASTER_NODE_PORT=$(shuf -n 1 -i 30000-65535)
    WORLD_SIZE=1
    NODE_RANK=0
    GPUS_PER_NODE=8
fi
MASTER_HOST="$VC_WORKER_HOSTS"
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
# export NCCL_SOCKET_IFNAME=ens2f5
# export GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}
export NCCL_NET_PLUGIN=none
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=15
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1

export HOST_IP=0.0.0.0
export VLLM_HOST_IP=0.0.0.0

working_dir=/path/to/workdir
cd $working_dir
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_API_KEY=""
nnode=$WORLD_SIZE
tagname=${tagname:-""}
dataver=${dataver:-"none"}
tag=qw-vl7b-${trainver}-${tagname}
rule_reward=${rule:-"none"}
sys=${sys:-"default"}
lr=${lr:-"10"}
algo=${algo:-"group_sft"}
temperature=${temperature:-"1.0"}
numref=0
fmt=${fmt:-"none"}
bsz=${bsz:-"512"}
rbuffer=${bsz:-"1024"}
nsamples=${nsamples:-"8"}
mbsz=${mbsz:-"4"}
maxlen=${maxlen:-"6144"}
lossver=${lossver:-"none"}
mode=${mode:-"none"}
nactor=${nactor:-"16"}
nvllm=${nvllm:-"8"}
filter=${filter:-"None"}
repeat=${repeat:-"0"}
nepoch=${nepoch:-"3"}
logp_bsz=${logp_bsz:-"8"}
maxtoken=${maxtoken:-"2048"}
tp=${tp:-"1"}
aux=${aux:-"0.05"}
evalsteps=${evalsteps:-"0"}
save_name="${tag}-${bsz}-lossver${lossver}-samplever${dataver}-fmt${fmt}-${algo}-n${nsamples}-ml${maxlen}-lr${lr}-sys${sys}-${nnode}node" # rbsize 1024->256

DATASET=/path/to/train.parquet
MODEL_CPK_NAME=${save_name}
PRETRAIN_MODEL=${policy}
testdata="/path/to/test.parquet"
SAVE_PATH=$working_dir/saves/$save_name
mkdir -p "${SAVE_PATH}"
# pip install -U deepspeed==0.15.0 # https://github.com/OpenRLHF/OpenRLHF/issues/776#issuecomment-2694472824
# 


post_args=""
if [ $nnode -gt 1 ]; then
    
        post_args=(--ref_num_nodes 0
            --ref_num_gpus_per_node 8 
            --actor_num_nodes ${nactor}
            --actor_num_gpus_per_node 8 
            --vllm_num_engines ${nvllm} 
            --vllm_tensor_parallel_size ${tp}
            --micro_train_batch_size ${mbsz} 
            --train_batch_size ${bsz} 
            --micro_rollout_batch_size ${logp_bsz}
            --rollout_batch_size ${rbuffer}
        )
    
else 
    post_args=(--ref_num_nodes 0
            --ref_num_gpus_per_node 8 
            --actor_num_nodes 4
            --actor_num_gpus_per_node 1 
            --vllm_num_engines 4 
            --vllm_tensor_parallel_size 1
            --adam_offload
            --micro_train_batch_size 4 
            --train_batch_size ${bsz}
            --micro_rollout_batch_size 4
            --rollout_batch_size ${rbuffer}
    )
fi
# :/usr/local/cuda/targets/x86_64-linux/lib
LD_LIBRARY_PATH_VALUE=/path/to/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export BNB_CUDA_VERSION=122
RUNTIME_ENV_JSON="{\"env_vars\": {\"LD_LIBRARY_PATH\": \"$LD_LIBRARY_PATH_VALUE\"}}"


if [ "$NODE_RANK" = "0" ]; then
    # Start Ray head node and capture the output
    ray_output=$(ray start --head --num-gpus 8)

    # Extract the IP address using grep and sed
    ip_address=$(echo "$ray_output" | grep -oP "ray start --address='\K[^']+")

    # Write the extracted IP address to a file named "ip.txt"
    mkdir -p ip_tmp
    echo "$ip_address" > ip_tmp/ip_${tagname}.txt
    cat ip_tmp/ip_${tagname}.txt

    

    if [ $nnode -gt 1 ]; then
      # Example usage (to set the environment variable):
      export GLOO_SOCKET_IFNAME=$(find_interface)
      echo "$GLOO_SOCKET_IFNAME" > ip_tmp/gloo_${tagname}.txt
      sleep 60
    else 
      unset GLOO_SOCKET_IFNAME
      unset NCLL_SOCKET_IFNAME
    fi
    ray status
    ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="$RUNTIME_ENV_JSON" \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --vllm_enable_sleep \
    --vllm_gpu_memory_utilization 0.85 \
    --vllm_sync_backend gloo \
    --pretrain $PRETRAIN_MODEL \
    --save_path $SAVE_PATH \
    --n_samples_per_prompt ${nsamples} \
    --max_epochs 1 \
    --num_episodes ${nepoch} \
    --filter ${filter} \
    --prompt_max_len 2048 \
    --max_out_tokens ${maxtoken} \
    --max_samples 100000 \
    --generate_max_len ${maxlen} \
    --advantage_estimator ${algo} \
    --zero_stage 3 \
    --controlled_shuffle ${repeat} \
    --bf16 \
    --actor_learning_rate ${lr}e-7 \
    --rule_reward ${rule_reward} \
    --temperature 1.0 \
    --val_temperature 0.6 \
    --top_p 0.95 \
    --training_mode ${mode} \
    --init_kl_coef 0.0 \
    --aux_loss_coef ${aux} \
    --entropy_loss_coef 0.0 \
    --prompt_data $DATASET \
    --input_key question \
    --apply_chat_template \
    --normalize_reward \
    --flash_attn \
    --gradient_checkpointing \
    --ckpt_path $SAVE_PATH \
    --save_steps 3 \
    --eval_steps ${evalsteps} \
    --max_ckpt_num 3 \
    --save_hf_ckpt \
    --disable_ds_ckpt \
    --disable_fast_tokenizer \
    --use_wandb $WANDB_API_KEY \
    --wandb_run_name $save_name \
    --system_prompt ${sys} \
    --use_kl_estimator_k3 \
    --wandb_project vlm-rl \
    --buffer_norm 0 \
    --train_vlm \
    --filter ${filter} \
    --eval_data ${testdata} \
    --data_version ${dataver} \
    --loss_version ${lossver} \
    --format ${fmt} \
    ${post_args[@]} 
   # --train_vlm 
else 
    sleep 15 
    # Read the IP address from the file and assign it to the variable "head_ip"
    head_ip=$(cat ip_tmp/ip_${tagname}.txt)
    gloo=$(cat ip_tmp/gloo_${tagname}.txt)
    export GLOO_SOCKET_IFNAME=$gloo
    echo "gloo: $GLOO_SOCKET_IFNAME"
    # Print the value of head_ip for verification
    echo "Head IP Address: $head_ip"

    ray start --address ${head_ip}
    # echo $HOST_IP
fi