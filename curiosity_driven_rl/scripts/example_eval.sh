benchmark=vstar
export working_dir="/home/ma-user/work/haozhe/workspace/lmm-r1/pixelreasoner/curiosity_driven_rl"
export policy="/home/ma-user/work/haozhe/workspace/lmm-r1/toolbest/newbest"
export savefolder=tooleval
export nvj_path="/home/ma-user/anaconda3/envs/lmm/lib/python3.10/site-packages/nvidia/nvjitlink/lib" # in case the system cannot fiind the nvjit library
############
export sys=vcot # define the system prompt
export MIN_PIXELS=401408
export MAX_PIXELS=4014080 # define the image resolution
export eval_bsz=64 # vllm will processes this many queries 
export tagname=eval_vstar_bestmodel_hasvllmengine2
export testdata=/home/ma-user/work/haozhe/workspace/lmm-r1/data/${benchmark}.parquet
# export testdata="${working_dir}/data/${benchmark}.parquet"
export num_vllm=8
export num_gpus=8
bash ${working_dir}/scripts/eval_vlm_new.sh