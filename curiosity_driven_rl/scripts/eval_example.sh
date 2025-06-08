export sys=vcot
export MIN_PIXELS=401408
export MAX_PIXELS=4014080
export savefolder=tooleval
export eval_bsz=64
export nvj_path="/path/to/nvidia/nvjitlink/lib"
export working_dir="/path/to/curiosity_driven_rl"
benchmark=vstar
policypath="/path/to/policy"
tagname=reproduce_best
bash /home/ma-user/work/haozhe/workspace/lmm-r1/pixelreasoner/curiosity_driven_rl/scripts/eval_7b.sh ${benchmark} ${tagname} ${policypath}