benchmark=${1}
if  [[ "$benchmark" == "m3u" ]]; then 
    export testdata="./data/MMMUPro_full.parquet"
elif  [[ "$benchmark" == "m3u_val" ]]; then 
    export testdata="./data/m3u_val.parquet"
elif [[ "$benchmark" == "emma" ]]; then 
    export factor=4
    export testdata="./data/emma_full.parquet"
elif [[ "$benchmark" == "mathverse" ]]; then 
    export testdata="./data/MathVerse_testmini.parquet"
elif [[ "$benchmark" == "mathvista" ]]; then 
    export testdata=./data/MathVista_testmini.parquet
elif [[ "$benchmark" == "mathvision" ]]; then 
    export testdata="./data/MathVision_test3040.parquet"
else 
    export testdata="./data/${benchmark}.parquet"
fi 

export num_vllm=8
export num_gpus=8
export tagname=eval_${2}_${benchmark}
export policy=${3} # /home/ma-user/work/data_mllm/pretrain_models/Qwen2.5-VL/Qwen2.5-VL-7B-Instruct

bash ./scripts/eval_vlm_new.sh

