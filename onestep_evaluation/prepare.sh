set -x
dataname=${1} # hfname VStar-EvalData-PixelReasoner
hfuser=${hfuser:-"TIGER-Lab"}
working_dir=/home/ma-user/work/haozhe/workspace/lmm-r1/pixelreasoner/curiosity_driven_rl
if [[ ! -f "hfd.sh" ]]; then
    echo "downloading hfd.sh"
    
    wget https://hf-mirror.com/hfd/hfd.sh
    chmod a+x hfd.sh
else
    echo "hfd.sh already exists."
fi

bash hfd.sh ${hfuser}/${dataname} --dataset --tool wget
cd ${dataname}
unzip images.zip
rm images.zip

# get the benchmark name
if [[ $(ls *.parquet 2>/dev/null | wc -l) -gt 0 ]]; then
    parquet_file=$(ls *.parquet | head -1)
    benchmarkname="${parquet_file%.parquet}"
    echo "benchmark name: $benchmarkname"
else
    echo "error no *.parquet under this folder" >&2
    exit 1
fi

# move the data to the data_folder
data_folder=${working_dir}/data
mkdir -p ${data_folder}
mv images ${data_folder}/${benchmarkname}_images
mv ${benchmarkname}.parquet ${data_folder}/

# clear the download cache
cd ..
rm -r ${dataname}

# rename the image path
python rename_imagepath.py ${working_dir}

 
