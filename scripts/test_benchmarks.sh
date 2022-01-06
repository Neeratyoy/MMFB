#! /bin/bash

declare -a tids=(10101 53 146818 146821 9952 146822 31 3917 168912 3 167119 12 146212 168911 9981 167120 14965 146606 7592 9977)

model=$1
fidelity=1
cores=$2

#path="tmp_dump/test/"$model"/"$fidelity"/benchmark/"
#path="nemo_dump/TabularData/"$model"/"$fidelity"/benchmark/"
path="/work/ws/nemo/fr_nm1068-hpobench-0/full/"$model/$fidelity"/benchmark/"

for tid in "${tids[@]}"
do
    echo "Task ID: "$tid
    python utils/test_benchmark_compress.py --path $path"task_"$tid"_new.pkl" --n_jobs $cores
    echo -e "\n"
done