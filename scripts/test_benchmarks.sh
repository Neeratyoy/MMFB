#! /bin/bash

declare -a tids=(10101 53 146818)

model=$1
fidelity=1

path="tmp_dump/test/"$model"/"$fidelity"/benchmark/"
path="nemo_dump/"$model"/"$fidelity"/benchmark/"
path="/work/ws/nemo/fr_nm217-hpobench-0/full/"$model/$fidelity"/benchmark/"

for tid in "${tids[@]}"
do
    echo "Task ID: "$tid
    python utils/test_benchmark_compress.py --path $path"task_"$tid"_new.pkl"
done