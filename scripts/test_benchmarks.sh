#! /bin/bash

declare -a tids=(10101 53 146818) # 146821 9952 146822 31 3917 168912 3)

model="svm"
fidelity=1
path="tmp_dump/test/"$model"/"$fidelity"/benchmark/"

for tid in "${tids[@]}"
do
    echo "Task ID: "$tid
    python utils/test_benchmark.py --path $path"task_"$tid"_new.pkl"
done