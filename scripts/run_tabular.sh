#! /bin/bash

model=$1
runtype=$2

# runs SVM if no arguments are specified
if [ -z "$model" ]
  then
    model="svm"
fi

# runs toy experiment if no arguments are specified
if [ -z "$runtype" ]
  then
    runtype="toy"
fi
config=$PWD"/arguments/"$runtype/$model"_args.yaml"

echo "Collecting "$runtype" benchmark for "$model" space"
echo "Loading args from "$config

# setting path variables to allow relative imports to work
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONPATH=$PWD"/../HPOBench/":$PYTHONPATH

time python3 $PWD/run_benchmark_dask.py --config $config