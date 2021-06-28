#! /bin/bash

model=$1
runtype=$2

codedir=$HOME'/Thesis/code/MMFB'
wspace='/work/ws/nemo/fr_nm217-hpobench-0/'

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
config=$codedir"/arguments/"$runtype/$model"_args.yaml"

echo "Collecting "$runtype" benchmark for "$model" space"
echo "Loading args from "$config

source $HOME/miniconda3/bin/activate mmfb

# setting path variables to allow relative imports to work
export PYTHONPATH=$codedir:$PYTHONPATH
export PYTHONPATH=$codedir"/../HPOBench/":$PYTHONPATH

time python3 $codedir/run_benchmark_dask.py --config $config