#! /bin/bash

model=$1
runtype=$2
taskid=$3

codedir=$HOME'/Thesis/code/MMFB'
wspace='/work/ws/nemo/fr_nm217-hpobench-0/'

config=$codedir"/arguments/nemo/"$runtype/$model"/args_"$taskid".yaml"

echo "Collecting "$runtype" benchmark for "$model" space"
echo "Loading args from "$config

source $HOME/miniconda3/bin/activate mmfb

# important for Dask to not fail on large cluster setups
source $codedir/scripts/nemo/config.sh

# setting path variables to allow relative imports to work
export PYTHONPATH=$codedir:$PYTHONPATH
export PYTHONPATH=$codedir"/../HPOBench/":$PYTHONPATH

time python3 $codedir/run_benchmark_dask.py --config $config