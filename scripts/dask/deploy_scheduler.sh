#! /bin/bash

scheduler=$1

codedir=$HOME'/Thesis/code/MMFB'

# important for Dask to not fail on large cluster setups
source $codedir/scripts/nemo/config.sh

export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONPATH=$PWD"/../HPOBench/":$PYTHONPATH

dask-scheduler --scheduler-file $scheduler --port 0