#! /bin/bash

filename="tmp_dump/scheduler.json"
envname="mmfb"

dirpath=$HOME'/Thesis/code/MMFB/'

echo "filename: "$filename
echo "envname: "$envname

# setting up environment
source $HOME/miniconda3/bin/activate $envname

# creating a Dask worker
PYTHONPATH=$PWD dask-worker --scheduler-file $dirpath$filename
