#! /bin/bash

filename="tmp_dump/scheduler.json"
envname="mmfb"

dirpath=$HOME'/Thesis/code/MMFB/'

echo "filename: "$filename
echo "envname: "$envname

# setting up environment
source $HOME/miniconda3/bin/activate $envname

# Creating a Dask scheduler
PYTHONPATH=$PWD dask-scheduler --scheduler-file $dirpath$filename


