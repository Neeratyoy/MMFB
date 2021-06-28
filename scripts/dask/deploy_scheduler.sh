#! /bin/bash

scheduler=$1

export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONPATH=$PWD"/../HPOBench/":$PYTHONPATH

dask-scheduler --scheduler-file $scheduler