#! /bin/bash

codedir=$PWD
nworkers=$1
scheduler=$2
localdir=$3

export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONPATH=$PWD"/../HPOBench/":$PYTHONPATH

for ((i=0; i<$nworkers; i++)); do
    nohup `dask-worker --scheduler-file $scheduler \
        --no-nanny --nprocs 1 --nthreads 1 --name 'worker'$i \
        --local-directory $localdir` &
    echo 'Created worker'$i;
    sleep 3;
done

sleep 4d