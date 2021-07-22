#! /bin/bash

codedir=$PWD
nworkers=$1
scheduler=$2
localdir=$3
id=$4

export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONPATH=$PWD"/../HPOBench/":$PYTHONPATH

# important for Dask to not fail on large cluster setups
export DASK_DISTRIBUTED__SCHEDULER__ALLOWED_FAILURES=10
export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=60
export DASK_DISTRIBUTED__COMM__RETRY__COUNT=5

for ((i=0; i<$nworkers; i++)); do
    nohup `dask-worker --scheduler-file $scheduler \
        --no-nanny --nprocs 1 --nthreads 1 --name 'worker'$id'_'$i \
        --local-directory $localdir` --reconnect &
    echo 'Created worker'$i;
    sleep 3;
done

sleep 4d