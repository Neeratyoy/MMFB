#! /bin/bash

codedir=$PWD
nworkers=$1
scheduler=$2
localdir=$3
id=$4

codedir=$HOME'/Thesis/code/MMFB'

export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONPATH=$PWD"/../HPOBench/":$PYTHONPATH

# important for Dask to not fail on large cluster setups
source $codedir/scripts/nemo/config.sh

for ((i=0; i<$nworkers; i++)); do
    nohup `dask-worker --scheduler-file $scheduler \
        --no-nanny --nprocs 1 --nthreads 1 --name 'worker'$id'_'$i \
        --local-directory $localdir` --reconnect &
    echo 'Created worker'$i;
    sleep 3;
done

# not sure if background jobs in a remote cluster can be killed if parent is killed
# sleep allows this parent job to be alive
sleep 4d