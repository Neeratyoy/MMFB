#! /bin/bash

#MSUB -l walltime=24:00:00
#MSUB -N worker
#MSUB -o /work/ws/nemo/fr_nm217-hpobench-0/msub-logs
#MSUB -e /work/ws/nemo/fr_nm217-hpobench-0/msub-logs

codedir=$HOME'/Thesis/code/MMFB'
wspace='/work/ws/nemo/fr_nm217-hpobench-0/'

export PYTHONPATH=$codedir:$PYTHONPATH
export PYTHONPATH=$codedir"/../HPOBench/":$PYTHONPATH

dask-worker --scheduler-file $codedir/tmp_dump/scheduler.json \
  --no-nanny --nprocs 1 --nthreads 1 --name 'worker'$1 \
  --local-directory $wspace
