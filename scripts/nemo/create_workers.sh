#! /bin/bash

#MSUB -l nodes=100:ppn=1
#MSUB -l walltime=96:00:00
#MSUB -N mmfb-workers
#MSUB -o /work/ws/nemo/fr_nm217-hpobench-0/msub-logs
#MSUB -e /work/ws/nemo/fr_nm217-hpobench-0/msub-logs

nworkers=100
codedir=$HOME'/Thesis/code/MMFB'
wspace='/work/ws/nemo/fr_nm217-hpobench-0/'

source $HOME/miniconda3/bin/activate mmfb

export PYTHONPATH=$codedir:$PYTHONPATH
export PYTHONPATH=$codedir"/../HPOBench/":$PYTHONPATH

id=$1

bash $codedir/scripts/dask/deploy_workers.sh $nworkers $wspace/scheduler/scheduler.json $wspace $id