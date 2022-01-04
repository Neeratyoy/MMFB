#! /bin/bash

#MSUB -l nodes=1:ppn=2
#MSUB -l walltime=96:00:00
#MSUB -N mmfb-scheduler
#MSUB -o /work/ws/nemo/fr_nm1068-hpobench-0/msub-logs
#MSUB -e /work/ws/nemo/fr_nm1068-hpobench-0/msub-logs

codedir=$HOME'/Thesis/code/MMFB'
wspace='/work/ws/nemo/fr_nm1068-hpobench-0/'

source $HOME/miniconda3/bin/activate mmfb

export PYTHONPATH=$codedir:$PYTHONPATH
export PYTHONPATH=$codedir"/../HPOBench/":$PYTHONPATH

# important for Dask to not fail on large cluster setups
source $codedir/scripts/nemo/config.sh

name=$1

bash $codedir/scripts/dask/deploy_scheduler.sh $wspace"/scheduler/"$name".json"