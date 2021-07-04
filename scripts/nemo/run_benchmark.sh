#! /bin/bash

#MSUB -l nodes=1:ppn=4
#MSUB -l walltime=96:00:00
#MSUB -N rf-toy
#MSUB -o /work/ws/nemo/fr_nm217-hpobench-0/msub-logs
#MSUB -e /work/ws/nemo/fr_nm217-hpobench-0/msub-logs

codedir=$HOME'/Thesis/code/MMFB'
wspace='/work/ws/nemo/fr_nm217-hpobench-0/'

space=$1
exp=$2
taskid=$3

echo $

export PYTHONPATH=$codedir:$PYTHONPATH
export PYTHONPATH=$codedir"/../HPOBench/":$PYTHONPATH

bash $codedir/scripts/nemo/run_tabular.sh $1 $2 $3