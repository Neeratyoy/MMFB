#! /bin/bash

#MSUB -l nodes=1:ppn=2
#MSUB -l walltime=96:00:00
#MSUB -N svm-full-10101
#MSUB -o /work/ws/nemo/fr_nm217-hpobench-0/msub-logs
#MSUB -e /work/ws/nemo/fr_nm217-hpobench-0/msub-logs

codedir=$HOME'/Thesis/code/MMFB'
wspace='/work/ws/nemo/fr_nm217-hpobench-0/'

space='svm'
exp='full'
taskid=10101

echo "Running "$exp" experiment on "$space" for task ID "$taskid

export PYTHONPATH=$codedir:$PYTHONPATH
export PYTHONPATH=$codedir"/../HPOBench/":$PYTHONPATH

# important to record reliable benchmark costs
source $codedir/scripts/nemo/config.sh

bash $codedir/scripts/nemo/run_tabular.sh $space $exp $taskid