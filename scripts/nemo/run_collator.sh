#! /bin/bash

#MSUB -l nodes=1:ppn=4
#MSUB -l walltime=96:00:00
#MSUB -N collator-svm
#MSUB -o /work/ws/nemo/fr_nm217-hpobench-0/msub-logs
#MSUB -e /work/ws/nemo/fr_nm217-hpobench-0/msub-logs

codedir=$HOME'/Thesis/code/MMFB'
wspace='/work/ws/nemo/fr_nm217-hpobench-0/'
model="svm"  # should match msub -N
sleep=10
njobs=4  # should match msub -l ppn
path=$1

source $HOME/miniconda3/bin/activate mmfb

export PYTHONPATH=$codedir:$PYTHONPATH
export PYTHONPATH=$codedir"/../HPOBench/":$PYTHONPATH

PYTHONPATH=$codedir python3 $codedir/file_collator_joblib.py --sleep $sleep --path $wspace$path \
  --config $codedir'/arguments/nemo/full/'$model'_args.yaml' --n_jobs $njobs