#! /bin/bash

#MSUB -l nodes=1:ppn=4
#MSUB -l walltime=96:00:00
#MSUB -N collator-svm
#MSUB -o /work/ws/nemo/fr_nm1068-hpobench-0/msub-logs
#MSUB -e /work/ws/nemo/fr_nm1068-hpobench-0/msub-logs

codedir=$HOME'/Thesis/code/MMFB'
wspace='/work/ws/nemo/fr_nm1068-hpobench-0/'
space='svm'
sleep=10
njobs=4  # should match msub -l ppn
path=$wspace'/full/svm/1'

source $HOME/miniconda3/bin/activate mmfb

export PYTHONPATH=$codedir:$PYTHONPATH
export PYTHONPATH=$codedir"/../HPOBench/":$PYTHONPATH

echo "Collecting from "$path

PYTHONPATH=$codedir python3 $codedir/file_collator_joblib.py --sleep $sleep --path $path \
  --config $codedir'/arguments/nemo/full/'$space'_args.yaml' --n_jobs $njobs