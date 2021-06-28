#! /bin/bash

#MSUB -l nodes=1:ppn=4
#MSUB -l walltime=96:00:00
#MSUB -N rf-toy
#MSUB -o /work/ws/nemo/fr_nm217-hpobench-0/msub-logs
#MSUB -e /work/ws/nemo/fr_nm217-hpobench-0/msub-logs

codedir=$HOME'/Thesis/code/MMFB'
wspace='/work/ws/nemo/fr_nm217-hpobench-0/'

export PYTHONPATH=$PWD:$PYTHONPATH

PYTHONPATH=$PWD python3 file_collator.sh --sleep 10 --path $wspace