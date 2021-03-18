#! /bin/bash

dirpath=$HOME'/Thesis/code/MMFB/'
wspace='/work/ws/nemo/fr_nm217-mmfb-0'

# filename='tmp_dump/scheduler.json'
# envname='mmfb'

msub -l nodes=1:ppn=2 -l walltime=96:00:00 -N mmfb_scheduler -o $dirpath'logs/' -e $dirpath'logs/' utils/dask_scheduler.sh

