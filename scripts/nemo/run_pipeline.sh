#! /bin/bash

codedir=$HOME'/Thesis/code/MMFB'
wspace='/work/ws/nemo/fr_nm217-hpobench-0/'

msub $codedir/scripts/nemo/create_scheduler.sh
sleep 10
msub $codedir/scripts/nemo/create_workers.sh
sleep 10
msub $codedir/scripts/nemo/run_benchmark.sh
sleep 10
msub $codedir/scripts/nemo/run_collator.sh