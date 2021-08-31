#!/ bin/bash/

codedir=$HOME'/Thesis/code/MMFB'

model=$1
nworkers=$2
scheduler=$3
ppn=20
ajobs=`expr $nworkers / $ppn`

echo "Submitting scheduler"
msub $codedir/scripts/nemo/create_scheduler.sh $scheduler

echo -e "\nSubmitting collator"
python scripts/nemo/submit_collator.py --space $model

echo -e "\nSubmitting first "$nworkers" workers with "$ajobs" jobs"
msub -t 1-$ajobs $codedir/scripts/nemo/arrayjob.moab $scheduler