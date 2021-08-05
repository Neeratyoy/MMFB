#!/ bin/bash/

codedir=$HOME'/Thesis/code/MMFB'

model=$1

echo "Submitting scheduler"
msub $codedir/scripts/nemo/create_scheduler.sh scheduler_$model

echo -e "\nSubmitting collator"
python scripts/nemo/submit_collator.py --space $model

echo -e "\nSubmitting first 100 workers"
msub -t 1-5 $codedir/scripts/nemo/arrayjob.moab scheduler_$model