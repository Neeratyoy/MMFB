#! /bin/bash

codedir=$HOME'/Thesis/code/MMFB'
echo $codedir

nbatch=$1
echo $nbatch

for ((i=0; i<$nbatch; i++)); do
    echo "Submitting batch "$i;
    msub $codedir/scripts/nemo/create_workers.sh $i;
done
