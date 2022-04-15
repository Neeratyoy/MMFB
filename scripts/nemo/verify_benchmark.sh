#! /bin/bash

model=$1
fidelity=1

#path="tmp_dump/test/"$model"/"$fidelity"/benchmark/"
#path="nemo_dump/"$model"/"$fidelity"/benchmark/"
path="/work/ws/nemo/fr_nm1068-hpobench-0/full/"$model/$fidelity"/benchmark/"

python utils/verify_pre_upload.py --path $path --model $model --skip_test
