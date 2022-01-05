#! /bin/bash

ws_allocate hpobench 100

cd '/work/ws/nemo/'$USER'-hpobench-0/'

mkdir scheduler
mkdir openml_splits
mkdir msub-logs

cd $HOME