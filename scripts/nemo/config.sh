#! /bin/bash

# important for Dask to not fail on large cluster setups
#export DASK_DISTRIBUTED__SCHEDULER__ALLOWED_FAILURES=10
export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=100
export DASK_DISTRIBUTED__COMM__RETRY__COUNT=2

# allows Dask schedulers to redistribute worker load
export DASK_DISTRIBUTED__SCHEDULER__WORK_STEALING=True

# important for sklearn models to run on single CPU cores
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1