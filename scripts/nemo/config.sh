#! /bin/bash

# important for Dask to not fail on large cluster setups
#export DASK_DISTRIBUTED__SCHEDULER__ALLOWED_FAILURES=10
export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=100
export DASK_DISTRIBUTED__COMM__RETRY__COUNT=2