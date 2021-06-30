# MMFB


The `run_benchmark_dask.py` script runs the tabular benchmark collection for a given model 
parameter space, for a chosen fidelity. The script can take command line arguments that 
are processed via `argparse`. It can also read predefined `yaml` config files. For example,
```python
python run_benchmark_dask.py --config arguments/toy/svm_args.yaml
```

The script creates the following directory structure based on the arguments passed. 
The folder `dump/` contains the dict output of each and every function evaluation for a run.
The dicts are stored in the format of `[task_id]_[md5 has of config]_[md5 hash of fidelity]_[seed].pkl` 
and therefore the script can be run for multiple tasks, seeds, at the same time.

```
..    
|
└───args.output_path/
│   └───args.exp_name (optional level)/
|   |   └───args.space
|   |   |   └───args.fidelity_choice
|   |   |   |   └───dump/
|   |   |   |   |   └───logs/
|   |   |   |   |   |   └───run_*.log
|   |   |   |   |   |   └───collator_*.log
|   |   |   |   |   └───dump/
|   |   |   |   |   |   └───... (pickle files dumped after each fn eval)
|   |   |   |   |   └───benchmark/
|   |   |   |   |   |   └───task_3.pkl
|   |   |   |   |   |   └───task_12.pkl
|   |   |   |   |   |   └───...
|   |   |   |   |   |   └───task_*.pkl
|   |   |   |   |   └───param_space.pkl (contains fidelity ConfigSpace dump)
|   |   |   |   └───param_space.pkl (contains model ConfigSpace dump) 
```

The above directory structure is important since the `file_collator.py` script relies on it. 
The collator needs to know the path till `args.fidelity_choice`. Following which it collects 
any files stored under `dump/`, collates them into a pickle file per task_id, and saves it under 
`benchmark/`.

Therefore, each instantiation of `file_collator.py` is designed to work for a *space-fidelity* 
combination.

**NOTE**: Need to pip install [this](https://github.com/Neeratyoy/HPOBench/tree/thesis-paper)
or clone it and include it in PYTHONPATH for `run_benchmark_dask` to work  