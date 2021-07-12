import os
import glom
import pickle
import argparse
import numpy as np
from collections import OrderedDict

from hpobench.benchmarks.ml.svm_benchmark import SVMBenchmark
from hpobench.benchmarks.ml.histgb_benchmark import HistGBBenchmark
from hpobench.benchmarks.ml.rf_benchmark import RandomForestBenchmark

from run_benchmark_dask import param_space_dict
from utils.util import get_discrete_configspace, load_yaml_args


def query(table, config, fidelity, seed=None):
    _config = OrderedDict(config.get_dictionary())
    _fidelity = OrderedDict(fidelity.get_dictionary())
    key_path = list(_config.values()) + list(_fidelity.values())
    val = glom.glom(table, glom.Path(*key_path), default=None)
    if val is None:
        print(key_path)
        raise ValueError("Table contains no entry for given config-fidelity combination!")
    if seed is None:
        seeds = list(val.keys())
        seed = np.random.choice(seeds)
    key_path.append(seed)
    val = glom.glom(table, glom.Path(*key_path), default=None)
    return val


def input_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--config",
        default=None,
        type=str,
        help="The complete filepath to a config file, which if present, overrides cmd arguments"
    )
    parser.add_argument(
        "--task_id",
        default=None,
        type=int,
        help="benchmark task_id"
    )
    parser.add_argument(
        "--iters",
        default=10,
        type=int,
        help="number of random samples to draw"
    )
    parser.add_argument(
        "--path",
        default=None,
        type=str,
        help="full path to benchmark file"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = input_arguments()
    _args = load_yaml_args(args.config)
    for k, v in _args.items():
        if k == "task_id":
            continue
        args.__dict__[k] = v

    benchmark = param_space_dict[args.space](
        task_id=args.task_id, seed=args.seed, fidelity_choice=args.fidelity_choice
    )
    config_space = benchmark.x_cs
    fidelity_space = benchmark.z_cs
    del benchmark

    config_space_discrete = get_discrete_configspace(config_space, args.x_grid_size)
    fidelity_space_discrete = get_discrete_configspace(fidelity_space, args.z_grid_size)

    print(config_space_discrete)
    print(fidelity_space_discrete)

    assert args.path is not None
    if not os.path.isfile(args.path):
        raise FileNotFoundError("Provided file path doesn't exist: {}".format(args.path))
    with open(args.path, "rb") as f:
        table = pickle.load(f)
    table = table['data']
    for i in range(args.iters):
        config = config_space_discrete.sample_configuration()
        fidelity = fidelity_space_discrete.sample_configuration()
        print(i, query(table, config, fidelity))
        print()

    choices = []