import os
import glom
import pickle
import argparse
import itertools
import numpy as np
from collections import OrderedDict

from hpobench.benchmarks.ml.ml_benchmark_template import metrics
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

    assert args.path is not None
    if not os.path.isfile(args.path):
        raise FileNotFoundError("Provided file path doesn't exist: {}".format(args.path))
    with open(args.path, "rb") as f:
        table = pickle.load(f)

    exp_args = table['exp_args']
    config_spaces = table['config_spaces']
    x = config_spaces['x']
    x_discrete = get_discrete_configspace(x, exp_args['x_grid_size'])
    table['config_spaces']['x_discrete'] = x_discrete
    z = config_spaces['z']
    z_discrete = get_discrete_configspace(z, exp_args['z_grid_size'], fidelity_space=True)
    table['config_spaces']['z_discrete'] = z_discrete

    param_list = []
    for name in np.sort(x_discrete.get_hyperparameter_names()):
        hp = x_discrete.get_hyperparameter(str(name))
        param_list.append(hp.sequence)
    for name in np.sort(z_discrete.get_hyperparameter_names()):
        hp = z_discrete.get_hyperparameter(str(name))
        param_list.append(hp.sequence)

    count = 0
    incumbents = dict()
    for m in metrics.keys():
        incumbents[m] = dict(train_scores=np.inf, val_scores=np.inf, test_scores=np.inf)

    for entry in itertools.product(*param_list):
        key_path = entry
        # key_path = [np.float32(_key) for _key in key_path]
        val = glom.glom(table['data'], glom.Path(*key_path), default=None)
        if val is None:
            print(key_path)
            raise ValueError("Table contains no entry for given config-fidelity combination!")
        for seed in val.keys():
            count += 1
            print(count, seed, val[seed], '\n')
            for m in metrics.keys():
                for k, v in incumbents[m].items():
                    if 1 - val[seed]['info'][k][m] < v:  # loss = 1 - accuracy
                        incumbents[m][k] = 1 - val[seed]['info'][k][m]
    print(incumbents)
    table['global_min'] = dict()
    for m in metrics.keys():
        table['global_min'][m] = dict(
            train=incumbents[m]["train_scores"],
            val=incumbents[m]["val_scores"],
            test=incumbents[m]["test_scores"]
        )
    if count != table['progress']:
        raise ValueError("Count mismatch: {} vs {}".format(count, table['progress']))
    with open(args.path, "wb") as f:
        pickle.dump(table, f)
