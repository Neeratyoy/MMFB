import os
import json
import glom
import pickle
import argparse
import itertools
import numpy as np
import pandas as pd
from collections import OrderedDict
import ConfigSpace as CS
from ConfigSpace.read_and_write import json as json_cs

from hpobench.benchmarks.ml.ml_benchmark_template import metrics

from utils.util import get_discrete_configspace


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


def dump_file(missing, path, task_id, space):
    output_path = path[:len(path) - path[::-1].find("/")]
    output_path = os.path.join(output_path, "{}_{}_missing.txt".format(space, task_id))
    missing = [str(ids) for ids in missing]
    with open(output_path, "w") as f:
        f.writelines("\n".join(missing))
    return output_path


def search_benchmark(entry, df):
    # https://stackoverflow.com/questions/17071871/how-do-i-select-rows-from-a-dataframe-based-on-column-values
    mask = np.array([True] * df.shape[0])
    for i, param in enumerate(df.drop("result", axis=1).columns):
        mask *= df[param].values == entry[i]
    idx = np.where(mask)
    if len(idx) != 1:
        return None
    idx = idx[0][0]
    result = df.iloc[idx]["result"]
    return result


def json_compatible_dict(data):
    for k, v in data.items():
        if isinstance(v, dict):
            v = json_compatible_dict(v)
            data[k] = v
        if hasattr(v, "dtype"):
            if "int" in v.dtype.name:
                data[k] = int(v)
            elif "float" in v.dtype.name:
                data[k] = float(v)
    return data


def input_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
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

    metadata = dict()
    exp_args = table['exp_args']
    metadata["exp_args"] = exp_args
    config_spaces = table['config_spaces']
    x = config_spaces['x']
    x_discrete = get_discrete_configspace(x, exp_args['x_grid_size'])
    table['config_spaces']['x_discrete'] = x_discrete
    z = config_spaces['z']
    z_discrete = get_discrete_configspace(z, exp_args['z_grid_size'], fidelity_space=True)
    table['config_spaces']['z_discrete'] = z_discrete
    n_seeds = exp_args["n_seeds"]
    exp_seed = exp_args["seed"]
    np.random.seed(exp_seed)
    seeds = np.random.randint(1, 10000, size=n_seeds)
    task_id = exp_args["task_id"]
    space = exp_args["space"]

    param_list = []
    param_names = []
    for name in np.sort(x_discrete.get_hyperparameter_names()):
        hp = x_discrete.get_hyperparameter(str(name))
        param_list.append(hp.sequence)
        param_names.append(hp.name)
    for name in np.sort(z_discrete.get_hyperparameter_names()):
        hp = z_discrete.get_hyperparameter(str(name))
        param_list.append(hp.sequence)
        param_names.append(hp.name)
    param_list.append(seeds)
    param_names.append("seed")
    param_names.append("result")
    df = pd.DataFrame([], columns=param_names)
    count = 0
    incumbents = dict()
    for m in metrics.keys():
        incumbents[m] = dict(train_scores=np.inf, val_scores=np.inf, test_scores=np.inf)
    missing = []
    for count, entry in enumerate(itertools.product(*param_list), start=1):
        key_path = entry
        # key_path = [np.float32(_key) for _key in key_path]
        val = glom.glom(table['data'], glom.Path(*key_path), default=None)
        if val is None:
            missing.append(count)
            continue
        entry = [np.float32(e) for e in entry]
        entry.append(val)
        _df = pd.DataFrame([entry], index=[count], columns=param_names)
        df = df.append(_df)
        print(count, val, '\n')
        for m in metrics.keys():
            for k, v in incumbents[m].items():
                if 1 - val['info'][k][m] < v:  # loss = 1 - accuracy
                    incumbents[m][k] = 1 - val['info'][k][m]
    df[param_names[:-1]] = df.drop("result", axis=1).astype(np.float32)
    df["seed"] = df["seed"].astype(int)
    print(incumbents)
    for k, v in incumbents.items():
        v["train"] = v["train_scores"]
        v["val"] = v["val_scores"]
        v["test"] = v["test_scores"]
        v.pop("train_scores")
        v.pop("val_scores")
        v.pop("test_scores")
        incumbents[k] = v

    assert len(missing) == 0, "Incomplete collection: {} missing evaluations!\n" \
                              "Dumping missing indexes at {}".format(
        len(missing), dump_file(missing, args.path, task_id, space)
    )
    # Dumping compressed files and other metadata
    output_path = os.path.join("/".join(args.path.split("/")[:-1]), str(task_id))
    os.makedirs(output_path, exist_ok=True)
    df.to_parquet(os.path.join(output_path, "{}_{}_data.parquet.gzip".format(space, task_id)))
    print("\nCompressed table saved!")
    metadata["global_min"] = incumbents
    # Converting discrete config space values to float: np.float32 not JSON serializable
    for hp in config_spaces["x_discrete"].get_hyperparameters():
        hp.default_value = float(hp.default_value)
        hp.sequence = tuple(np.array(hp.sequence).astype(float))
    for hp in config_spaces["z_discrete"].get_hyperparameters():
        if isinstance(hp.default_value, (np.float16, np.float32, np.float64)):
            hp.default_value = float(hp.default_value)
            hp.sequence = tuple(float(val) for val in hp.sequence)
        else:
            hp.default_value = int(hp.default_value)
            hp.sequence = tuple(int(val) for val in hp.sequence)
    for k, _space in config_spaces.items():
        config_spaces[k] = json_cs.write(_space)
    metadata["config_spaces"] = config_spaces
    with open(os.path.join(output_path, "{}_{}_metadata.json".format(space, task_id)), "w") as f:
        json.dump(json_compatible_dict(metadata), f)
    print("Updated with global minimas!")
    print("All files saved!")
