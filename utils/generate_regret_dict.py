import os
import yaml
import argparse
from joblib.parallel import Parallel, parallel_backend, delayed

from hpobench.benchmarks.ml import TabularBenchmark
from HPOBenchExperimentUtils.utils.runner_utils import load_benchmark_settings


def update_dict_entry(key, path, time_limit):
    model, task_id = key.split("_")
    task_id = int(task_id)
    benchmark = TabularBenchmark(data_dir=path, model=model, task_id=task_id)
    ystar_valid = benchmark.global_minimums["val"]["acc"]
    ystar_test = benchmark.global_minimums["test"]["acc"]
    template_dict = dict(
        xlim_lo=10 ** -1,
        ylim_lo=0,
        ylim_up=1,
        xscale="log",
        yscale="log",
        ystar_valid=ystar_valid,
        ystar_test=ystar_test
    )
    config_dict = {key: template_dict}
    return config_dict


def check_key(key, model_list=["svm", "lr", "rf", "xgb", "nn"]):
    key_split = key.split("_")
    if len(key_split) != 2:
        return False
    try:
       _ = int(key_split[1])
    except:
        return False
    if key_split[0] in model_list:
        return True
    return False


def input_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--path",
        default="nemo_dump/TabularData",
        type=str,
        help="The path where all models directories are"
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        help="The path to dump yaml file"
    )
    parser.add_argument(
        "--n_jobs",
        default=4,
        type=int,
        help="number of cores"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = input_arguments()

    benches = load_benchmark_settings()
    benches = {k:v for k, v in benches.items() if check_key(k)}

    with parallel_backend(backend="multiprocessing", n_jobs=args.n_jobs):
        dicts = Parallel()(
            delayed(update_dict_entry)(
                key, args.path, value["time_limit_in_s"]
            ) for key, value in benches.items()
        )
    tabular_dict = dict()
    for entry in dicts:
        tabular_dict.update(entry)
    print("Collected {} keys...".format(len(tabular_dict)))
    with open(os.path.join(args.output_path, "tabular_plot_config.yaml"), "w") as f:
        f.writelines(yaml.dump(tabular_dict))
    print("Done!")
