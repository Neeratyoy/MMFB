"""
Script to update exp yaml args with optimization budget
"""

import os
import argparse
import numpy as np
import pandas as pd

from plotters.plot_utils import fidelity_names
from run_benchmark_dask import param_space_dict
from utils.test_benchmark_compress import json_compatible_dict
from utils.util import all_task_ids_by_in_mem_size, dump_yaml_args, generate_HB_fidelities

from hpobench.benchmarks.ml import TabularBenchmark

from run_benchmark_dask import param_space_dict


def update_yaml_entry(model, task_id, time_limit, raw=False):
    template_dict = dict(
        time_limit_in_s=1000,
        cutoff_in_s=1000,
        mem_limit_in_mb=6000,
        import_from="ml.xgboost_benchmark",
        import_benchmark="",
        main_fidelity="",
        is_surrogate=True,
        bench_args=dict(task_id=12345, model="model")
    )
    template_dict["time_limit_in_s"] = time_limit
    template_dict["cutoff_in_s"] = time_limit
    template_dict["bench_args"]["task_id"] = task_id
    template_dict["bench_args"]["model"] = model
    template_dict["import_from"] = "ml.tabular_benchmark"
    template_dict["import_benchmark"] = "TabularBenchmark"
    template_dict["main_fidelity"] = fidelity_names[model]
    if raw:
        template_dict.update({"import_from": "ml"})
        # imports the MF version
        template_dict.update({"import_bencmark": param_space_dict[model][1].__name__})
        template_dict.update({"is_surrogate": False})
    return template_dict


def input_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--model",
        default="svm",
        type=str,
        choices=param_space_dict.keys(),
        help="The model to load"
    )
    parser.add_argument(
        "--task_id",
        default=None,
        type=int,
        nargs="+",
        choices=all_task_ids_by_in_mem_size,
        help="The model to load"
    )
    parser.add_argument(
        "--path",
        default=None,
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
        "--nsamples",
        default=100,
        type=int,
        help="Number of random samples to draw"
    )
    parser.add_argument(
        "--min_runtime",
        default=120,
        type=float,
        help="Minimum runtime to cap to"
    )
    parser.add_argument(
        "--max_runtime",
        default=None,
        type=float,
        help="Minimum runtime to cap to"
    )
    parser.add_argument(
        "--hb_brackets",
        default=1,
        type=int,
        help="Number of HB brackets to compute optimization budget over"
    )
    parser.add_argument(
        "--raw",
        default=False,
        action="store_true",
        help="To run raw benchmark and not tabular"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = input_arguments()
    print(args)

    model_yaml = dict()

    if args.task_id is None:
        _path = os.path.join(args.path, args.model)
        args.task_id = os.listdir(_path)
        args.task_id = [int(tid) for tid in args.task_id if os.path.isdir(os.path.join(_path, tid))]

    if args.nsamples is not None:
        raise NotImplementedError("No logic to handle nsamples. `hb_brackets` calculates budget!")

    # iterating over benchmarks
    for count, task_id in enumerate(np.sort(args.task_id), start=1):
        # print(task_id, end="\r")
        benchmark = TabularBenchmark(
            data_dir=args.path, model=args.model, task_id=task_id
        )
        table = benchmark.table.copy()
        seeds = benchmark.table.seed.unique()
        min_budget = benchmark.original_fs.get_hyperparameter(fidelity_names[args.model]).lower
        max_budget = benchmark.original_fs.get_hyperparameter(fidelity_names[args.model]).upper
        del benchmark
        # subsetting per fidelity
        fidelity = fidelity_names[args.model]
        budgets = table[fidelity].unique()
        table_per_budget = {budget: table[table[fidelity] == budget] for budget in budgets}
        del table
        # calculating average cost per fidelity
        cost_per_budget = dict()
        for budget in budgets:
            # subsetting table per seed for each config-fidelity
            cost_per_seed = {
                seed: table_per_budget[budget][
                    table_per_budget[budget].seed == seed
                ] for seed in seeds
            }
            cost_per_seed = {
                seed: [res["cost"] for res in cost_per_seed[seed].result] for seed in seeds
            }
            # `cost_per_seed` contains the cost of each config-fidelity per seed
            # sum of these costs across seeds will be considered as cost for each config-fidelity
            costs = np.zeros(len(cost_per_seed[seeds[0]]))
            for seed in seeds:
                costs += cost_per_seed[seed]
            # `costs` is a list of costs per config-fidelity, summed over all seeds for a budget
            # computing average cost of a config-fidelity for this budget
            print(budget, min(costs), max(costs))
            cost_per_budget[budget] = np.mean(costs)
        # calculating number of configs evaluated per fidelity for an HB bracket
        hb_map = generate_HB_fidelities(
            min_budget=min_budget, max_budget=max_budget, eta=3, hb_brackets=args.hb_brackets
        )
        # computing optimization budget cost for specified HB brackets
        time_limit_in_s = np.max((
            np.sum([cost_per_budget[budget] * hb_map[budget] for budget in budgets]),
            args.min_runtime
        ))
        if args.max_runtime is not None:
            time_limit_in_s = np.min((time_limit_in_s, args.max_runtime))
        print("{:<2}. Task ID {}: {} seconds -- {}".format(
            count, task_id, time_limit_in_s, np.log10(time_limit_in_s))
        )
        # Update yaml config
        new_config = update_yaml_entry(args.model, task_id, time_limit_in_s, raw=args.raw)
        if args.raw:
            key_name = "{}_{}_raw".format(args.model, task_id)
        else:
            key_name = "{}_{}".format(args.model, task_id)
        model_yaml[key_name] = new_config
    model_yaml = json_compatible_dict(model_yaml)
    filename = "benchmark_settings_raw" if args.raw else "benchmark_settings"
    file_path = os.path.join(args.output_path, "{}_{}.yaml".format(args.model, filename))
    dump_yaml_args(model_yaml, file_path)
    print("Saved at {}!".format(file_path))
