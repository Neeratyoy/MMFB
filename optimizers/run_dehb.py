import os
import sys
import pickle
import argparse
import numpy as np
from matplotlib import pyplot as plt

from hpobench.benchmarks.ml import TabularBenchmark
from hpobench.benchmarks.ml.ml_benchmark_template import metrics

from dehb import DEHB


def target_function(config, budget, **kwargs):
    benchmark = kwargs["benchmark"]
    metric = kwargs["metric"]
    test = kwargs["test"]
    fidelity_name = kwargs["fidelity_name"]
    fidelity = benchmark.z_cs.sample_configuration()
    if isinstance(fidelity[fidelity_name], (int, np.int32, np.int64)):
        fidelity[fidelity_name] = int(budget)
    else:
        fidelity[fidelity_name] = np.float32(budget)
    if test:
        res = benchmark.objective_function_test(config, fidelity, metric=metric)
    else:
        res = benchmark.objective_function(config, fidelity, metric=metric)
    res["fitness"] = res["function_value"]
    res.pop("function_value")
    return res


def process_history_for_plotting(history):
    full_trace = []
    for h in history:
        res = dict()
        res['function_value'] = h[1]
        res['cost'] = h[2]
        res["info"] = h[-1]
        full_trace.append(res)
    return full_trace


def input_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--path",
        default=None,
        type=str,
        help="the filepath to the tabular data files"
    )
    parser.add_argument(
        "--model",
        default="svm",
        type=str,
        help="the model space for the hyperparameters"
    )
    parser.add_argument(
        "--task_id",
        default=10101,
        type=int,
        help="the task ID to load"
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="seed to initalize the config spaces"
    )
    parser.add_argument(
        "--fevals",
        default=None,
        type=int,
        help="number of function evaluations"
    )
    parser.add_argument(
        "--metric",
        default="acc",
        type=str,
        choices=list(metrics.keys()),
        help="score metrics to choose from"
    )
    parser.add_argument(
        "--test",
        default=False,
        action="store_true",
        help="to return results on test split"
    )
    parser.add_argument(
        "--output_path",
        default="./results",
        type=str,
        help="output path to dump optimisation trace and plot"
    )
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="prints progress to stdout"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = input_arguments()
    print(os.environ["PYTHONPATH"])

    benchmark = TabularBenchmark(
        path=args.path, model=args.model, task_id=args.task_id, seed=args.seed
    )
    task_id = benchmark.exp_args['task_id']
    space = benchmark.exp_args['space']
    max_fidelity = benchmark.get_max_fidelity()
    fidelity_info = benchmark.get_fidelity_range()
    if len(fidelity_info) > 1:
        assert "Supports only multi-fidelity (1-d) and not multi-multi-fidelity (>1-d)!"
    fidelity_name, min_budget, max_budget = fidelity_info[0]
    global_best = benchmark.get_global_min()['val']
    eval_type = "test" if args.test else "val"
    output_path = os.path.join(args.output_path, space, str(task_id))
    os.makedirs(output_path, exist_ok=True)

    dehb = DEHB(
        f=target_function, cs=benchmark.x_cs,
        min_budget=min_budget, max_budget=max_budget, eta=3,
        n_workers=1, output_path="./dehb_dump"
    )
    trace, costs, full_trace = dehb.run(
        fevals=args.fevals, verbose=args.verbose, save_history=False, save_intermediate=False,
        # **kwargs list
        benchmark=benchmark, metric=args.metric, test=args.test, fidelity_name=fidelity_name
    )

    plt.tight_layout()
    plt.plot(np.cumsum(costs), np.array(trace) - global_best, label="DEHB")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Task ID {} for {}".format(task_id, space))
    plt.xlabel("Wallclock time in seconds")
    plt.ylabel("Loss regret")
    plt.legend()
    filename = "dehb_{}_{}_{}".format(space, task_id, eval_type)
    plt.savefig(os.path.join(output_path, "{}.png".format(filename)))
    with open(os.path.join(output_path, "{}.pkl".format(filename)), "wb") as f:
        pickle.dump(process_history_for_plotting(full_trace), f)
