import os
import pickle
import argparse
import numpy as np
from matplotlib import pyplot as plt

from hpobench.benchmarks.ml import TabularBenchmark
from hpobench.benchmarks.ml.ml_benchmark_template import metrics


def input_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--path",
        default=None,
        type=str,
        help="the complete filepath to the tabular data file"
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
        "--metrics",
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
        "--mf",
        default=False,
        action="store_true",
        help="to run random search on the fidelity space or just use maximum fidelity"
    )
    parser.add_argument(
        "--output_path",
        default="./results",
        type=str,
        help="output path to dump optimisation trace and plot"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = input_arguments()

    benchmark = TabularBenchmark(table_path=args.path, seed=args.seed)
    task_id = benchmark.exp_args['task_id']
    space = benchmark.exp_args['space']
    max_fidelity = benchmark.get_max_fidelity()
    global_best = benchmark.get_global_min()['val']
    eval = "test" if args.test else "val"

    full_trace = []
    trace = []
    inc = np.inf
    cost = []
    for i in range(1, args.fevals + 1):
        print("{:<6}/{:<6}".format(i, args.fevals), end='\r')
        config = benchmark.x_cs.sample_configuration()
        fidelity = benchmark.z_cs.sample_configuration()
        if not args.mf:
            for k, v in max_fidelity.items():
                fidelity[k] = v
        if args.test:
            res = benchmark.objective_function_test(config, fidelity, metric=args.metrics)
        else:
            res = benchmark.objective_function(config, fidelity, metric=args.metrics)
        if res["function_value"] < inc:
            inc = res["function_value"]
        trace.append(inc - global_best)
        cost.append(res["cost"])
        full_trace.append(res)
    plt.plot(np.cumsum(cost), trace, label="RS")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Task ID {} for {}".format(task_id, space))
    plt.xlabel("Wallclock time in seconds")
    plt.ylabel("Loss regret")
    plt.legend()
    filename = "rs_{}_{}_{}_{}".format(task_id, space, args.mf, args.test)
    plt.savefig(os.path.join(args.output_path, "{}.png".format(filename)))
    with open(os.path.join(args.output_path, "{}.pkl".format(filename)), "wb") as f:
        pickle.dump(full_trace, f)
