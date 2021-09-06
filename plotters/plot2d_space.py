import os
import argparse
import itertools
import numpy as np
from matplotlib import pyplot as plt

from plotters.plot_utils import fidelity_names

from hpobench.benchmarks.ml import TabularBenchmark


def process_table(table, model, num_hps):
    """ Function that takes the mean of loss and sum of costs for only the full budget evaluations
    """
    # extracting only info of interest, loss and cost
    table["y"] = [res["function_value"] for res in table.result.values]
    table["cost"] = [res["cost"] for res in table.result.values]
    table = table.drop("result", axis=1)
    # keeping only full budget evaluations
    fidelity = fidelity_names[model]
    full_budget = table[fidelity].max()
    table = table[table[fidelity] == full_budget]
    seeds = table.seed.unique()
    loss = np.zeros(table.shape[0] // len(seeds))
    costs = np.zeros(table.shape[0] // len(seeds))
    for seed in seeds:
        loss += table[table.seed == seed].y.values
        costs += table[table.seed == seed].cost.values
    loss = loss / len(seeds)
    final_table = table[table.seed == seed].iloc[:, :num_hps]
    final_table["y"] = loss
    final_table["cost"] = costs
    return final_table


def input_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--task_id",
        default=10101,
        type=int,
        help="task_id"
    )
    parser.add_argument(
        "--path",
        default=None,
        type=str,
        help="path of interest"
    )
    parser.add_argument(
        "--space",
        default="svm",
        type=str,
        help="model space"
    )
    parser.add_argument(
        "--output_path",
        default="./results",
        type=str,
        help="path to dump plot"
    )
    parser.add_argument(
        "--pdf",
        default=False,
        action="store_true",
        help="path to dump plot"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = input_arguments()
    benchmark = TabularBenchmark(data_dir=args.path, model=args.space, task_id=args.task_id)
    table = process_table(
        benchmark.table, args.space, len(benchmark.configuration_space.get_hyperparameters())
    )
    # hard-coding works on account of 2D hyperparameter spaces
    x1 = table.iloc[:, 0]
    x1_min, x1_max = table.iloc[:, 0].values.min(), table.iloc[:, 0].values.max()
    x2_min, x2_max = table.iloc[:, 1].values.min(), table.iloc[:, 1].values.max()
    x2 = table.iloc[:, 1]
    y = table["y"]
    plt.clf()
    contour = plt.tricontourf(x1, x2, y, levels=12, cmap="RdBu_r")
    plt.colorbar(contour, label="loss")
    plt.title("{} on Task ID {}".format(args.space.upper(), args.task_id), size=25)
    labels = np.sort(benchmark.configuration_space.get_hyperparameter_names())
    plt.xlabel(labels[0], fontsize=20)
    plt.ylabel(labels[1], fontsize=20)
    if args.space == "svm":
        plt.xscale("log", base=2)
        plt.yscale("log", base=2)
    else:
        plt.xscale("log")
        plt.yscale("log")
    xticks = list(set(list(plt.xticks()[0]) + [x1_max, x1_min]))
    yticks = list(set(list(plt.yticks()[0]) + [x2_max, x2_min]))
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.tight_layout()

    format = "pdf" if args.pdf else "png"
    plotname = os.path.join(args.output_path, "{}_{}.{}".format(args.space, args.task_id, format))
    plt.savefig(plotname)
    print("Plot saved to: ", plotname)
