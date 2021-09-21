import os
import argparse
import itertools
import numpy as np
from matplotlib import pyplot as plt

from plotters.plot_utils import fidelity_names, process_table

from hpobench.benchmarks.ml import TabularBenchmark


paper_tasks = [
    10101, 53, 146818, 146821, 9952, 146822, 31, 3917, 168912, 3, 167119, 12, 146212,
    168911, 9981, 167120, 14965, 146606, 7592, 9977
]
ntasks_done = dict(
    lr=20,
    svm=20,
    rf=20,
    xgb=20,
    nn=8
)


def input_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--task_id",
        default=None,
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
        default=None,
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
    models = list(ntasks_done.keys()) if args.space is None else [args.space]
    for model in models:
        tasks = paper_tasks[:ntasks_done[model]] if args.task_id is None else [args.task_id]
        for task_id in tasks:
            benchmark = TabularBenchmark(data_dir=args.path, model=model, task_id=task_id)
            table = process_table(
                benchmark.table, model, len(benchmark.configuration_space.get_hyperparameters())
            )
            # hard-coding works on account of 2D hyperparameter spaces
            x1 = table.iloc[:, 0]
            x1_min, x1_max = table.iloc[:, 0].values.min(), table.iloc[:, 0].values.max()
            x2_min, x2_max = table.iloc[:, 1].values.min(), table.iloc[:, 1].values.max()
            x2 = table.iloc[:, 1]
            y = table["y"]
            plt.clf()
            contour = plt.tricontourf(x1, x2, y, levels=12, cmap="RdBu_r")
            cbar = plt.colorbar(contour)
            cbar.set_label("loss", size=18)
            cbar.ax.tick_params(labelsize=15)
            plt.title("{} on Task ID {}".format(model.upper(), task_id), size=25)
            labels = np.sort(benchmark.configuration_space.get_hyperparameter_names())
            plt.xlabel(labels[0], fontsize=20)
            plt.ylabel(labels[1], fontsize=20)
            if model == "svm":
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
            plotname = os.path.join(args.output_path, "{}_{}.{}".format(model, task_id, format))
            plt.savefig(plotname)
            print("Plot saved to: ", plotname)
