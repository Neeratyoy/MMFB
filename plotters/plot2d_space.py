import os
import argparse
import itertools
import numpy as np
from matplotlib import pyplot as plt

from utils.util import map_to_config

from hpobench.benchmarks.ml import TabularBenchmark


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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = input_arguments()
    benchmark = TabularBenchmark(path=args.path, model=args.space, task_id=args.task_id)
    x_cs = benchmark.x_cs
    z_cs = benchmark.z_cs
    param_list = []
    for hp_name in np.sort(benchmark.x_cs.get_hyperparameter_names()):
        hp = benchmark.x_cs.get_hyperparameter(str(hp_name))
        param_list.append(hp.sequence)
    x1 = []
    x2 = []
    y = []
    max_fidelity = benchmark.get_max_fidelity()
    for i, entry in enumerate(itertools.product(*param_list)):
        print(i, end='\r')
        hp1, hp2 = entry
        x1.append(hp1)
        x2.append(hp2)
        config = map_to_config(benchmark.x_cs, [hp1, hp2])
        fidelity = benchmark.z_cs.sample_configuration()
        for k, v in max_fidelity.items():
            fidelity[k] = v
        res = benchmark.objective_function(config, fidelity)
        y.append(res["function_value"])
    x1 = np.apply_along_axis(np.log10, 0, x1)
    x2 = np.apply_along_axis(np.log10, 0, x2)
    plt.clf()
    contour = plt.tricontourf(x1, x2, y, levels=12, cmap="RdBu_r")
    plt.colorbar(contour, label="loss")
    plt.title("{} on Task ID {}".format(args.space, args.task_id), size=15)
    labels = np.sort(benchmark.x_cs.get_hyperparameter_names())
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plotname = os.path.join(args.output_path, "{}_{}.pdf".format(args.space, args.task_id))
    plt.savefig(plotname)
    print("Plot saved to: ", plotname)
