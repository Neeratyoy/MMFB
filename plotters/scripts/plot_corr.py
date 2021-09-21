import os
import argparse

import matplotlib.pyplot as plt

from utils.util import dump_yaml_args
from plotters.plot_utils import line_box_corr_plot, fidelity_names


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
        "--model",
        default=list(ntasks_done.keys()),
        type=str,
        nargs="+",
        help="model space"
    )
    parser.add_argument(
        "--task_id",
        default=None,
        type=int,
        nargs="+",
        help="task_ids"
    )
    parser.add_argument(
        "--load_path",
        default=None,
        type=str,
        help="path of interest"
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
    models = list(ntasks_done.keys()) if args.model is None else args.model
    for model in models:
        print(model)
        output_path = os.path.join(args.output_path, model, "corr")
        tids = paper_tasks[:ntasks_done[model]] if args.task_id is None else args.task_id
        plt.clf()
        plt = line_box_corr_plot(
            plt, tids, fidelity_names[model], model, args.load_path, output_path
        )
    print("Done!")
