import os
import argparse

import matplotlib.pyplot as plt

from utils.util import dump_yaml_args
from plotters.plot_utils import rank_incumbent_across_fidelities


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


def collate_aucs():
    dfs = []
    for model in models:
        yaml_path = Path(path) / model / "auc/{}_aucs.yaml".format(model)
        _auc = load_yaml_args(yaml_path)
        dfs.append(pd.DataFrame.from_dict(_auc, columns=["AUC"], orient="index"))
    df = pd.concat(dfs, axis=1)
    _models = [m.upper() for m in models]
    df.columns = ["SVM", "LogReg", "RF", "XGB", "MLP"]
    df = df.reindex(paper_tasks)
    df.boxplot(grid=False, fontsize=20, rot=15)
    plt.title("AUC score distribution across space", fontsize=20)
    plt.tight_layout()
    plt.savefig(Path(path) / "auc_boxplot.pdf")


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
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "pdf"],
        help="plot format"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = input_arguments()
    models = list(ntasks_done.keys()) if args.model is None else args.model
    for model in models:
        output_path = os.path.join(args.output_path, model, "auc")
        os.makedirs(output_path, exist_ok=True)
        tids = paper_tasks[:ntasks_done[model]] if args.task_id is None else args.task_id
        aucs = rank_incumbent_across_fidelities(
            model, tids, args.load_path, output_path, args.format
        )
        auc_map = {tid: float(aucs[i]) for i, tid in enumerate(tids)}
        dump_yaml_args(auc_map, os.path.join(output_path, "{}_aucs.yaml".format(model)))
        print("Saved at", os.path.join(output_path, "{}_aucs.yaml".format(model)))
