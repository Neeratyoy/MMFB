import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kendalltau
from matplotlib import pyplot as plt

from hpobench.benchmarks.ml import TabularBenchmark


def load_benchmark(model, task_id, path="nemo_dump/{}/1/{}/{}/"):
    model_path = path.format(model, model, task_id)
    benchmark = TabularBenchmark(model_path, task_id=task_id, model=model)
    vals = [res["function_value"] for res in benchmark.table.result.values]
    seeds = benchmark.table.seed.unique().tolist()
    per_seed = dict()
    for seed in seeds:
        per_seed[seed] = benchmark.table.seed == 8916
    return benchmark, vals, per_seed


def get_normalized_costs(table, fid_name):
    fids = dict()
    costs = dict()
    for fid in table.iter.unique():
        fids[fid] = table[table[fid_name] == fid]
    for k, v in fids.items():
        costs[k] = np.array([res["cost"] for res in v.result.values])
    max_fid = table[fid_name].values.max()
    for k, v in costs.items():
        costs[k] /= costs[max_fid]
    return costs


def get_val_test_corr(table, metric="acc"):
    val = [res["info"]["val_scores"][metric] for res in table.result.values]
    test = [res["info"]["test_scores"][metric] for res in table.result.values]
    kendall_corr, _ = kendalltau(val, test)
    pearson_corr, _ = pearsonr(val, test)
    spearman_corr, _ = spearmanr(val, test)
    return kendall_corr, pearson_corr, spearman_corr


def loss_density_plot(plt, vals, task_id, model):
    # Ridge plot for all datastes -- https://www.python-graph-gallery.com/ridgeline-graph-seaborn
    regret = np.array(vals) - np.min(vals)
    plt.clf()
    sns.kdeplot(regret, x="Loss regret", clip=(0, 1), bw_method=0.1, shade=True)
    plt.xlabel("Loss regret")
    # plt.xlim(np.min(regret), np.max(regret))
    plt.xscale("log")
    plt.title("{} on Task ID: {}".format(model, task_id))
    return plt


def generate_loss_density(plt, tids, model, path):
    tids = [int(t) for t in tids]
    for tid in np.sort(tids):
        print(tid, end='\r')
        benchmark, vals, per_seed = load_benchmark(model, tid)
        vals = np.array(vals)[per_seed[np.random.choice(list(per_seed.keys()))]]
        plt = loss_density_plot(plt, vals, tid, model)
        plt.savefig(os.path.join(path, "{}_{}.pdf".format(model, tid)))
    return plt


# 31, 10101, 146212, 146818, 167119,

def cost_density_plot(table, model, task_id):
    costs = [res["cost"] for res in table.result.values]
    sns.kdeplot(costs, x="Wallclock time (in s)", bw_method=0.05, shade=True)
    plt.xlabel("Wallclock time (in s)")
    plt.title("{} on Task ID: {}".format(model, task_id))


def get_corr(tids, model, metric, corr_type="kendall"):
    corr_df = []
    for tid in tids:
        tid = int(tid)
        print(tid, end="\r")
        benchmark, val, _ = load_benchmark(model, tid)
        k, p, s = get_val_test_corr(benchmark.table, metric)
        corr_df.append(pd.DataFrame(dict(kendall=k, pearson=p, spearman=s), index=[tid]))
    df = pd.concat(corr_df)
    sns.boxplot(y=df[corr_type])
    return df


def get_fidelity_correlation(ax, table, fid_name, corr_type="kendall"):
    fids = dict()
    for fid in table[fid_name].unique():
        fids[fid] = table[table[fid_name] == fid]
        fids[fid] = [res["function_value"] for res in fids[fid].result.values]
    df = pd.DataFrame(fids)
    A = np.ones((len(fids), len(fids)))
    mask = np.triu(A, 1)
    A = np.ma.array(A, mask=mask)
    mat = ax.matshow(df.corr(corr_type) * A)
    plt.colorbar(mat, label="{} correlation".format(corr_type))
    plt.xticks(range(len(fids)), list(fids.keys()))
    plt.yticks(range(len(fids)), list(fids.keys()))
    ax.xaxis.set_ticks_position("bottom")
    return ax


def full_space_fidelity_correlation(tids, fid_name, model, corr_type="kendall"):
    model_space = dict()
    tids =[int(t) for t in tids]
    for tid in np.sort(tids):
        print(tid, end="\r")
        benchmark, _, _ = load_benchmark(model, tid)
        table = benchmark.table
        del benchmark
        fids = dict()
        corr_df = dict()
        fidelities = table[fid_name].unique()
        for fid in fidelities:
            fids[fid] = table[table[fid_name] == fid]
            fids[fid] = [res["function_value"] for res in fids[fid].result.values]
        df = pd.DataFrame(fids)
        model_space[tid] = df.corr(corr_type)

    for i in range(len(fidelities) - 1):
        f1, f2 = fidelities[i], fidelities[-1]
        key = "{:.2f}_{:.2f}".format(f1, f2)
        corr_df[key] = []
        for k, v in model_space.items():
            corr_df[key].append(v[f1][f2])
    df = pd.DataFrame(corr_df)
    return df


def line_box_corr_plot(plt, tids, fid_name, model, corr_type="kendall"):
    df = full_space_fidelity_correlation(tids, fid_name, model, corr_type)
    plt.clf()
    plt.plot(range(1, df.shape[1]+1), df.mean().values)
    plt.boxplot(df.values)
    plt.xticks(range(1, df.shape[1]+1), df.columns)
    plt.title("{} correlation of fidelities to full budget for {}".format(corr_type, model))
    return plt


def input_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--model",
        default=None,
        type=str,
        help="model to load"
    )
    parser.add_argument(
        "--task_id",
        default=None,
        type=int,
        help="task_id to load"
    )
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
    print("Loading benchmark...")
    benchmark, vals, per_seed = load_benchmark(args.model, args.task_id)
    print("Plotting...")
    loss_density_plot(vals, args.task_id, args.model)
    plt.show()
    pass
