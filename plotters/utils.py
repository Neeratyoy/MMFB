import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau, rankdata

from run_benchmark_dask import param_space_dict
from hpobench.benchmarks.ml import TabularBenchmark


fidelity_names = dict(
    rf="n_estimators",
    lr="iter",
    xgb="n_estimators",
    svm="subsample",
    nn="iter"
)


def load_benchmark(model, task_id, path="nemo_dump/{}/1/{}/{}/"):
    """ Loads the parquet compressed tabular benchmark """
    model_path = path.format(model, model, task_id)
    benchmark = TabularBenchmark(model_path, task_id=task_id, model=model)
    vals = [res["function_value"] for res in benchmark.table.result.values]
    seeds = benchmark.table.seed.unique().tolist()
    per_seed = dict()
    for seed in seeds:
        per_seed[seed] = benchmark.table.seed == 8916
    return benchmark, vals, per_seed


def get_normalized_costs(table, fid_name):
    """ Normalizes the cost for each config based on cost on max budget """
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


def cost_loss_relation(model, tid):
    benchmark, _, _ = load_benchmark(model, tid)
    full_budget = benchmark.table[fidelity_names[model]].max()
    table = benchmark.table[benchmark.table[fidelity_names[model]] == full_budget]
    data = dict()
    data["costs"] = [res["cost"] for res in table.result.values]
    data["loss"] = [res["function_value"] for res in table.result.values]
    df = pd.DataFrame(data)
    sns.regplot(x="costs", y="loss", data=df, line_kws={"color": "black"},
                scatter_kws={"alpha": 0.1}, order=1)
    # plt.yscale("log")
    plt.ylabel("loss")
    plt.xlabel("cost")
    plt.title("Loss vs Cost of full budget evaluation for {} on Task ID {}".format(model, tid))
    plt.show()
    return df


def get_val_test_corr(table, metric="acc"):
    """ Returns the correlation scores for val-test split for each config-fidelity-seed """
    val = [res["info"]["val_scores"][metric] for res in table.result.values]
    test = [res["info"]["test_scores"][metric] for res in table.result.values]
    kendall_corr, _ = kendalltau(val, test)
    pearson_corr, _ = pearsonr(val, test)
    spearman_corr, _ = spearmanr(val, test)
    return kendall_corr, pearson_corr, spearman_corr


def regret_density_plot(plt, vals, task_id, model):
    """ Plots a density plot of the regret score for a model on a dataset """
    #TODO: currently regret is being computed by taking the minimum loss achieved across all
    # recorded evaluations across all seeds, but maybe taking the minimum as average of seeds
    # can be another option or a more valid one potentially

    # Ridge plot for all datastes -- https://www.python-graph-gallery.com/ridgeline-graph-seaborn
    regret = np.array(vals) - np.min(vals)

    plt.clf()
    sns.kdeplot(regret, x="Loss regret", clip=(0, 1), bw_method=0.1, shade=True)
    plt.xlabel("Loss regret")
    # plt.xlim(np.min(regret), np.max(regret))
    plt.xscale("log")
    plt.title("{} on Task ID: {}".format(model, task_id))
    return plt


def generate_regret_density(plt, tids, model, path):
    """ Generate density plots for all task IDs passed for a given model """
    tids = [int(t) for t in tids]
    for tid in np.sort(tids):
        print(tid, end='\r')
        benchmark, vals, per_seed = load_benchmark(model, tid)
        vals = np.array(vals)[per_seed[np.random.choice(list(per_seed.keys()))]]
        plt = regret_density_plot(plt, vals, tid, model)
        plt.savefig(os.path.join(path, "{}_{}.pdf".format(model, tid)))
    return plt


def cost_across_models_per_dataset(plt, tids, models, path):
    """ Generate boxplot distributions of costs on full budget for a dataset across model spaces """
    for tid in np.sort(tids):
        tid = int(tid)
        print(tid, end="\r")
        cost_lists = []
        for model in models:
            benchmark, val, _ = load_benchmark(model, tid)
            full_budget = benchmark.table[fidelity_names[model]].max()
            table = benchmark.table[benchmark.table[fidelity_names[model]] == full_budget]
            costs = [res["cost"] for res in table.result.values]
            cost_lists.append(costs)
        plt.clf()
        plt.boxplot(cost_lists)
        plt.xticks(range(1, len(cost_lists) + 1), models)
        plt.yscale("log")
        plt.savefig(os.path.join(path, "{}.png".format(tid)))


def calc_area_under_curve(vals, width):
    """ Calculates AUC using trapezoid rule of (a+b)*h/2 """
    area = 0
    for i in range(len(vals) - 1):
        area += (vals[i] + vals[i+1])
    area *= width / 2
    return area


def rank_incumbent_across_fidelities(model, tids, path):
    """ Plot to show how the full budget best config is not the best in lower fidelities """
    aac_list = []
    for tid in np.sort(tids):
        print(tid, end='\r')
        benchmark, val, _ = load_benchmark(model, tid)
        full_budget = benchmark.table[fidelity_names[model]].max()
        fidelities = benchmark.table[fidelity_names[model]].unique()
        seeds = benchmark.table.seed.unique()
        ranks = dict()
        for seed in seeds:
            ranks[seed] = []
            table = benchmark.table[benchmark.table.seed == seed]
            table_per_fid = dict()
            for f in fidelities:
                table_per_fid[f] = table[table[fidelity_names[model]] == f]
            val = [np.float32(res["function_value"]) for res in table_per_fid[full_budget].result]
            min_val = np.min(val)
            min_idxs = np.where(val == min_val)[0]

            for i, min_idx in enumerate(min_idxs):
                print("{:<4}/{:<4}".format(i+1, len(min_idxs)), end='\r')
                temp = []
                for f in np.sort(fidelities):
                    _val = [np.float32(res["function_value"]) for res in table_per_fid[f].result]
                    temp.append(rankdata(_val)[min_idx])
                ranks[seed].append(temp)
            ranks[seed] = np.array(ranks[seed]).median(axis=0)
        x = range(1, len(fidelities) + 1)
        df = pd.DataFrame(ranks)
        size = table_per_fid[full_budget].shape[0]
        m = (size - df.transpose().median().values) / size
        max_area = 1 * (len(fidelities) - 1)
        auc = calc_area_under_curve(m, 1)
        # area above curve (AAC)
        aac = 1 - auc / max_area
        aac_list.append(aac)
        plt.clf()
        plt.plot(x, m, color="blue")
        plt.fill_between(x, 0, m, alpha=0.3, color="blue")
        plt.fill_between(x, m, 1, alpha=0.6, color="red")
        plt.hlines(y=[1], xmin=1, xmax=len(fidelities), alpha=0.3, linestyles="--", color="gray")
        plt.vlines(x=x, ymin=0, ymax=1, alpha=0.3, linestyles="--", color="gray")
        plt.xticks(x, fidelities)
        plt.ylim(0, 1.01)
        plt.ylabel("%-tile rank of best configuration")
        plt.xlabel("fidelity values")
        plt.title("Rank of best config across fidelities for {} on Task ID {}".format(model, tid))
        plt.text(x=x[-2], y=0.60, s="AAC:\n{:.4f}".format(aac), fontdict={"fontsize": 20})
        plt.savefig(os.path.join(path, "{}_{}.png".format(model, tid)))
    plt.clf()
    plt.boxplot(aac_list)
    plt.savefig(os.path.join(path, "{}.pdf".format(model)))


def cost_density_plot(table, model, task_id):
    """  """
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
    """ Computes correlation score for loss across fidelities """
    # stores a F x F correlation matrix for fidelity values per task_id
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
    # generate pairwise correlation score for every lower fidelity with max budget for all datasets
    for i in range(len(fidelities) - 1):
        f1, f2 = fidelities[i], fidelities[-1]
        key = "{:.2f}_{:.2f}".format(f1, f2)
        corr_df[key] = []
        for k, v in model_space.items():
            corr_df[key].append(v[f1][f2])
    df = pd.DataFrame(corr_df)
    return df, model_space


def line_box_corr_plot(plt, tids, fid_name, model, path, corr_type="kendall"):
    """ Box-line plot for regret trend across datasets """
    df, model_space = full_space_fidelity_correlation(tids, fid_name, model, corr_type)
    plt.clf()
    plt.plot(range(1, df.shape[1]+1), df.mean().values)
    plt.boxplot(df.values)
    plt.xticks(range(1, df.shape[1]+1), df.columns)
    plt.title("{} correlation of fidelities to full budget for {}".format(corr_type, model))
    plt.savefig(os.path.join(path, "{}.pdf".format(model)))

    for k, v in model_space.items():
        plt.clf()
        mat = plt.matshow(v.values)
        plt.colorbar(mat, label="{} correlation".format(corr_type))
        plt.title("{} on Task ID {}".format(model, k))
        plt.savefig(os.path.join(path, "{}_{}.png".format(model, k)))
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

    # helper
    model = "xgb"
    path = "results/{}/aac".format(model)
    tids = os.listdir("nemo_dump/{}/1/{}".format(model, model))
    tids = [int(t) for t in tids]

    pass
