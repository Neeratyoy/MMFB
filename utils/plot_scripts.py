import os
import pickle
import numpy as np


def get_curve_from_trace(plt, full_trace, label, global_best=0):
    trace = []
    costs = []
    inc = np.inf
    for entry in full_trace:
        if entry["function_value"] < inc:
            inc = entry["function_value"]
        trace.append(inc)
        costs.append(entry["cost"])
    plt.plot(np.cumsum(costs), np.array(trace) - global_best, label=label)
    return plt


def plot_comparison(plt, path, space, task_id, global_best=0, eval_type="val"):
    filepath = os.path.join(path, space, str(task_id))
    plt.clf()
    plt.tight_layout()
    # RS - full budget
    with open(os.path.join(filepath, "rs_{}_{}_{}_full.pkl".format(space, task_id, eval_type)), "rb") as f:
        full_trace = pickle.load(f)
    plt = get_curve_from_trace(plt, full_trace, "RS (full)", global_best)
    with open(os.path.join(filepath, "rs_{}_{}_{}_multi.pkl".format(space, task_id, eval_type)), "rb") as f:
        full_trace = pickle.load(f)
    plt = get_curve_from_trace(plt, full_trace, "RS (multi)", global_best)
    with open(os.path.join(filepath, "dehb_{}_{}_{}.pkl".format(space, task_id, eval_type)), "rb") as f:
        full_trace = pickle.load(f)
    plt = get_curve_from_trace(plt, full_trace, "DEHB", global_best)
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Task ID {} for {}".format(task_id, space))
    plt.xlabel("Wallclock time in seconds")
    plt.ylabel("Loss regret")
    plt.legend()
    filename = "{}_{}_{}".format(space, task_id, eval_type)
    plt.savefig(os.path.join(path, space, "{}.png".format(filename)))
