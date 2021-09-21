import os
import time
import pickle
import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import make_scorer, max_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from hpobench.benchmarks.ml import TabularBenchmark

from dehb import DEHB


mae_scorer = make_scorer(max_error)


def target_function(config, budget, **kwargs):
    np.random.seed(kwargs["seed"])
    clf = RandomForestRegressor(
        **config.get_dictionary(),
        n_estimators=int(budget),
    )
    subsample = 1
    train_idx = np.random.choice(
        np.arange(len(kwargs["train_X"])),
        size=int(subsample * len(kwargs["train_X"])),
        replace=False
    )
    # Training and scoring train-valid split
    start = time.time()
    clf.fit(kwargs["train_X"].iloc[train_idx, :], kwargs["train_y"].iloc[train_idx])
    score = mae_scorer(clf, kwargs["valid_X"], kwargs["valid_y"])
    cost = time.time() - start
    result = {
        "fitness": score,
        "cost": cost,
        "info": {
            "config": config.get_dictionary(),
            "fidelity": fidelity
        }
    }
    return result


def create_datasets_from_benchmark(benchmark):
    """ Extracts 2 DataFrames for loss and cost from a TabularBenchmark
    """
    nhps = len(benchmark.configuration_space.get_hyperparameters())
    nfs = len(benchmark.fidelity_space.get_hyperparameters())
    benchmark.table["y"] = [res["function_value"] for res in benchmark.table.result.values]
    benchmark.table["cost"] = [res["cost"] for res in benchmark.table.result.values]
    benchmark.table = benchmark.table.drop("result", axis=1)
    per_seed_table = dict()
    for seed in benchmark.table.seed.unique():
        per_seed_table[seed] = benchmark.table[benchmark.table.seed == seed]
    per_config_y = np.zeros(per_seed_table[seed].shape[0])
    per_config_cost = np.zeros(per_seed_table[seed].shape[0])
    for seed in benchmark.table.seed.unique():
        per_config_y += per_seed_table[seed].y.values
        # Summing costs across seeds
        per_config_cost += per_seed_table[seed].cost.values
    # Averaging loss across seeds
    per_config_y /= len(benchmark.table.seed.unique())
    loss_df = per_seed_table[seed].iloc[:, :(nhps + nfs)]
    loss_df["loss"] = per_config_y
    cost_df = per_seed_table[seed].iloc[:, :(nhps + nfs)]
    cost_df["cost"] = per_config_cost
    return loss_df, cost_df


def create_splits(df, val_size=0.1, seed=1):
    """ Assumes last column (-1) to be the target variable and splits into train-validation
    """
    df_X = df.iloc[:, :-1]
    df_y = df.iloc[:, -1]
    train_X, valid_X, train_y, valid_y = train_test_split(
        df_X, df_y, test_size=val_size, random_state=seed
    )
    return train_X, train_y, valid_X, valid_y


def input_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--data_dir",
        default="nemo_dump/TabularData",
        type=str,
        help="The path where all models directories are"
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        help="The path to dump yaml file"
    )
    parser.add_argument(
        "--model",
        default=None,
        type=str,
        required=True,
        help="The model name"
    )
    parser.add_argument(
        "--task_id",
        default=None,
        type=int,
        required=True,
        help="The task_id"
    )
    parser.add_argument(
        "--fevals",
        default=100,
        type=int,
        help="The task_id"
    )
    parser.add_argument(
        "--n_workers",
        default=1,
        type=int,
        help="number of cores"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = input_arguments()

    output_path = Path(args.output_path)
    os.makedirs(str(output_path), exist_ok=True)

    benchmark = TabularBenchmark(data_dir=args.data_dir, model=args.model, task_id=args.task_id)
    loss_df, cost_df = create_datasets_from_benchmark(benchmark)

    benchmark = TabularBenchmark(data_dir=args.data_dir, model="rf", task_id=args.task_id)
    fidelity = benchmark.original_fs.get_hyperparameter("n_estimators")
    min_budget, max_budget = fidelity.lower, fidelity.upper

    dehb = DEHB(
        cs=benchmark.original_cs, f=target_function, min_budget=min_budget, max_budget=max_budget,
        n_workers=1, output_path="./dehb_output/"
    )

    fevals = 50
    # Loss fitting
    train_X, train_y, valid_X, valid_y = create_splits(loss_df, val_size=0.1, seed=1)
    _ = dehb.run(
        fevals=fevals, save_history=False, save_intermediate=False, verbose=True,
        # kwargs
        train_X=train_X, train_y=train_y, valid_X=valid_X, valid_y=valid_y, seed=1
    )
    config = dehb.vector_to_configspace(dehb.inc_config)
    loss_model = RandomForestRegressor(**config.get_dictionary(), n_estimators=max_budget)
    loss_model.fit(loss_df.iloc[:, :-1], loss_df.iloc[:, -1])
    with open(output_path / "surr_loss_{}_{}.pkl".format(args.model, args.task_id), "wb") as f:
        pickle.dump(loss_model, f)

    # Cost fitting
    train_X, train_y, valid_X, valid_y = create_splits(cost_df, val_size=0.1, seed=1)
    dehb.reset()
    _ = dehb.run(
        fevals=fevals, save_history=False, save_intermediate=False, verbose=True,
        # kwargs
        train_X=train_X, train_y=train_y, valid_X=valid_X, valid_y=valid_y, seed=1
    )
    cost_model = RandomForestRegressor(**config.get_dictionary(), n_estimators=max_budget)
    cost_model.fit(loss_df.iloc[:, :-1], loss_df.iloc[:, -1])
    with open(output_path / "surr_cost_{}_{}.pkl".format(args.model, args.task_id), "wb") as f:
        pickle.dump(cost_model, f)
