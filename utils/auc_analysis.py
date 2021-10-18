import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import DBSCAN, KMeans

from utils.util import load_yaml_args


def load_benches(path, model):
    df = pd.read_parquet(path.format(model, model))
    df = df.to_dict(orient="list")
    for k, v in df.items():
        df[k] = ["{}_{}".format(model, tid) for tid in v]
    return df


def table_of_auc_classes(auc_map):
    table = dict(low=dict(), med=dict(), high=dict())
    for model in models:  # sans `nn`
        for k in table.keys():
            table[k].update({model: auc_map[model][k]})
    for k, v in table.items():
        table[k] = pd.DataFrame.from_dict(v)
        for col in table[k].columns:
            table[k][col] = table[k][col].sort_values(ignore_index=True)
    return table


if __name__ == "__main__":
    path = Path("/media/neeratyoy/Mars/Freiburg/Thesis/code/MMFB/results/")
    models = ["svm", "lr", "rf", "xgb", "nn"]

    df = dict()
    for model in models:
        df[model] = load_yaml_args(path / model / "auc/{}_aucs.yaml".format(model))
    df = pd.DataFrame.from_dict(df)
    _df = df.copy()
    tids = df.index

    auc_map = dict()
    size = 5
    mid_lower = df.shape[0] // 2 - size // 2
    for model in models:
        auc_map[model] = dict(low=[], med=[], high=[])
        auc_map[model]["low"] = df[model].sort_values().index[:size].values
        auc_map[model]["med"] = df[model].sort_values().index[mid_lower:mid_lower+size].values
        auc_map[model]["high"] = df[model].sort_values().index[-size:].values
        auc_map[model] = pd.DataFrame.from_dict(auc_map[model])
        auc_map[model].to_parquet(path / model / "auc/{}_auc_map.gzip".format(model))

    # without NN
    df = df.drop("nn_", axis=1)
    clus_km = KMeans(n_clusters=2).fit(df.values)
    labels = clus_km.labels_

