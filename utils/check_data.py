##################################################################
# To be run on cluster w/o internet to check for cached datasets #
##################################################################

import sys
import openml
import argparse
import numpy as np

path = '/'.join(__file__.split('/')[:-2])
sys.path.append(path)
from benchmark import RandomForestBenchmark


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_tasks", type=int, default=3, help="Number of tasks")
    args = parser.parse_args()

    automl_benchmark = openml.study.get_suite(218)
    task_ids = automl_benchmark.tasks[:args.n_tasks]

    # Fetches from OpenML server
    for i, task_id in enumerate(task_ids, start=1):
        print("{}/{} --- {}".format(i, len(task_ids), task_id), end=" ")
        try:
            benchmark = RandomForestBenchmark(task_id=task_id, seed=np.random.randint(1, 1000))
            benchmark.load_data_automl()
            config = benchmark.x_cs.sample_configuration()
            fidelity = benchmark.z_cs.sample_configuration()
            benchmark.objective(config, fidelity)
        except Exception as e:
            print(e)
            continue
        print()
