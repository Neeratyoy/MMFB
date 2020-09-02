##################################################################
# To be run on cluster w/o internet to check for cached datasets #
##################################################################

import sys
import openml
import numpy as np


path = '/'.join(__file__.split('/')[:-2])
sys.path.append(path)
from benchmark import RandomForestBenchmark


automl_benchmark = openml.study.get_suite(218)
task_ids = automl_benchmark.tasks

# Fetches from OpenML server
for i, task_id in enumerate(task_ids, start=1):
    print("{}/{} --- {}".format(i, len(task_ids), task_id), end=" ")
    try:
        benchmark = RandomForestBenchmark(task_id=task_id, seed=np.random.randint(1, 1000))
        benchmark.load_data_automl()
        config = benchmark.cs.sample_configuration()
        fidelity = benchmark.f_cs.sample_configuration()
        benchmark.objective(config, fidelity)
    except Exception as e:
        print(e)
        continue
    print()
