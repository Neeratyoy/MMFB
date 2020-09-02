import sys
import time
import itertools
import numpy as np
import ConfigSpace as CS
from benchmark import RandomForestBenchmark

from joblib import Parallel, delayed


if __name__ == "__main__":

    print(193.20521, "==>", 98.32635, end='\n')

    num_workers = int(sys.argv[1])
    print("Executing with {} worker(s)...".format(num_workers))

    task_ids = [
        3,       # kr-vs-kp
        # 168335,  # MiniBooNE   ### why failed???
        # 168331,  # volkert
        # 31,      # credit-g
        # 9977     # nomao
    ]

    seed = np.random.randint(1, 1000)
    n_configs = 5
    fidelity_space_granularity = 5

    results = {}

    start = time.time()
    # for task_id in task_ids:
    def p1(task_id):
        benchmark = RandomForestBenchmark(task_id=task_id, seed=seed)
        benchmark.load_data_automl()
        fidelity_grid = []
        eps = 1e-10
        results[task_id] = []
        print("Beginning task_id={}\n{}\n".format(task_id, '=' * 25))

        # Create list of configs
        configs = benchmark.get_config(size=n_configs)

        # Create list of fidelities
        for i, parameter in enumerate(benchmark.f_cs.get_hyperparameters()):
            if isinstance(parameter, CS.UniformIntegerHyperparameter) and \
                    (parameter.upper - parameter.lower) < fidelity_space_granularity:
                step = 1
            else:
                step = (parameter.upper - parameter.lower) / \
                       (fidelity_space_granularity - 1)

            grid_points = np.arange(
                start=parameter.lower, stop=parameter.upper + eps, step=step
            )

            if isinstance(parameter, CS.UniformIntegerHyperparameter):
                grid_points = np.ceil(grid_points).astype(int)

            grid_points = np.clip(grid_points, a_min=None, a_max=parameter.upper)
            fidelity_grid.append(grid_points)
        fidelity_grid = list(itertools.product(*fidelity_grid))
        fidelities = []
        for i, fidelity in enumerate(fidelity_grid):
            dummy_fidelity = benchmark.f_cs.sample_configuration()
            config = fidelity[-1]
            for j, parameter in enumerate(benchmark.f_cs.get_hyperparameters()):
                dummy_fidelity[parameter.name] = fidelity[j]
            fidelities.append(dummy_fidelity)

        # all combinations of evaluations to be made
        evaluations = list(itertools.product(*([configs, fidelities])))

        with Parallel(n_jobs=num_workers) as parallel2:
            results[task_id] = parallel2(
                delayed(benchmark.objective)(evaluations[idx][0], evaluations[idx][1])
                for idx in range(len(evaluations))
            )
        return results

    # parallelizes over tasks
    with Parallel(n_jobs=num_workers) as parallel1:
        output = parallel1(delayed(p1)(task_id) for task_id in task_ids)

    for i in range(len(output)):
        results.update(output[i])

    end =time.time()
    print()
    for task_id in task_ids:
        print("\nTask_id: {}\n{}".format(task_id, "-"*25))
        print(results[task_id][0])
        print(results[task_id][-1])
    print()
    print("Time taken: {:<.5f} seconds".format(end- start))
