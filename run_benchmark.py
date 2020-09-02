import time
import itertools
import numpy as np
import ConfigSpace as CS
from benchmark import RandomForestBenchmark


if __name__ == "__main__":

    task_ids = [
        3,       # kr-vs-kp
        # 168335,  # MiniBooNE   ### why failed???
        # 168331,  # volkert
        31,      # credit-g
        9977     # nomao
    ]

    seed = np.random.randint(1, 1000)
    n_configs = 5
    fidelity_space_granularity = 5
    results = {}

    benchmark = RandomForestBenchmark(task_id=task_ids[0], seed=seed)

    # Create list of configs
    configs = benchmark.get_config(size=n_configs)

    # Create list of fidelities
    fidelity_grid = []
    eps = 1e-10
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
    evaluations = list(itertools.product(*([task_ids, configs, fidelities])))

    start = time.time()
    for i in range(len(evaluations)):
        print("Evaluating {} out of {}".format(
            i + 1, len(task_ids) * n_configs * fidelity_space_granularity ** 2
        ), end='\r')
        task_id, config, fidelity = evaluations[i]
        benchmark = RandomForestBenchmark(task_id=task_id, seed=seed)
        benchmark.load_data_automl()
        # if task_id not in results.keys():
        #     results[task_id] = []
        # results[task_id].append(benchmark.objective(config, fidelity))
        results.update({
            (task_id,
             config.__hash__(),
             fidelity.__hash__(),
             seed): benchmark.objective(config, fidelity)
        })

    print("Time taken since beginning: {:<.5f} seconds".format(time.time() - start))

    import pickle
    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)
