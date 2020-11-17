import os
import sys
import time
import pickle
import hashlib
import itertools
import numpy as np
import ConfigSpace as CS
from benchmark import RandomForestBenchmark

from dask.distributed import Client, wait
from dask.diagnostics import ProgressBar


MAX_TASK_LIMIT_SIZE = 1000  # to have a load limit on number of futures done/pending kept in memory


def config2hash(config):
    """ Generate md5 hash to distinguish and look up configurations
    """
    return hashlib.md5(repr(config).encode('utf-8')).hexdigest()


if __name__ == "__main__":

    num_workers = int(sys.argv[1])
    print("Executing with {} worker(s)...".format(num_workers))
    client = Client(n_workers=num_workers, processes=True, threads_per_worker=1)
    total_compute_alloted = sum(client.nthreads().values())
    print(client)

    # sample tasks
    # TODO: decide properly
    task_ids = [
        3,       # kr-vs-kp
        # 168335,  # MiniBooNE   ### why failed???
        # 168331,  # volkert
        31,      # credit-g
        # 9977     # nomao
    ]

    n_configs = 5  # number of configurations to be evaluated
    fidelity_space_granularity = 5  # the number of partitions of each fidelity dimension
    n_seeds = 4  # number of seeds each config-fidelity will be evaluated
    seeds = np.random.randint(1, 1000, size=n_seeds)  # each run will collect for different seeds
    results = {}

    # placeholder initialisation with one of the tasks
    benchmark = RandomForestBenchmark(task_id=task_ids[0], seed=seeds[0])

    # Create list of configs
    config_list = benchmark.get_config(size=n_configs)
    configs = {}
    # storing a fixed 'n_configs' set that will be evaluated across all task IDs
    for config in config_list:
        configs.update({config2hash(config): config})

    # create list of fidelities
    fidelity_grid = []
    eps = 1e-10  # to allow the < 'upper' parameter of np.arange to select 'parameter.upper'
    # 'benchmark.f_cs' contains list of fidelities (2 or more)
    for i, parameter in enumerate(benchmark.f_cs.get_hyperparameters()):
        upper, lower = (10 ** parameter.upper, 10 ** parameter.lower) if parameter.log \
            else (parameter.upper, parameter.lower)
        if isinstance(parameter, CS.UniformIntegerHyperparameter) and \
                (upper - lower) < fidelity_space_granularity:
            step = 1
        else:
            step = (upper - lower) / (fidelity_space_granularity - 1)
        # creates a sequence of points on a fidelity dimensions spanning the range
        grid_points = np.arange(
            start=lower, stop=upper + eps, step=step
        )
        if parameter.log:
            grid_points = np.log10(grid_points)

        if isinstance(parameter, CS.UniformIntegerHyperparameter):
            grid_points = np.ceil(grid_points).astype(int)

        grid_points = np.clip(grid_points, a_min=None, a_max=parameter.upper)
        fidelity_grid.append(grid_points)

    # creates a grid of all combinations of divisions of each fidelity dimension (fidelity_grid)
    fidelity_grid = list(itertools.product(*fidelity_grid))
    fidelity_list = []  # to store all combinations of the fidelity space as ConfigSpace
    for i, fidelity in enumerate(fidelity_grid):
        dummy_fidelity = benchmark.f_cs.sample_configuration()
        for j, parameter in enumerate(benchmark.f_cs.get_hyperparameters()):
            dummy_fidelity[parameter.name] = fidelity[j]
        fidelity_list.append(dummy_fidelity)

    fidelities = {}
    # storing a fixed 'fidelity_list' that will be used to evaluate all configs across all tasks
    for f in fidelity_list:
        fidelities.update({config2hash(f): f})

    path = [os.path.join('/'.join(__file__.split('/')[:-1]), 'tmp_dump')]

    # one of the most crucial steps since it collects all possible jobs that will be scheduled
    # all combinations of evaluations to be made
    # len(evaluations) = n_configs * fidelity_granularity * n_fidelities
    evaluations = list(itertools.product(
        *([task_ids, list(configs.keys()), list(fidelities.keys()), list(path)])
    ))  #TODO: could be a large list so may want to process in batches
    np.random.shuffle(evaluations)
    # 'evaluations' contains a randomly ordered list where each element will be submitted to a
    # worker informing it of the task_id to run on, the configuration to evaluate and the fidelity

    # function to be submitted to workers for evaluation
    def loop_fn(evaluation):
        task_id, config_hash, fidelity_hash, path = evaluation
        collated_result = {}
        for seed in seeds:  # seeds available in the scope
            benchmark = RandomForestBenchmark(task_id=task_id, seed=seed)
            benchmark.load_data_automl()
            # the lookup dict key for each evaluation is a 4-element tuple
            result = {
                (task_id,
                 config_hash,
                 fidelity_hash,
                 seed): benchmark.objective(configs[config_hash], fidelities[fidelity_hash])
            }
            collated_result.update(result)
        # file_collator should collect the pickle files dumped below
        with open("{}/{}_{}_{}.pkl".format(path, task_id, config_hash, fidelity_hash), 'wb') as f:
            pickle.dump(collated_result, f)
        return [{}]

    start = time.time()
    futures = []
    run_history = []
    total_wait = 0

    # submitting jobs to Dask
    if len(evaluations) < MAX_TASK_LIMIT_SIZE:
        futures = client.map(loop_fn, evaluations)
        wait_start = time.time()
        wait(futures, return_when='ALL_COMPLETED')
        total_wait += time.time() - wait_start
    else:
        for evaluation in evaluations:
            done_futures = []

            # adding new evaluation to be made
            ### at this point the length of futures may become larger than total_compute_alloted
            ### however, the number of incomplete tasks are always under total_compute_alloted
            futures.append(client.submit(loop_fn, evaluation))

            # maintain queue length as the total_compute_alloted available
            if len(futures) > MAX_TASK_LIMIT_SIZE:
                # wait till at least one future is released for the next evaluation
                wait_start = time.time()
                done_futures.extend(wait(futures, return_when='FIRST_COMPLETED').done)
                total_wait += time.time() - wait_start
            else:
                done_futures.extend([f for f in futures if f.done()])

            # storing result and clean up
            for future in done_futures:
                # run_history.append(future.result())
                run_history.extend(future.result())
                futures.remove(future)
        # end of loop to distribute jobs

    print("Gathering {} futures".format(len(futures)))
    for i, result in enumerate(client.gather(futures), start=0):
        print("{}/{}".format(i, len(futures)), end='\r')
        run_history.extend(result)

    results = {}
    for i in range(len(run_history)):
        results.update(run_history[i])

    client.close()

    print("Time taken since beginning: {:<.5f} seconds".format(time.time() - start))
    print("Total time spent waiting: {:<.5f} seconds".format(total_wait))
    print("Total number of jobs submitted: {}".format(len(evaluations)))
    print("Total number of results logged: {}".format(len(run_history)))
