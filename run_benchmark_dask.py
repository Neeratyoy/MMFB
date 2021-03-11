import os
import sys
import time
import openml
import pickle
import hashlib
import argparse
import itertools
import numpy as np
from loguru import logger
from distributed import Client
from typing import Dict, Tuple
from multiprocessing.managers import BaseManager

from benchmark import RandomForestBenchmark
from utils.util import get_parameter_grid, map_to_config


logger.configure(handlers=[{"sink": sys.stdout, "level": "INFO"}])
_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB"
}


def config2hash(config):
    """ Generate md5 hash to distinguish and look up configurations
    """
    return hashlib.md5(repr(config).encode('utf-8')).hexdigest()


def return_dict(combination: Tuple) -> Dict:
    assert len(combination) == 5
    evaluation = dict()
    evaluation["task_id"] = combination[0]
    evaluation["config"] = combination[1]
    evaluation["fidelity"] = combination[2]
    evaluation["seed"] = combination[3]
    evaluation["path"] = combination[4]
    return evaluation


def compute(evaluation: dict, benchmarks: dict=None) -> str:
    """ Function to evaluate a configuration-fidelity on a task for a seed

    Parameters
    ----------
    evaluation : dict
        5-element dictionary containing all information required to perform a run
    benchmarks : dict
        a nested dictionary of the hierarchy task_id-seeds containing instantiations of the
        benchmarks for each task_id - seed pair that is shared across all Dask workers

    Returns
    -------
    str
    """
    task_id = evaluation["task_id"]
    config = evaluation["config"]
    config_hash = config2hash(config)
    fidelity = evaluation["fidelity"]
    fidelity_hash = config2hash(fidelity)
    seed = evaluation["seed"]
    path = evaluation["path"]

    if benchmarks is None:
        benchmark = RandomForestBenchmark(task_id=task_id, seed=seed)
    else:
        benchmark = benchmarks[task_id][seed]
    # the lookup dict key for each evaluation is a 4-element tuple
    result = {
        (task_id,
         config2hash(config),
         config2hash(fidelity),
         seed): benchmark.objective(config, fidelity)
    }
    # file_collator should collect the pickle files dumped below
    name = "{}/{}_{}_{}_{}.pkl".format(path, task_id, config_hash, fidelity_hash, seed)
    with open(name, 'wb') as f:
        pickle.dump(result, f)
    return "success"


class DaskHelper:
    """ Manages Dask client and provides associated helper functions.
    """
    def __init__(self, n_workers):
        self.n_workers = n_workers
        self.client = Client(
            n_workers=self.n_workers,
            processes=True,
            threads_per_worker=1,
            scheduler_port=0
        )
        self.futures = []
        self.shared_data = None

    def __getstate__(self):
        """ Allows the object to picklable while having Dask client as a class attribute.
        """
        d = dict(self.__dict__)
        d["client"] = None  # hack to allow Dask client to be a class attribute
        return d

    def __del__(self):
        """ Ensures a clean kill of the Dask client and frees up a port.
        """
        if hasattr(self, "client") and isinstance(self, Client):
            self.client.close()

    def submit_job(self, func, x):
        if self.shared_data is not None:
            self.futures.append(self.client.submit(func, x, self.shared_data))
        else:
            self.futures.append(self.client.submit(func, x))

    def is_worker_available(self, verbose=False):
        """ Checks if at least one worker is available to run a job
        """
        if self.n_workers == 1:
            # in the synchronous case, one worker is always available
            return True
        workers = sum(self.client.nthreads().values())
        if len(self.futures) >= workers:
            # pause/wait if active worker count greater allocated workers
            return False
        return True

    def fetch_futures(self, retries=1, wait_time=0.05):
        """ Removes the futures which are done from the list

        Loops over the list a given number of times, waiting for a given time, to check if any or
        more futures have finished. If at any time, all futures have been processed, it breaks out.
        """
        counter = 0
        while counter < retries:
            if self.n_workers > 1:
                self.futures = [future for future in self.futures if not future.done()]
                if len(self.futures) == 0:
                    break
            else:
                # Dask not invoked in the synchronous case (n_workers=1)
                self.futures = []
            time.sleep(wait_time)
            counter += 1
        return None

    def distribute_data_to_workers(self, data):
        """ Shares data across all workers to not serialize and transfer with every job
        """
        self.shared_data = self.client.scatter(data, broadcast=True)

    def is_worker_alive(self):
        return len(self.futures) > 0


def input_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--n_tasks",
        default=3,
        type=int,
        help="The number of tasks to run data collection on from the AutoML benchmark suite"
    )
    parser.add_argument(
        "--x_grid_size",
        default=10,
        type=int,
        help="The size of grid steps to split each dimension of the observation space.\nFor a "
             "parameter with range [0, 20] and step size of 10, the resultant grid would be "
             "[0, 2.22, 4.44, ..., 17.78, 20], using numpy.linspace."
    )
    parser.add_argument(
        "--z_grid_size",
        default=10,
        type=int,
        help="The size of grid steps to split each dimension of the fidelity space.\nFor a "
             "parameter with range [0, 20] and step size of 10, the resultant grid would be "
             "[0, 2.22, 4.44, ..., 17.78, 20], using numpy.linspace."
    )
    parser.add_argument(
        "--fidelity_choice",
        default=0,
        type=int,
        help="Choice of fidelity space:\n"
             "0: Both number of trees (n_estimators) and dataset subset fraction (subsample)\n"
             "1: n_estimators as fidelity with dataset subset fraction fixed (subsample=1)\n"
             "2: subsample as fidelity with number of trees fixed (n_estimators=100)"
    )
    parser.add_argument(
        "--n_seeds",
        default=4,
        type=int,
        help="The number of different seeds to evaluate each configuration-fidelity on."
    )
    parser.add_argument(
        "--n_workers",
        default=1,
        type=int,
        help="The number of workers to distribute the evaluation over."
    )
    parser.add_argument(
        "--output_path",
        default="./dump",
        type=str,
        help="The directory to dump and store results and benchmark files."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = input_arguments()

    # Creating storage directory
    path = os.path.join(os.getcwd(), args.output_path)
    os.makedirs(path, exist_ok=True)

    # Logging details
    log_suffix = time.strftime("%x %X %Z")
    log_suffix = log_suffix.replace("/", '-').replace(":", '-').replace(" ", '_')
    logger.add(
        "{}/run_{}.log".format(path, log_suffix),
        **_logger_props
    )
    print("Logging at {}/run_{}.log".format(path, log_suffix))

    # Load tasks
    logger.info("Loading AutoML benchmark suite from OpenML for {} tasks".format(args.n_tasks))
    automl_benchmark = openml.study.get_suite(218)
    task_ids = automl_benchmark.tasks[:args.n_tasks]
    np.random.shuffle(task_ids)

    # Selecting seeds
    seeds = np.random.randint(1, 10000, size=args.n_seeds)
    logger.info("Seeds selected {}".format(seeds))

    # Loading benchmarks
    benchmarks = dict()
    for task_id in task_ids:
        benchmarks[task_id] = dict()
        for seed in seeds:
            logger.info("Processing benchmark for task {} for seed {}".format(task_id, seed))
            benchmarks[task_id][seed] = RandomForestBenchmark(
                task_id=task_id, seed=seed, fidelity_choice=args.fidelity_choice
            )
            benchmarks[task_id][seed].load_data_automl()
    # Placeholder benchmark to retrieve parameter spaces
    benchmark = benchmarks[task_ids[0]][seeds[0]]

    # Retrieving observation space and populating grid
    x_cs = benchmark.x_cs
    logger.info("Populating grid for observation space...")
    grid_config = get_parameter_grid(x_cs, args.x_grid_size, convert_to_configspace=False)
    logger.info("{} unique configurations generated".format(len(grid_config)))

    # Retrieving fidelity spaces and populating grid
    z_cs = benchmark.z_cs
    logger.info("Populating grid for fidelity space...")
    grid_fidelity = get_parameter_grid(z_cs, args.z_grid_size, convert_to_configspace=False)
    logger.info("{} unique configurations generated".format(len(grid_fidelity)))

    # Dask initialisation
    num_workers = args.n_workers
    logger.info("Executing with {} worker(s)...".format(num_workers))
    if num_workers > 1:
        client = DaskHelper(args.n_workers)
        # Essential step:
        # More than speeding up data management by Dask, creating the benchmark class objects once
        # and sharing it across all the workers mean that for each task-seed instance, the
        # validation split remains the same across evaluations, making it a fair collection of data
        # client.distribute_data_to_workers(benchmarks)

    start = time.time()
    total_combinations = len(task_ids) * len(grid_config) * len(grid_fidelity) * args.n_seeds
    for i, combination in enumerate(
            itertools.product(*[task_ids, grid_config, grid_fidelity, seeds, [path]]), start=1
    ):
        logger.info("{}/{}".format(i, total_combinations))
        combination = list(combination)
        combination[1] = map_to_config(x_cs, combination[1])
        combination[2] = map_to_config(z_cs, combination[2])
        if num_workers == 1:
            compute(return_dict(combination), benchmarks)
            continue
        # for a combination selected, need to wait until it is submitted to a worker
        # client.submit_job() is an asynchronous call, followed by a break which allows the
        # next combination to be submitted if client.is_worker_available() is True
        while True:
            if client.is_worker_available():
                client.submit_job(compute, return_dict(combination))
                break
            else:
                client.fetch_futures(retries=1, wait_time=0.05)
    if num_workers > 1 and client.is_worker_alive():
        logger.info("Waiting for pending workers...")
    while num_workers > 1 and client.is_worker_alive():
        client.fetch_futures(retries=1, wait_time=0.05)
    end = time.time()

    logger.info("{} unique configurations evaluated on {} different fidelity combinations for {} "
                "seeds on {} different tasks in {:.2f} seconds".format(
        len(grid_config), len(grid_fidelity), args.n_tasks, args.n_seeds, end - start
    ))
