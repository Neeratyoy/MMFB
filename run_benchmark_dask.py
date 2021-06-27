import os
import sys
import time
import yaml
import openml
import pickle
import hashlib
import warnings
import argparse
import itertools
import numpy as np
import yaml
from loguru import logger
from typing import Dict, Tuple
from distributed import Client
from pympler.asizeof import asizeof

sys.path.append(os.path.join(os.path.dirname(__file__), "../HPOBench"))
from hpobench.benchmarks.ml.svm_benchmark_2 import SVMBenchmark
from hpobench.benchmarks.ml.histgb_benchmark import HistGBBenchmark
from hpobench.benchmarks.ml.rf_benchmark import RandomForestBenchmark

from utils.util import get_parameter_grid, map_to_config, DotDict, DaskHelper


logger.configure(handlers=[{"sink": sys.stdout, "level": "INFO"}])
_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB"
}

param_space_dict = dict(
    rf=RandomForestBenchmark,
    svm=SVMBenchmark,
    histgb=HistGBBenchmark
)


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
    task_path = os.path.join(path, str(task_id))
    os.makedirs(task_path, exist_ok=True)

    benchmark = benchmarks[task_id][seed]
    # the lookup dict key for each evaluation is a 4-element tuple
    result = benchmark.objective(config, fidelity)
    result['info']['seed'] = seed
    # file_collator should collect the pickle files dumped below
    name = "{}/{}_{}_{}_{}.pkl".format(task_path, task_id, config_hash, fidelity_hash, seed)
    with open(name, 'wb') as f:
        pickle.dump(result, f)
    return "success"


def load_yaml_args(filename):
    with open(filename, "r") as f:
        # https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation
        args = yaml.load(f, Loader=yaml.FullLoader)
    return DotDict(args)


def dump_yaml_args(args, filename):
    with open(filename, "w") as f:
        f.writelines(yaml.dump(args))
    return


def input_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--config",
        default=None,
        type=str,
        help="The complete filepath to a config file, which if present, overrides cmd arguments"
    )
    parser.add_argument(
        "--space",
        default="rf",
        type=str,
        choices=["rf", "svm", "histgb"],
        help="The number of tasks to run data collection on from the AutoML benchmark suite"
    )
    parser.add_argument(
        "--n_tasks",
        default=None,
        type=int,
        help="The number of tasks to run data collection on from the AutoML benchmark suite"
    )
    parser.add_argument(
        "--task_id",
        default=None,
        type=int,
        help="The task_id to run from the AutoML benchmark suite"
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
             "0 : the black-box setup with max. fidelites (n_estimators=100; subsample=1\n"
             "1 : n_estimators as fidelity with dataset subset fraction fixed (subsample=1)\n"
             "2 : subsample as fidelity with number of trees fixed (n_estimators=100)\n"
             ">2: Both number of trees (n_estimators) and dataset subset fraction (subsample)"
    )
    parser.add_argument(
        "--n_seeds",
        default=3,
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
        "--scheduler_file",
        default=None,
        type=str,
        help="The file path to the Dask scheduler information."
    )
    parser.add_argument(
        "--output_path",
        default="./tmp_dump",
        type=str,
        help="The directory to dump and store results and benchmark files."
    )
    parser.add_argument(
        "--seed",
        default=1234,
        type=int,
        help="The seed for the complete benchmark collection"
    )
    parser.add_argument(
        "--exp_name",
        default=None,
        type=str,
        help="Creates an experiment directory and if script executed with cmd arguments, "
             "dumps an yaml file with the arguments at the same level"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = input_arguments()
    if args.config is not None and os.path.isfile(args.config):
        args = load_yaml_args(args.config)
        exp_name = None
    if args.config is None and args.exp_name is not None:
        exp_name = args.exp_name
        args.output_path = os.path.join(args.output_path, args.exp_name)

    # Choosing parameter space
    param_space = param_space_dict[args.space]

    # Task input check
    automl_benchmark = openml.study.get_suite(218)
    task_ids = automl_benchmark.tasks
    if args.n_tasks is None and args.task_id is None:
        warnings.warn("Both task_id or number of tasks were not specified. "
                      "Will run all {} tasks!".format(len(task_ids)))
        args.n_tasks = len(task_ids)
    elif args.n_tasks is None and args.task_id is not None:
        if args.task_id not in task_ids:
            raise ValueError("Not a valid Task ID from among: {}".format(task_ids))
        task_ids = [args.task_id]
        args.n_tasks = 1
    elif args.n_tasks is not None and args.task_id is None:
        if args.n_tasks > len(task_ids):
            warnings.warn("{} tasks not available. "
                          "Running with {} tasks instead!".format(args.n_tasks, len(task_ids)))
        task_ids = task_ids[:args.n_tasks]
    else:
        raise ValueError("Should specify either n_tasks or task_id, not both!")

    # Setting seed
    np.random.seed(args.seed)

    # Creating storage directories
    base_path = os.path.join(os.getcwd(), args.output_path, args.space)
    os.makedirs(base_path, exist_ok=True)
    if exp_name is not None:
        dump_yaml_args(args.__dict__, os.path.join(base_path, "{}_args.yaml".format(args.exp_name)))
    path = os.path.join(base_path, str(args.fidelity_choice))
    os.makedirs(path, exist_ok=True)
    os.makedirs("{}/logs".format(path), exist_ok=True)
    os.makedirs("{}/dump".format(path), exist_ok=True)
    dump_path = os.path.join(path, "dump")
    os.makedirs(dump_path, exist_ok=True)
    print("Base Path: ", base_path)
    print("Path: ", path)

    # Logging details
    log_suffix = time.strftime("%x %X %Z")
    log_suffix = log_suffix.replace("/", '-').replace(":", '-').replace(" ", '_')
    logger.add(
        "{}/logs/run_{}.log".format(path, log_suffix),
        **_logger_props
    )
    print("Logging at {}/logs/run_{}.log".format(path, log_suffix))

    # Load tasks
    logger.info("Loaded AutoML benchmark suite from OpenML for {} tasks".format(args.n_tasks))
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
            benchmarks[task_id][seed] = param_space(
                task_id=task_id, seed=seed, fidelity_choice=args.fidelity_choice
            )
            benchmarks[task_id][seed].load_data_from_openml()
    # Placeholder benchmark to retrieve parameter spaces
    benchmark = benchmarks[task_ids[0]][seeds[0]]

    # Saving a copy of the ConfigSpaces used for this run
    with open(os.path.join(base_path, "param_space.pkl"), "wb") as f:
        pickle.dump(benchmark.x_cs, f)
    with open(os.path.join(path, "param_space.pkl"), "wb") as f:
        pickle.dump(benchmark.z_cs, f)

    # Retrieving observation space and populating grid
    x_cs = benchmark.x_cs
    logger.info("Populating grid for observation space...")
    grid_config = get_parameter_grid(x_cs, args.x_grid_size, convert_to_configspace=False)
    logger.info("{} unique observations generated".format(len(grid_config)))
    logger.debug("Observation space grid size: {:.2f} MB".format(asizeof(grid_config) / 1024 ** 2))

    # Retrieving fidelity spaces and populating grid
    z_cs = benchmark.z_cs
    logger.info("Populating grid for fidelity space...")
    grid_fidelity = get_parameter_grid(z_cs, args.z_grid_size, convert_to_configspace=False)
    logger.info("{} unique fidelity configurations generated".format(len(grid_fidelity)))
    logger.debug("Fidelity space grid size: {:.2f} MB".format(asizeof(grid_fidelity) / 1024 ** 2))

    # Dask initialisation
    if args.scheduler_file is not None and os.path.isfile(args.scheduler_file):
        logger.info("Connecting to scheduler...")
        client = Client(scheduler_file=args.scheduler_file)
        client = DaskHelper(client=client)
        num_workers = client.n_workers
        logger.info("Dask Client information: {}".format(client.client))
    else:
        num_workers = args.n_workers
        client = None
        if num_workers > 1:
            logger.info("Creating Dask client...")
            client = DaskHelper(n_workers=args.n_workers)
            # Essential step:
            # More than speeding up data management by Dask, creating the benchmark class objects
            # once and sharing it across all the workers mean that for each task-seed instance,
            # the validation split remains the same across evaluations
            client.distribute_data_to_workers(benchmarks)
            logger.info("Dask Client information: {}".format(client.client))
    logger.info("Executing with {} worker(s)".format(num_workers))

    start = time.time()
    total_combinations = len(task_ids) * len(grid_config) * len(grid_fidelity) * args.n_seeds
    for i, combination in enumerate(
            itertools.product(*[task_ids, grid_config, grid_fidelity, seeds, [dump_path]]), start=1
    ):
        logger.info("{}/{}".format(i, total_combinations))
        logger.debug("Running for {:.2f} seconds".format(time.time() - start))
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
                # benchmarks should be provided as a second argument to compute() by dask as
                # the benchmarks are already distributed across the workers
                client.submit_job(compute, return_dict(combination))
                break
            else:
                client.fetch_futures(retries=1, wait_time=0.05)
    if num_workers > 1 and client.is_worker_alive():
        logger.info("Waiting for pending workers...")
        while num_workers > 1 and client.is_worker_alive():
            client.fetch_futures(retries=1, wait_time=0.05)
    end = time.time()

    logger.info(
        "{} unique configurations evaluated on {} different fidelity combinations for {} "
        "seeds on {} different tasks in {:.2f} seconds".format(
            len(grid_config), len(grid_fidelity), args.n_seeds, args.n_tasks, end - start
        )
    )
