import sys
import pickle
import hashlib
import warnings
import argparse
from typing import Dict
from loguru import logger

from hpobench.benchmarks.ml import *

from utils.util import *
from utils.util import all_task_ids_by_in_mem_size, read_openml_splits


logger.configure(handlers=[{"sink": sys.stdout, "level": "INFO"}])
_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB"
}

param_space_dict = dict(
    # 0 corresponds to the complete multi-multi-fidelity benchmark
    # 1 corresponds to the multi-fidelity benchmark using a single fidelity variable
    # 2 corresponds to the black-box benchmark using a no fidelities or full fidelities
    lr={0: LRBenchmark, 1: LRBenchmarkMF, 2: LRBenchmarkBB},
    rf={0: RandomForestBenchmark, 1: RandomForestBenchmarkMF, 2: RandomForestBenchmarkBB},
    nn={0: NNBenchmark, 1: NNBenchmarkMF, 2: NNBenchmarkBB},
    svm={0: SVMBenchmark, 1: SVMBenchmarkMF, 2: SVMBenchmarkBB},
    xgb={0: XGBoostBenchmark, 1: XGBoostBenchmarkMF, 2: XGBoostBenchmarkBB},
)


def config2hash(config):
    """ Generate md5 hash to distinguish and look up configurations
    """
    return hashlib.md5(repr(config).encode('utf-8')).hexdigest()


def return_dict(combination: Tuple) -> Dict:
    """ Converts a tuple to a dict that is recognizable by the function submitted to the workers
    """
    assert len(combination) == 11
    evaluation = dict()
    evaluation["task_id"] = combination[0]
    evaluation["config"] = combination[1]
    evaluation["fidelity"] = combination[2]
    evaluation["seed"] = combination[3]
    evaluation["path"] = combination[4]
    evaluation["data_path"] = combination[5]
    evaluation["record_train"] = combination[6]
    evaluation["record_lcs"] = combination[7]
    evaluation["lc_every_k"] = combination[8]
    evaluation["id"] = combination[9]
    evaluation["space"] = combination[10]
    return evaluation


def compute(evaluation: dict) -> str:
    """ Function to evaluate a configuration-fidelity on a task for a seed

    Parameters
    ----------
    evaluation : dict
        multi-element dictionary containing all information required to perform a run

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
    data_path = evaluation["data_path"]
    # When workers run on Nemo, each submitted job creates a tmp directory for that job as long as
    # the job is alive. This is an SSD disk which should allow fast reads of data splits. Moreover,
    # this directory being unique to the worker(s) running this function, prevents parallel access
    # of the data splits and should be faster. In case, this directory doesn't exist, the default
    # data_path passed will be used. Therefore, the check that openml_splits/[task_id] exists
    # under the tmp directory is important to ensure no failures.
    if "TMPDIR" in os.environ:
        tmp_path = os.path.join(os.environ["TMPDIR"], "openml_splits")
        if os.path.isdir(tmp_path):
            data_path = tmp_path
    print("Data path: ", data_path)
    record_train = evaluation["record_train"]
    record_lcs = evaluation["record_lcs"]
    lc_every_k = evaluation["lc_every_k"]
    i = evaluation["id"]
    model_space = evaluation["space"]
    task_path = os.path.join(path, str(task_id))
    os.makedirs(task_path, exist_ok=True)

    start = time.time()
    benchmark = model_space(
        task_id=task_id,
        rng=seed,
        data_path=data_path
    )
    # the lookup dict key for each evaluation is a 4-element tuple
    end1 = time.time()
    print("Time to load: {:.5f}".format(end1 - start))
    # setting `rng=None` to the objective call ensures the random state passed during the benchmark
    # instantiation is used during the ML model instantiation too
    result = benchmark.objective_function(
        config, fidelity, rng=None, record_train=record_train,
        get_learning_curve=record_lcs, lc_every_k=lc_every_k
    )
    print("Time to evaluate: {:.5f}".format(time.time() - end1))
    result['info']['seed'] = seed
    # file_collator should collect the pickle files dumped below
    name = "{}/{}_{}_{}_{}_{}.pkl".format(task_path, task_id, config_hash, fidelity_hash, seed, i)
    with open(name, 'wb') as f:
        pickle.dump(result, f)
    print("Saved at {}".format(task_path))
    return "success"


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
        choices=list(param_space_dict.keys()),
        help="The number of tasks to run data collection on from the AutoML benchmark suite"
    )
    parser.add_argument(
        "--data_path",
        default="./tmp_dump/openml_splits/",
        type=str,
        help="The directory to load data splits from"
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
        "--include_SH",
        default=False,
        action="store_true",
        help="Includes geometric budget spacing from Successive Halving"
    )
    parser.add_argument(
        "--fidelity_choice",
        default=0,
        type=int,
        help="Choice of fidelity space:\n"
             "0 : the multi-multi-fidelity version\n"
             "1 : the multi-fidelity version using only one fidelity variable\n"
             "2 : the black-box version\n"
    )
    parser.add_argument(
        "--record_train",
        default=False,
        action="store_true",
        help="If True, records the evaluation statistics on the training set."
    )
    parser.add_argument(
        "--record_lcs",
        default=False,
        action="store_true",
        help="If True, records the learning curves on the validation and test sets."
    )
    parser.add_argument(
        "--lc_every_k",
        default=1,
        type=int,
        help="The number of iterations after which evaluations will be recorded for LCs."
    )
    parser.add_argument(
        "--n_seeds",
        default=5,
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
        help="The seed for the complete benchmark collection runs"
    )
    parser.add_argument(
        "--exp_name",
        default=None,
        type=str,
        help="Creates an experiment directory and if script executed with cmd arguments, "
             "dumps an yaml file with the arguments at the same level"
    )
    parser.add_argument(
        "--restart_id",
        default=0,
        type=int,
        help="Runs and computes the experiment from `restart_id` onwards"
    )
    parser.add_argument(
        "--missing",
        default=None,
        type=str,
        help="File of missing evaluations"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = input_arguments()

    exp_name = None
    if args.config is not None and os.path.isfile(args.config):
        # if config file provided, load args from it and ignore other cmd args
        args = load_yaml_args(args.config)
    if args.config is None and args.exp_name is not None:
        # if config file not provided and exp_name provided, dump current cmd args under exp_name
        exp_name = args.exp_name
        args.output_path = os.path.join(args.output_path, args.exp_name)

    # Choosing parameter space
    param_space = param_space_dict[args.space][args.fidelity_choice]

    # Task input check
    task_ids = all_task_ids_by_in_mem_size
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

    # Creating storage directories
    base_path = os.path.join(os.getcwd(), args.output_path, args.space)
    os.makedirs(base_path, exist_ok=True)
    if exp_name is not None:
        filename = os.path.join(base_path, "{}_{}_args.yaml".format(args.exp_name, args.space))
        dump_yaml_args(args.__dict__, filename)
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

    # Setting seed
    np.random.seed(args.seed)
    # Selecting seeds
    seeds = np.random.randint(1, 10000, size=args.n_seeds)
    logger.info("Seeds selected {}".format(seeds))

    # Placeholder benchmark to retrieve parameter spaces
    benchmark = param_space(
        task_id=task_ids[0],
        rng=seeds[0],
        data_path=args.data_path
    )

    # Saving a copy of the ConfigSpaces used for this run
    with open(os.path.join(base_path, "param_space.pkl"), "wb") as f:
        pickle.dump(benchmark.configuration_space, f)  # hyper-parameter configuration space
    with open(os.path.join(path, "fidelity_space.pkl"), "wb") as f:
        pickle.dump(benchmark.fidelity_space, f)  # fidelity configuration space

    # Retrieving observation space and populating grid
    configuration_space = benchmark.configuration_space
    logger.info("Populating grid for observation space...")
    grid_config = get_parameter_grid(
        configuration_space, args.x_grid_size, convert_to_configspace=False
    )
    logger.info("{} unique observations generated".format(len(grid_config)))
    logger.debug("Observation space grid size: {:.2f} MB".format(obj_size(grid_config)))

    # Retrieving fidelity spaces and populating grid
    fidelity_space = benchmark.fidelity_space
    logger.info("Populating grid for fidelity space...")
    grid_fidelity = get_fidelity_grid(
        fidelity_space, args.z_grid_size,
        convert_to_configspace=False, include_sh_budgets=args.include_SH
    )
    logger.info("{} unique fidelity configurations generated".format(len(grid_fidelity)))
    logger.debug("Fidelity space grid size: {:.2f} MB".format(obj_size(grid_fidelity)))

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
    #         # Essential step:
    #         # More than speeding up data management by Dask, creating the benchmark class objects
    #         # once and sharing it across all the workers mean that for each task-seed instance,
    #         # the validation split remains the same across evaluations
    #         # client.distribute_data_to_workers(benchmarks)
            logger.info("Dask Client information: {}".format(client.client))
    logger.info("Executing with {} worker(s)".format(num_workers))

    # Loading file of index of missing evaluations to collect only those evaluations
    missing = []
    if args.missing is not None and os.path.isfile(args.missing) and len(task_ids) == 1:
        with open(args.missing, "r") as f:
            missing = f.readlines()
        missing = [int(id.strip()) for id in missing]

    start = time.time()
    total_combinations = len(task_ids) * len(grid_config) * len(grid_fidelity) * args.n_seeds
    for i, combination in enumerate(
            itertools.product(*[task_ids, grid_config, grid_fidelity, seeds, [dump_path]]), start=1
    ):
        logger.info("{}/{}".format(i, total_combinations))
        logger.debug("Running for {:.2f} seconds".format(time.time() - start))
        # if missing indexes exist and current index i is not a part of it, assume it is collected
        if len(missing) and i not in missing:
            continue
        if i < args.restart_id:
            logger.debug("Skipping and continuing evaluation..")
            continue
        combination = list(combination)
        combination[1] = map_to_config(configuration_space, combination[1])
        combination[2] = map_to_config(fidelity_space, combination[2])
        combination.append(args.data_path)
        combination.append(args.record_train)
        combination.append(args.record_lcs)
        combination.append(args.lc_every_k)
        combination.append(i)
        combination.append(param_space)
        if num_workers == 1:
            compute(return_dict(combination))
            continue
        # for a combination selected, need to wait until it is submitted to a worker
        # client.submit_job() is an asynchronous call, followed by a break which allows the
        # next combination to be submitted if client.is_worker_available() is True
        while True:
            # returns a list of available workers and their addresses
            workers = client.is_worker_available()
            if workers:
                logger.debug("Available worker count: {}".format(len(workers)))
                client.submit_job(compute, return_dict(combination), workers)
                # sleep allows a job to be submitted to a worker such that the next round of
                # available worker counts is a closer approximation to the actual availability
                time.sleep(0.05)  # 50 milliseconds
                break
            else:
                time.sleep(0.1)  # wait for 100 milliseconds before querying for futures
                client.fetch_futures(retries=1, wait_time=0.0)
    if num_workers > 1 and client.is_worker_alive():
        logger.info("Waiting for pending workers...")
        while num_workers > 1 and client.is_worker_alive():
            client.fetch_futures(retries=1, wait_time=5)  # 5 seconds
    end = time.time()

    logger.info(
        "{} unique configurations submitted on {} different fidelity combinations for {} "
        "seeds on {} different tasks in {:.2f} seconds".format(
            len(grid_config), len(grid_fidelity), args.n_seeds, args.n_tasks, end - start
        )
    )
