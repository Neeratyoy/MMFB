import os
import time
import yaml
import openml
import itertools
import numpy as np
import pandas as pd
import ConfigSpace as CS
from distributed import Client
from pympler.asizeof import asizeof
from typing import Union, List, Tuple


all_task_ids = [
    3, 12, 31, 53, 3917, 3945, 7592, 7593, 9952, 9977, 9981, 10101, 14965, 34539, 146195, 146212,
    146606, 146818, 146821, 146822, 146825, 167119, 167120, 168329, 168330, 168331,
    168332,  # took long
    168335, 168337, 168338, 168868, 168908, 168909, 168910, 168911, 168912, 189354, 189355, 189356
]

all_task_ids_by_disk_size = [
    146821, 10101, 146818, 31, 53, 3, 9952, 146822, 3917, 168912, 168911, 34539, 167119, 7592,
    14965, 12, 146195, 146212, 9981,  # < 40 MB
    168329, 167120, 189354, 146606, 9977,  # 40-100 MB
    168330, 168335, 168910, 168908, 7593, 168331, 3945, 168868, 168909,  # 100-500 MB
    189355, 189356,  # >500 MB
    146825, 168332, 168337, 168338  # >1000 MB
]

all_task_ids_by_in_mem_size = [
    10101, 53, 146818, 146821, 9952, 146822, 31, 3917, 168912, 3, 167119, 12, 146212, 168911,
    9981, 168329, 167120, 14965, 146606,  # < 30 MB
    168330, 7592, 9977, 168910, 168335, 146195, 168908, 168331,  # 30-100 MB
    168868, 168909, 189355, 146825, 7593,  # 100-500 MB
    168332, 168337, 168338,  # > 500 MB
    189354, 34539,  # > 2.5k MB
    3945,  # >20k MB
    # 189356  # MemoryError: Unable to allocate 1.50 TiB; array size (256419, 802853) of type float64
]


def obj_size(obj):
    """ Returns size in MB
    """
    return asizeof(obj) / 1024 ** 2


def profile_tasks(n=40):
    """ Reads all_task_ids to generate task meta data and sort task IDs by dataset size on disk
    """
    selected_qualities = [
        "NumberOfInstances", "NumberOfFeatures", "NumberOfClasses", "PercentageOfNumericFeatures",
        "PercentageOfSymbolicFeatures", "Size(in MB)"
    ]
    meta_d = dict()
    for task_id in all_task_ids[:n]:
        print(task_id)
        task = openml.tasks.get_task(task_id, download_data=False)
        # fetches dataset
        dataset = openml.datasets.get_dataset(task.dataset_id, download_data=False)
        X, y, categorical_ind, feature_names = dataset.get_data(
            target=task.target_name, dataset_format="dataframe"
        )
        size = (asizeof(X) + asizeof(y)) / (1024 ** 2)
        qualities = {k: v for k, v in dataset.qualities.items() if k in selected_qualities}
        qualities[selected_qualities[-1]] = size
        meta_d[task_id] = qualities
    profile = pd.DataFrame.from_dict(meta_d, orient="index").sort_values(by="Size(in MB)")
    return profile


def profile_benchmarks(n=40, model=None):
    """ Reads all_task_ids_by_disk_size to sort task IDs by dataset size in memory post-transform
    """
    skip_tasks = [189356]
    if model is None:
        raise ValueError("Pass a benchmark class name (example: SVMBenchmark) under `model`!")
    meta_d = dict()
    for task_id in all_task_ids_by_disk_size[:n]:
        if task_id in skip_tasks:
            continue
        print(task_id)
        benchmark = model(task_id=task_id)
        meta_d[task_id] = obj_size(benchmark.train_X) + obj_size(benchmark.train_y) + \
                          obj_size(benchmark.valid_X) + obj_size(benchmark.valid_y) + \
                          obj_size(benchmark.test_X) + obj_size(benchmark.test_y)
    profile = pd.DataFrame.from_dict(meta_d, orient="index", columns=["Size(in MB)"])
    profile = profile.sort_values(by="Size(in MB)")
    return profile


def map_to_config(cs: CS.ConfigurationSpace, vector: Union[np.array, List]) -> CS.Configuration:
    """Return a ConfigSpace object with values assigned from the vector

    Parameters
    ----------
    cs : ConfigSpace.ConfigurationSpace
    vector : np.array or list

    Returns
    -------
    ConfigSpace.Configuration
    """
    config = cs.sample_configuration()
    for i, name in enumerate(np.sort(cs.get_hyperparameter_names())):
        hp = cs.get_hyperparameter(str(name))
        if isinstance(hp, CS.UniformIntegerHyperparameter):
            config[hp.name] = int(vector[i])
        elif isinstance(hp, CS.UniformFloatHyperparameter):
            # clip introduced to handle the edge case when a 64-bit float type casting introduces
            # extra precision which can make the number lower than the hard hp.lower limit
            config[hp.name] = np.clip(float(vector[i]), hp.lower, hp.upper)
        else:
            config[hp.name] = vector[i]
    return config


def generate_SH_fidelities(
        min_budget: Union[int, float], max_budget: Union[int, float], eta: int = 3
) -> np.ndarray:
    """ Creates a geometric progression of budgets/fidelities based on Successive Halving
    """
    max_SH_iter = -int(np.log(min_budget / max_budget) / np.log(eta)) + 1
    budgets = max_budget * np.power(eta, -np.linspace(start=max_SH_iter-1, stop=0, num=max_SH_iter))
    return budgets


def get_parameter_grid(
        cs: CS.ConfigurationSpace,
        grid_step_size: int = 10,
        convert_to_configspace: bool = False
) -> Union[List[Tuple], List[CS.Configuration]]:
    """Generates a grid from cartesian product of the parameters spaced out at given step size

    Parameters
    ----------
    cs : ConfigSpace.ConfigurationSpace
    grid_step_size : int
        The number of steps to divide a parameter dimension into
    convert_to_configspace : bool
        If True, returns a list of ConfigSpace objects of each point in the grid
        If False, returns a list of tuples containing the values of each point in grid

    Returns
    -------
    list
    """
    param_ranges = []
    for name in np.sort(cs.get_hyperparameter_names()):
        hp = cs.get_hyperparameter(str(name))
        if isinstance(hp, CS.CategoricalHyperparameter):
            param_ranges.append(hp.choices)
        elif isinstance(hp, CS.OrdinalHyperparameter):
            param_ranges.append(hp.sequence)
        elif isinstance(hp, CS.Constant):
            param_ranges.append([hp.value])
        else:
            if hp.log:
                grid = np.exp(np.linspace(
                    np.log(hp.lower), np.log(hp.upper), grid_step_size
                ))
                grid = np.clip(grid, hp.lower, hp.upper).astype(np.float32)
            else:
                grid = np.linspace(hp.lower, hp.upper, grid_step_size).astype(np.float32)
            grid = grid.astype(int) if isinstance(hp, CS.UniformIntegerHyperparameter) else grid
            param_ranges.append(grid)
    full_grid = itertools.product(*param_ranges)
    if not convert_to_configspace:
        return list(full_grid)
    config_list = []
    for _config in full_grid:
        config_list.append(map_to_config(cs, _config))
    return config_list


def get_fidelity_grid(
        cs: CS.ConfigurationSpace,
        grid_step_size: int = 10,
        convert_to_configspace: bool = False,
        include_sh_budgets: bool = True
) -> Union[List[Tuple], List[CS.Configuration]]:
    """Generates a grid from cartesian product of the fidelity spaced out at given step size

    Parameters
    ----------
    cs : ConfigSpace.ConfigurationSpace
    grid_step_size : int
        The number of steps to divide a parameter dimension into
    convert_to_configspace : bool
        If True, returns a list of ConfigSpace objects of each point in the grid
        If False, returns a list of tuples containing the values of each point in grid
    include_sh_budgets : bool
        If True, additionally includes budget spacing from Hyperband for eta={2,3,4}

    Returns
    -------
    list
    """
    param_ranges = []
    for name in np.sort(cs.get_hyperparameter_names()):
        hp = cs.get_hyperparameter(str(name))
        if isinstance(hp, CS.Constant):
            param_ranges.append([hp.value])
        else:
            if hp.log:
                grid = np.exp(np.linspace(
                    np.log(hp.lower), np.log(hp.upper), grid_step_size
                ))
                grid = np.clip(grid, hp.lower, hp.upper).astype(np.float32)
            else:
                grid = np.linspace(hp.lower, hp.upper, grid_step_size).astype(np.float32)
            if include_sh_budgets:
                hb_grid = np.array([])
                for eta in [2, 3, 4]:
                    hb_grid = np.concatenate(
                        (hb_grid, generate_SH_fidelities(hp.lower, hp.upper, eta))
                    ).astype(np.float32)
                grid = np.unique(np.concatenate((hb_grid, grid)))
            grid = grid.astype(int) if isinstance(hp, CS.UniformIntegerHyperparameter) else grid
            param_ranges.append(np.unique(grid))
    full_grid = itertools.product(*param_ranges)
    if not convert_to_configspace:
        return list(full_grid)
    config_list = []
    for _config in full_grid:
        config_list.append(map_to_config(cs, _config))
    return config_list


def get_discrete_configspace(
        configspace: CS.ConfigurationSpace,
        grid_size: int=10,
        seed: Union[int, None] = None,
        fidelity_space: bool = False
):
    """ Generates a new discretized ConfigurationSpace from a generally defined space

    Given the discretization grid size for each dimension, the new ConfigurationSpace contains
    each hyperparmater as an OrdinalParameter with the discretized values for that dimension as
    the sequence of choices available for that hyperparameter.

    Parameters
    ----------
    configspace : ConfigSpace.ConfigurationSpace
    grid_size : int
        The number of steps to divide a parameter dimension into
    seed : int

    Returns
    -------
    ConfigSpace.ConfigurationSpace
    """
    if fidelity_space:
        grid_list = pd.DataFrame(get_fidelity_grid(configspace, grid_size))
    else:
        grid_list = pd.DataFrame(get_parameter_grid(configspace, grid_size))
    cs = CS.ConfigurationSpace(seed=seed)
    hp_names = np.sort(configspace.get_hyperparameter_names()).tolist()
    for i, k in enumerate(hp_names):
        choices = grid_list.iloc[:, i].unique()
        if isinstance(configspace.get_hyperparameter(k), CS.UniformIntegerHyperparameter):
            choices = choices.astype(int)
        elif isinstance(configspace.get_hyperparameter(k), CS.UniformFloatHyperparameter):
            choices = choices.astype(np.float32)
        cs.add_hyperparameter(CS.OrdinalHyperparameter(str(k), choices))
    return cs


def load_yaml_args(filename):
    with open(filename, "r") as f:
        # https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation
        args = yaml.load(f, Loader=yaml.FullLoader)
    return DotDict(args)


def dump_yaml_args(args, filename):
    with open(filename, "w") as f:
        f.writelines(yaml.dump(args))
    return


def arg_yaml_all_task(space, dest="nemo", exp_type="toy"):
    args = load_yaml_args(os.path.join("arguments", dest, exp_type, "{}_args.yaml".format(space)))
    args.n_tasks = None
    args.exp_name = exp_type
    for task_id in all_task_ids:
        args.task_id = task_id
        filename = os.path.join("arguments", dest, exp_type, space, "args_{}.yaml".format(task_id))
        dump_yaml_args(dict(args), filename)
    return


class DotDict(dict):
    """dot.notation access to dictionary attributes

    sourced from: https://stackoverflow.com/a/23689767/8363967
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class DaskHelper:
    """ Manages Dask client and provides associated helper functions.
    """
    def __init__(self, n_workers=1, client=None):
        if client is not None and isinstance(client, Client):
            self.client = client
            self.n_workers = self._get_n_workers()
        else:
            self.n_workers = n_workers
            self.client = Client(
                n_workers=self.n_workers,
                processes=True,
                threads_per_worker=1,
                scheduler_port=0
            )
        self.futures = []
        self.shared_data = None
        self.worker_list = None

    def _get_n_workers(self):
        self.n_workers = len(self.client.ncores())
        return self.n_workers

    def _get_worker_list(self):
        worker_list = list(self.client.ncores().keys())
        return worker_list

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
        workers = self._get_n_workers()
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
        if self.worker_list is None:
            self.worker_list = self._get_worker_list()
            self.shared_data = self.client.scatter(data, broadcast=True)
        current_worker_list = list(self.client.ncores().keys())
        if len(set(current_worker_list) - set(self.worker_list)) > 0:
            # redistribute data across workers only when new workers have been added
            self.shared_data = self.client.scatter(data, broadcast=True)
        self.worker_list = self._get_worker_list()

    def is_worker_alive(self):
        return len(self.futures) > 0
