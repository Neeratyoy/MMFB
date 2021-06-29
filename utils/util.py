import time
import yaml
import itertools
import numpy as np
import pandas as pd
import ConfigSpace as CS
from distributed import Client
from typing import Union, List, Tuple


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
    for i, hp in enumerate(cs.get_hyperparameters()):
        if isinstance(hp, CS.UniformIntegerHyperparameter):
            config[hp.name] = int(vector[i])
        elif isinstance(hp, CS.UniformFloatHyperparameter):
            config[hp.name] = float(vector[i])
        else:
            config[hp.name] = vector[i]
    return config


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
    for hp in cs.get_hyperparameters():
        if isinstance(hp, CS.CategoricalHyperparameter):
            param_ranges.append(hp.choices)
        elif isinstance(hp, CS.OrdinalHyperparameter):
            param_ranges.append(hp.sequences)
        elif isinstance(hp, CS.Constant):
            param_ranges.append([hp.value])
        else:
            if hp.log:
                param_ranges.append(
                    np.exp(np.linspace(
                        np.log(hp.lower), np.log(hp.upper), grid_step_size
                    ).astype(np.float32))
                )
            else:
                param_ranges.append(
                    np.linspace(hp.lower, hp.upper, grid_step_size).astype(np.float32)
                )
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
        seed: Union[int, None]=None
):
    """ Generates a new discretized ConfigurationSpace from a generally defined space

    Given the discretization grid size for each dimension, the new ConfigurationSpace contains
    each hyperparmater as an OrdinalParameter with the discretized values for that dimension as the
    sequence of choices available for that hyperparameter.

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
            self.n_workers = len(self.client.ncores())
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
        workers = len(self.client.ncores())
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
