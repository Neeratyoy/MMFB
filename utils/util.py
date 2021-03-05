import itertools
import numpy as np
import ConfigSpace as CS
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
        else:
            if hp.log:
                param_ranges.append(
                    np.exp(np.linspace(np.log(hp.lower), np.log(hp.upper), grid_step_size))
                )
            else:
                param_ranges.append(np.linspace(hp.lower, hp.upper, grid_step_size))
    full_grid = itertools.product(*param_ranges)
    if not convert_to_configspace:
        return list(full_grid)
    config_list = []
    for _config in full_grid:
        config_list.append(map_to_config(cs, _config))
    return config_list
