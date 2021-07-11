import os
import sys
import time
import pickle
import argparse

import glom
import numpy as np
from copy import deepcopy
from loguru import logger
from collections import OrderedDict
from joblib.parallel import Parallel, parallel_backend, delayed

from utils.util import *


logger.configure(handlers=[{"sink": sys.stdout, "level": "INFO"}])
_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB"
}


def retrieve_task_ids(filenames):
    task_ids = []
    for i, filename in enumerate(filenames, start=1):
        try:
            task_id = int(filename.split("/")[-1].split("_")[0])
        except ValueError:
            continue
        task_ids.append(task_id)
    return np.unique(task_ids)


def update_table_with_new_entry(
        main_data: dict, new_entry: dict, config: dict, fidelity: dict
) -> dict:
    """ Updates the benchmark dict-hierarchy with a new function evaluation entry

    The storage is in a nested dict structure where the keys are arranged in the order of the
    configuration parameters ordered by their name, fidelity parameters ordered by their names
    and the seed. The final value element in the dict contains another dict returned by the actual
    function evaluations containing the result, cost, other misc. information.
    Given that the depth of this dict data will vary for different parameter space, the package
    `glom` is used. Wherein, the sequence of keys can be provided for easy retrieval, and
    assignment of values even for varying depth of a hierarchical dict.
    """
    seed = new_entry['info']['seed']
    key_nest = []
    for k, v in config.items():
        key_nest.append(np.float32(v))
        if glom.glom(main_data, glom.Path(*key_nest), default=None) is None:
            glom.assign(main_data, glom.Path(*key_nest), dict())
    for k, v in fidelity.items():
        key_nest.append(np.float32(v))
        if glom.glom(main_data, glom.Path(*key_nest), default=None) is None:
            glom.assign(main_data, glom.Path(*key_nest), dict())
    key_nest.append(seed)
    if glom.glom(main_data, glom.Path(*key_nest), default=None) is None:
        glom.assign(main_data, glom.Path(*key_nest), dict())
    glom.assign(main_data, glom.Path(*key_nest), new_entry)
    return main_data


def save_task_file(task_id, task_dict, path, config_spaces, exp_args):
    obj = task_dict
    task_dict['config_spaces'] = config_spaces
    task_dict['exp_args'] = exp_args.__dict__
    old_file = os.path.join(path, "task_{}_old.pkl".format(task_id))
    new_file = os.path.join(path, "task_{}_new.pkl".format(task_id))
    if os.path.isfile(new_file):
        # renaming the existing new file as old
        os.rename(new_file, old_file)
    else:
        # first iteration where a file is yet to be written
        pass
    # saving the new data as the new file
    with open(new_file, 'wb') as f:
        pickle.dump(obj, f)
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sleep", type=int, default=10, help="Sleep in seconds")
    parser.add_argument("--path", type=str, default="./tmp_dump",
                        help="Directory for files for a fidelity choice")
    parser.add_argument("--max_batch_size", type=int, default=100000,
                        help="The number of files to process per loop iteration")
    parser.add_argument("--config", type=str, default=None,
                        help="Full path to experiment config for which collator is running")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="The number of parallel cores for speeding up file writes per task")
    args = parser.parse_args()

    sleep_wait = args.sleep

    # Creating storage directories
    path = args.path
    base_path = os.path.join(path, "..")
    dump_path = os.path.join(path, "dump")
    output_path = os.path.join(path, "benchmark")
    os.makedirs(output_path, exist_ok=True)

    while not os.path.isfile(os.path.join(base_path, "param_space.pkl")) and \
            not os.path.isfile(os.path.join(path, "param_space.pkl")):
        time.sleep(sleep_wait)

    with open(os.path.join(base_path, "param_space.pkl"), "rb") as f:
        x_cs = pickle.load(f)
    with open(os.path.join(path, "fidelity_space.pkl"), "rb") as f:
        z_cs = pickle.load(f)
    exp_args = load_yaml_args(args.config)
    x_cs_discrete = get_discrete_configspace(x_cs, exp_args.x_grid_size)
    z_cs_discrete = get_discrete_configspace(z_cs, exp_args.z_grid_size)
    config_spaces = dict(
        x=x_cs, x_discrete=x_cs_discrete, z=z_cs, z_discrete=z_cs_discrete
    )

    # Logging details
    log_suffix = time.strftime("%x %X %Z")
    log_suffix = log_suffix.replace("/", '-').replace(":", '-').replace(" ", '_')
    logger.add(
        "{}/logs/collator_{}.log".format(path, log_suffix),
        **_logger_props
    )
    print("Logging at {}/logs/collator_{}.log".format(path, log_suffix))

    initial_file_list = os.listdir(path)
    file_count = 0
    task_datas = dict()
    processing_time = 0

    while True:
        #TODO: possibly empty out task_datas
        # list available tasks
        task_ids = [int(tid) for tid in os.listdir(dump_path)]
        if len(task_ids) == 0:
            continue
        batch_size = args.max_batch_size // len(task_ids)

        # collect all files in the directory
        file_list = []
        for tid in task_ids:
            _file_list = os.listdir(os.path.join(dump_path, str(tid)))[:batch_size]
            file_list.extend([os.path.join(str(tid), f) for f in _file_list])
        # shuffling allows progress of data collection across runs for all datasets available
        np.random.shuffle(file_list)
        logger.info("\tSnapshot taken from directory --> {} files found!".format(len(file_list)))
        logger.info("\tsleeping...")

        # sleep to allow disk writes to be completed for the collected file names
        sleep_time = sleep_wait - processing_time  # accounts for processing time in wait time
        time.sleep(sleep_wait if sleep_time < 0 else sleep_time)

        _task_ids = retrieve_task_ids(file_list)
        logger.info("\tTask IDs found: {}".format(_task_ids))

        for tid in _task_ids:
            #TODO: load task_datas if available on disk
            if tid not in task_datas.keys():
                task_datas[tid] = dict(progress=0, data=dict())

        start = time.time()
        logger.info("\tStarting collection...")
        for i, filename in enumerate(file_list, start=1):
            logger.debug("\tProcessing {}/{}".format(i, len(file_list)), end='\r')
            # ignore files that are named as 'task_*.pkl or run_*.log'
            if (filename.split('_')[0] == "task" and filename.split('.')[-1] == "pkl") or \
                    (filename.split('_')[0] == "run" and filename.split('.')[-1] == "log") or \
                    os.path.isdir(os.path.join(dump_path, filename)):
                continue

            try:
                with open(os.path.join(dump_path, filename), "rb") as f:
                    res = pickle.load(f)
            except FileNotFoundError:
                # if file was collected with os.listdir but deleted in the meanwhile, ignore it
                continue

            # OrderedDict ensure consistency when building the hierarchy of dicts as lookup table
            # since the hyperparameter and fidelity names don't change, ordering them ensures
            # both storage and lookup can match
            config = OrderedDict(res['info']['config'])
            fidelity = OrderedDict(res['info']['fidelity'])
            res['info'].pop('config')
            res['info'].pop('fidelity')
            task_id = int(filename.split('/')[0].split('_')[0])
            progress_id = int(filename.split('.pkl')[0].split('_')[-1])
            task_datas[task_id]['progress'] = max(task_datas[task_id]['progress'], progress_id)
            task_datas[task_id]['data'] = update_table_with_new_entry(
                task_datas[task_id]['data'], res, config, fidelity
            )
            file_count += 1

            try:
                # deleting data file that was processed
                os.remove(os.path.join(dump_path, filename))
            except FileNotFoundError:
                continue

        end = time.time()
        processing_time = end - start
        logger.info("\tFinished batch processing in {:.3f} seconds".format(processing_time))
        logger.info("\tUpdating benchmark data files...")

        with parallel_backend(backend="loky", n_jobs=args.n_jobs):
            Parallel()(
                delayed(save_task_file)(
                    task_id, obj, output_path, config_spaces, exp_args
                ) for task_id, obj in task_datas.items()
            )
        logger.info("\tContinuing to next batch")
        logger.info("\t{}".format("-" * 25))

    logger.info("Done!")
    logger.info("Total files processed: {}".format(file_count))
