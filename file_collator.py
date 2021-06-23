import os
import sys
import time
import pickle
import argparse
import numpy as np
from copy import deepcopy
from loguru import logger
from joblib.parallel import Parallel, parallel_backend, delayed


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


def update_table_with_new_entry(main_data, new_entry):
    config_hash = list(new_entry.keys())[0]
    fidelity_hash = list(new_entry[config_hash].keys())[0]
    seed = list(new_entry[config_hash][fidelity_hash].keys())[0]
    if config_hash not in main_data.keys():
        main_data.update(new_entry)
    elif fidelity_hash not in main_data[config_hash].keys():
        main_data[config_hash].update(new_entry[config_hash])
    elif seed not in main_data[config_hash][fidelity_hash].keys():
        main_data[config_hash][fidelity_hash].update(new_entry[config_hash][fidelity_hash])
    else:
        main_data[config_hash][fidelity_hash][seed].update(
            new_entry[config_hash][fidelity_hash][seed]
        )
    return main_data


def save_task_file(task_id, task_dict, path):
    obj = task_dict
    if os.path.isfile(os.path.join(path, "task_{}.pkl".format(task_id))):
        # if file exists, read the file, append, write
        with open(os.path.join(path, "task_{}.pkl".format(task_id)), 'rb') as f:
            data = pickle.load(f)
        # with time `data` will be larger in size than `obj`
        for _config in obj.keys():
            for _fidelity in obj[_config].keys():
                for _seed in obj[_config][_fidelity].keys():
                    _data = {_config: {_fidelity: {_seed: obj[_config][_fidelity][_seed]}}}
                    data = update_table_with_new_entry(data, _data)
        obj = data
    with open(os.path.join(path, "task_{}.pkl".format(task_id)), 'wb') as f:
        pickle.dump(obj, f)
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sleep", type=int, default=10, help="Sleep in seconds")
    parser.add_argument("--path", type=str, default="./tmp_dump", help="Directory for files")
    parser.add_argument("--max_batch_size", type=int, default=100000,
                        help="The number of files to process per loop iteration")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="The number of parallel cores for speeding up file writes per task")
    args = parser.parse_args()

    sleep_wait = args.sleep
    path = args.path
    os.makedirs(path, exist_ok=True)
    os.makedirs("{}/logs".format(path), exist_ok=True)
    dump_path = os.path.join(path, "dump")
    os.makedirs(dump_path, exist_ok=True)
    output_path = os.path.join(path, "benchmark")
    os.makedirs(output_path, exist_ok=True)

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

    while True:
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
        time.sleep(sleep_wait)

        _task_ids = retrieve_task_ids(file_list)
        logger.info("\tTask IDs found: {}".format(_task_ids))
        task_datas = dict()
        for tid in _task_ids:
            task_datas[tid] = dict()

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

            for k, v in res.items():
                task_id, config_hash, fidelity_hash, seed = k
                # structure of how records are saved to disk to allow quick lookups
                # each dict object is associated to a fixed task_id
                # each such task will have a list of configurations stored with their md5 hash
                # each config under each task will have all the fidelities it was evaluated on
                # each task-config-fidelity would have been evaluated on a different seed
                obj = {
                    config_hash: {
                        fidelity_hash: {
                            seed: v
                        }
                    }
                }
                task_datas[task_id] = update_table_with_new_entry(task_datas[task_id], obj)
                file_count += 1

            try:
                # deleting data file that was processed
                os.remove(os.path.join(dump_path, filename))
                # os.remove(os.path.join(dump_path, filename))
            except FileNotFoundError:
                continue

        logger.info("\tFinished batch processing in {:.3f} seconds".format(time.time() - start))
        logger.info("\tUpdating benchmark data files...")

        with parallel_backend(backend="loky", n_jobs=args.n_jobs):
            Parallel()(
                delayed(save_task_file)(task_id, obj, output_path) for task_id, obj in task_datas.items()
            )
        logger.info("\tContinuing to next batch")
        logger.info("\t{}".format("-" * 25))

    logger.info("Done!")
    logger.info("Total files processed: {}".format(file_count))
