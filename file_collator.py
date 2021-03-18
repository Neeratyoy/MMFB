import os
import sys
import time
import pickle
import argparse
from loguru import logger


logger.configure(handlers=[{"sink": sys.stdout, "level": "INFO"}])
_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB"
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sleep", type=int, default=10, help="Sleep in seconds")
    parser.add_argument("--path", type=str, default="./tmp_dump", help="Directory for files")
    parser.add_argument("--max_batch_size", type=int, default=100000,
                        help="The number of files to process per loop iteration")
    args = parser.parse_args()

    sleep_wait = args.sleep
    path = args.path
    os.makedirs(path, exist_ok=True)
    os.makedirs("{}/logs".format(path), exist_ok=True)

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
        if not os.path.isdir(path):
            os.mkdir(path)

        # collect all files in the directory
        file_list = os.listdir(path)[:args.max_batch_size]
        logger.info("\tSnapshot taken from directory --> {} files found!".format(len(file_list)))
        logger.info("\tsleeping...")

        # sleep to allow disk writes to be completed for the collected file names
        time.sleep(sleep_wait)

        start = time.time()
        logger.info("\tStarting collection...")
        for i, filename in enumerate(file_list, start=1):
            logger.debug("\tProcessing {}/{}".format(i, len(file_list)), end='\r')
            # ignore files that are named as 'task_x.pkl or run_*.log'
            if (filename.split('_')[0] == "task" and filename.split('.')[-1] == "pkl") or \
                    (filename.split('_')[0] == "run" and filename.split('.')[-1] == "log") or \
                    os.path.isdir(os.path.join(path, filename)):
                continue

            try:
                with open(os.path.join(path, filename), "rb") as f:
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
                # if no data for this task_id seen yet, create file
                if not os.path.isfile(os.path.join(path, "task_{}.pkl".format(task_id))):
                    # create the file
                    with open(os.path.join(path, "task_{}.pkl".format(task_id)), 'wb') as f:
                        pickle.dump(obj, f)
                    continue

                with open(os.path.join(path, "task_{}.pkl".format(task_id)), 'rb') as f:
                    main_data = pickle.load(f)

                if config_hash not in main_data.keys():
                    main_data.update(obj)
                elif fidelity_hash not in main_data[config_hash].keys():
                    main_data[config_hash].update(obj[config_hash])
                elif seed not in main_data[config_hash][fidelity_hash].keys():
                    main_data[config_hash][fidelity_hash].update(obj[config_hash][fidelity_hash])
                else:
                    main_data[config_hash][fidelity_hash][seed].update(
                        obj[config_hash][fidelity_hash][seed]
                    )

                # updating file for the task_id
                with open(os.path.join(path, "task_{}.pkl".format(task_id)), 'wb') as f:
                    pickle.dump(main_data, f)
                file_count += 1

            try:
                os.remove(os.path.join(path, filename))  # deleting data file that was processed
            except FileNotFoundError:
                continue
        logger.info("\tFinished batch processing in {:.3f} seconds".format(time.time() - start))
        logger.info("\tContinuing to next batch")
        logger.info("\t{}".format("-" * 25))

    logger.info("Done!")
    logger.info("Total files processed: {}".format(file_count))
