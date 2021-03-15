import os
import sys
import time
import pickle
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sleep", type=int, default=10, help="Sleep in seconds")
    parser.add_argument("--path", type=str, default="./tmp_dump", help="Directory for files")
    parser.add_argument("--max_batch_size", type=int, default=100000,
                        help="The number of files to process per loop iteration")
    args = parser.parse_args()

    sleep_wait = args.sleep
    path = args.path

    initial_file_list = os.listdir(path)

    while True:
        if not os.path.isdir(path):
            os.mkdir(path)

        # collect all files in the directory
        file_list = os.listdir(path)[:args.max_batch_size]
        print("|-- Snapshot taken from directory --> {} files found!".format(len(file_list)))
        print("|...sleeping...")

        # sleep to allow disk writes to be completed for the collected file names
        time.sleep(sleep_wait)

        print("|-- Starting collection")
        for i, filename in enumerate(file_list, start=1):
            print("|-- Processing {}/{}".format(i, len(file_list)), end='\r')
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

            try:
                os.remove(os.path.join(path, filename))  # deleting data file that was processed
            except FileNotFoundError:
                continue
        print("\n|-- Continuing")
        print("|{}".format("-" * 25))

    print("Done!")
