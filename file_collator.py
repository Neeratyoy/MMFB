import os
import time
import json
import pickle
import numpy as np


start = time.time()

failed_removals = []
deleted_files = []

while True:

    path = 'tmp_dump/'
    if not os.path.isdir(path):
        os.mkdir(path)

    file_list = os.listdir(path)

    # all_results = {}
    # for filename in file_list:
    #     with open(os.path.join(path, filename), 'rb') as f:
    #         res = pickle.load(f)
    #     if filename.split('_')[0] != 'task':
    #         all_results.update(res)

    # wait for a minute after the snapshot of files in directory

    print("\nSnapshot taken from directory --> {} files found!".format(len(file_list)))
    time.sleep(10)

    previous_task_id = None
    
    print("Starting collection...")
    for i, filename in enumerate(file_list, start=1):
        print("{}/{}".format(i, len(file_list)), end='\r')
        if filename.split('_')[0] == 'task' or filename in failed_removals:
            continue

        with open(os.path.join(path, filename), 'rb') as f:
            res = pickle.load(f)

        for k, v in res.items():
            task_id, config_hash, fidelity_hash, seed = k
            obj = {
                task_id: {
                    config_hash: {
                        fidelity_hash: {
                            seed: v
                        }
                    }
                }
            }
            if not os.path.isfile(os.path.join(path, "task_{}.pkl".format(task_id))):
                # create the file
                with open(os.path.join(path, "task_{}.pkl".format(task_id)), 'wb') as f:
                    pickle.dump(obj, f)
                continue

            with open(os.path.join(path, "task_{}.pkl".format(task_id)), 'rb') as f:
                main_data = pickle.load(f)
            
            if config_hash not in main_data[task_id].keys():
                main_data[task_id].update(obj[task_id])
            elif fidelity_hash not in main_data[task_id][config_hash].keys():
                main_data[task_id][config_hash].update(obj[task_id][config_hash])
            elif seed not in main_data[task_id][config_hash][fidelity_hash].keys():
                main_data[task_id][config_hash][fidelity_hash].update(obj[task_id][config_hash][fidelity_hash])
            else:
                raise Exception("Collision!")

            with open(os.path.join(path, "task_{}.pkl".format(task_id)), 'wb') as f:
                pickle.dump(main_data, f)
        
        os.remove(os.path.join(path, filename))

    temp_file_list = os.listdir(path)
    if len(temp_file_list) !=0 and all([file_name.split('_')[0] == 'task' for file_name in temp_file_list]):
        break
            
print("Done!")
