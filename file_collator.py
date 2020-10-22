import os
import time
import pickle


while True:

    path = 'tmp_dump/'
    if not os.path.isdir(path):
        os.mkdir(path)

    # collect all files in the directoe
    file_list = os.listdir(path)

    print("\nSnapshot taken from directory --> {} files found!".format(len(file_list)))
    time.sleep(10)

    previous_task_id = None
    
    print("Starting collection...")
    for i, filename in enumerate(file_list, start=1):
        print("{}/{}".format(i, len(file_list)), end='\r')
        # ignore files that are named as 'task_x.pkl'
        if filename.split('_')[0] == 'task':
            continue

        with open(os.path.join(path, filename), 'rb') as f:
            res = pickle.load(f)

        for k, v in res.items():
            task_id, config_hash, fidelity_hash, seed = k
            # structure of how records are saved to disk to allow quick lookups
            # the parent key is given by the task_id
            # each task will have a list of configurations stored with their md5 hash
            # each config under each task will have all the fidelities it was evaluated on
            # each task-config-fidelity would have been evaluated on a different seed
            obj = {
                task_id: {
                    config_hash: {
                        fidelity_hash: {
                            seed: v
                        }
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

            # find the right level in the dict hierarchy to append new data
            if config_hash not in main_data[task_id].keys():
                main_data[task_id].update(obj[task_id])
            elif fidelity_hash not in main_data[task_id][config_hash].keys():
                main_data[task_id][config_hash].update(obj[task_id][config_hash])
            elif seed not in main_data[task_id][config_hash][fidelity_hash].keys():
                main_data[task_id][config_hash][fidelity_hash].update(obj[task_id][config_hash][fidelity_hash])
            else:
                raise Exception("Collision!")

            # updating file for the task_id
            with open(os.path.join(path, "task_{}.pkl".format(task_id)), 'wb') as f:
                pickle.dump(main_data, f)
        
        os.remove(os.path.join(path, filename))  # deleting file that was appended to task data

    temp_file_list = os.listdir(path)  # current snapshot of directory
    # break/terminate file collection if only files remaining in the directory are task data files
    if len(temp_file_list) !=0 and all([file_name.split('_')[0] == 'task' for file_name in temp_file_list]):
        break
            
print("Done!")
