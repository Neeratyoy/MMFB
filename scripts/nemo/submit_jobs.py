import os
import time
import argparse
import subprocess

from utils.util import all_task_ids_by_in_mem_size


def edit_submit_script(space, exp_type, task_id):
    with open("scripts/nemo/run_benchmark.sh", "r") as f:
        script = f.readlines()
    # chanding msub job name
    target = script[4].strip().split(" ")
    target[-1] = "{}-{}-{}\n".format(space, exp_type, str(task_id))
    script[4] = " ".join(target)
    # changing space argument
    script[11] = "space=\'{}\'\n".format(space)
    # changing exp type argument
    script[12] = "exp=\'{}\'\n".format(exp_type)
    # changing task ID argument
    script[13] = "taskid={}\n".format(task_id)
    with open("scripts/nemo/run_benchmark.sh", "w") as f:
        f.writelines(script)
    return


def input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=None, nargs="+", type=int)
    parser.add_argument("--space", default=None, type=str)
    parser.add_argument("--exp_type", default="full", type=str, choices=["full", "toy"])
    parser.add_argument("--codedir", default="$HOME'/Thesis/code/MMFB'", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = input_args()
    task_ids = all_task_ids_by_in_mem_size if args.tasks is None else args.tasks
    if args.space is None:
        raise ValueError("Enter a valid parameter space!")
    print(task_ids)
    script_path = os.path.join(args.codedir, "scripts/nemo/run_benchmark.sh")
    print(script_path)
    for task_id in task_ids:
        edit_submit_script(args.space, args.exp_type, task_id)
        subprocess.call(["msub", script_path])
        time.sleep(3)
