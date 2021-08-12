import os
import time
import argparse
import subprocess

from utils.util import all_task_ids_by_in_mem_size, load_yaml_args, dump_yaml_args


def edit_submit_script(codedir, space, exp_type, task_id):
    with open(os.path.join(codedir, "scripts/nemo/run_benchmark.sh"), "r") as f:
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
    with open(os.path.join(codedir, "scripts/nemo/run_benchmark.sh"), "w") as f:
        f.writelines(script)
    return


def edit_args_scheduler(codedir, space, task_id, scheduler):
    base_path = "arguments/nemo/full/{}/".format(space)
    config = load_yaml_args(os.path.join(codedir, base_path, "args_{}.yaml".format(task_id)))
    _sch = config.scheduler_file.split('/')
    _sch[-1] = "{}.json".format(scheduler)
    config.scheduler_file = "/".join(_sch)
    dump_yaml_args(dict(config), os.path.join(codedir, base_path, "args_{}.yaml".format(task_id)))
    return


def input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=None, nargs="+", type=int)
    parser.add_argument("--space", default=None, type=str)
    parser.add_argument("--exp_type", default="full", type=str, choices=["full", "toy"])
    parser.add_argument("--codedir", default="/home/fr/fr_fr/fr_nm217/Thesis/code/MMFB", type=str)
    parser.add_argument("--sleep", default=5, type=float)
    parser.add_argument("--scheduler", default=None, type=str, help="scheduler file name")
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
        edit_submit_script(args.codedir, args.space, args.exp_type, task_id)
        if args.scheduler is not None:
            edit_args_scheduler(args.codedir, args.space, task_id, args.scheduler)
        subprocess.call(["msub", script_path])
        time.sleep(args.sleep)
