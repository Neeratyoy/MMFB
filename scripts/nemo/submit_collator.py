import os
import time
import argparse
import subprocess

from utils.util import all_task_ids_by_in_mem_size


def edit_submit_script(codedir, space, sleep=None):
    with open(os.path.join(codedir, "scripts/nemo/run_collator.sh"), "r") as f:
        script = f.readlines()
    # chanding msub job name
    target = script[4].strip().split(" ")
    target[-1] = "collator-{}\n".format(space)
    script[4] = " ".join(target)
    # changing space argument
    script[10] = "space=\'{}\'\n".format(space)
    # changing sleep argument
    if sleep is not None:
        script[11] = "sleep={}\n".format(sleep)
    # changing path argument
    script[13] = "path=$wspace\'/full/{}/1\'\n".format(space)
    with open(os.path.join(codedir, "scripts/nemo/run_collator.sh"), "w") as f:
        f.writelines(script)
    return


def input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", default=None, type=str)
    parser.add_argument("--sleep", default=None, type=int)
    parser.add_argument("--codedir", default="/home/fr/fr_fr/fr_nm217/Thesis/code/MMFB", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = input_args()

    script_path = os.path.join(args.codedir, "scripts/nemo/run_collator.sh")
    edit_submit_script(args.codedir, args.space, args.sleep)
    subprocess.call(["msub", script_path])
