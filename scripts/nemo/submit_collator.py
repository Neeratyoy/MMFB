"""
Script that edits run_collator.sh dynamically to have relevant parameters and job names
"""

import os
import argparse
import subprocess


def edit_submit_script(codedir, space, fidelity=1, sleep=None):
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
    script[13] = "path=$wspace\'/full/{}/{}\'\n".format(space, fidelity)
    with open(os.path.join(codedir, "scripts/nemo/run_collator.sh"), "w") as f:
        f.writelines(script)
    return


def input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", default=None, type=str)
    parser.add_argument("--fidelity", default=1, type=int, choices=[0, 1, 2, 3])
    parser.add_argument("--sleep", default=None, type=int)
    parser.add_argument("--codedir", default="/home/fr/fr_fr/fr_nm1068/Thesis/code/MMFB", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = input_args()

    script_path = os.path.join(args.codedir, "scripts/nemo/run_collator.sh")
    edit_submit_script(args.codedir, args.space, args.fidelity, args.sleep)
    subprocess.call(["msub", script_path])
