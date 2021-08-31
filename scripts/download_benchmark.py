import os
import glob
import argparse
import subprocess


tabular_multi_fidelity_urls = dict(
    xgb="https://ndownloader.figshare.com/files/30469920",
    svm="https://ndownloader.figshare.com/files/30379359",
    lr="https://ndownloader.figshare.com/files/30379038",
    rf="https://ndownloader.figshare.com/files/30469089",
    nn="https://ndownloader.figshare.com/files/30379005"
)
models = list(tabular_multi_fidelity_urls.keys())


def input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", default="/work/dlclarge1/mallik-hpobench/scripts/download_benchmark.sh", type=str)
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--path", default="/work/dlclarge1/mallik-hpobench/DataDir/TabularData/", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = input_args()
    for model in models:
        print("\nRunning bash {} {} {} {}\n\n".format(args.script, tabular_multi_fidelity_urls[model], args.path, model))
        subprocess.call(["bash", args.script, tabular_multi_fidelity_urls[model], args.path, model])
    zips = glob.glob(os.path.join(args.path, "/*.zip"))
    for zip in zips:
        os.remove(zip)
    print("\nDone!")
