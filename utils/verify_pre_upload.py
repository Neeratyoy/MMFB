import os
import argparse
from glob import glob
from zipfile import ZipFile

from hpobench.benchmarks.ml import TabularBenchmark

from run_benchmark_dask import param_space_dict


def input_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--path",
        default=None,
        type=str,
        help="full path to benchmark file"
    )
    parser.add_argument(
        "--model",
        default=None,
        type=str,
        choices=list(param_space_dict.keys()),
        help="model to test"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = input_arguments()

    assert os.path.isdir(args.path), "Not a valid path!"
    path_list = glob(os.path.join(args.path, "*/"))

    print("{} directories found for processing...".format(len(path_list)))

    passes = []
    fails = []
    for i, path in enumerate(path_list, start=1):
        print("Verifying {:>2}/{:>2}: {}".format(i, len(path_list), path))
        task_id = int(path.split("/")[-2])
        try:
            data_dir = os.path.join(*path.split("/")[:-3])
            benchmark = TabularBenchmark(data_dir=data_dir, model=args.model, task_id=task_id)
            config = benchmark.configuration_space.sample_configuration()
            fidelity = benchmark.fidelity_space.sample_configuration()
            res = benchmark.objective_function(config, fidelity)
            res = benchmark.objective_function_test(config, fidelity)
            print("PASSED!")
            passes.append(task_id)
        except Exception as e:
            fails.append(task_id)
            print("FAILED: {}".format(repr(e)))
    print()
    if len(fails) == 0:
        print("All files successully verified! Beginning zip archiving of all verified files...")
        with ZipFile(os.path.join(args.path, "{}.zip".format(args.model)), "w") as zip_obj:
            for i, path in enumerate(path_list, start=1):
                print("Compressing {:>2}/{:>2}: {}".format(i, len(path_list), path))
                task_id = path.split("/")[-2]
                file_list = os.listdir(path)
                for file_name in file_list:
                    _path = os.path.join(path, file_name)
                    zip_obj.write(_path, os.path.join(task_id, file_name))
        print("Done!")
