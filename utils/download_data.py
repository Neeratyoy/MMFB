###################################################
# Should download and cache all required datasets #
###################################################

import openml
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_tasks", type=int, default=3, help="Number of tasks")
    args = parser.parse_args()

    automl_benchmark = openml.study.get_suite(218)
    task_ids = automl_benchmark.tasks[:args.n_tasks]

    # Fetches from OpenML server
    for i, task_id in enumerate(task_ids, start=1):
        print("{}/{} --- {}".format(i, len(task_ids), task_id), end=" ")
        try:
            task = openml.tasks.get_task(task_id, download_data=False)
            dataset = openml.datasets.get_dataset(task.dataset_id, download_data=True)
        except Exception as e:
            print(e)
            continue
