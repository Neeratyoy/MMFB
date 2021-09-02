"""
Script to print latex table collecting Task IDs used and the #obs, #feat and OpenML URLs
"""

import openml
import pandas as pd


all_task_ids_by_in_mem_size = [
    10101, 53, 146818, 146821, 9952, 146822, 31, 3917, 168912, 3, 167119, 12, 146212, 168911,
    9981, 168329, 167120, 14965, 146606,  # < 30 MB
    168330, 7592, 9977, 168910, 168335, 146195, 168908, 168331,  # 30-100 MB
    168868, 168909, 189355, 146825, 7593,  # 100-500 MB
    168332, 168337, 168338,  # > 500 MB
    189354, 34539,  # > 2.5k MB
    3945,  # >20k MB
    # 189356  # MemoryError: Unable to allocate 1.50 TiB; array size (256419, 802853) of type float64
]
paper_tasks = [
    10101, 53, 146818, 146821, 9952, 146822, 31, 3917, 168912, 3, 167119, 12, 146212, 168911,
    9981, 167120, 14965, 146606, 7592, 9977
]


if __name__ == "__main__":
    # essential step to print full string width with URL for easy, correct, copy-paste
    pd.options.display.max_colwidth = 100

    task_ids = paper_tasks
    dfs = []
    latex_str = "\\href{{{}}}{{{}}}"
    cols = ["name", "tid", "#obs", "#feat"]
    for tid in task_ids:
        print("{:>5}".format(tid), end="\r")
        task = openml.tasks.get_task(tid, download_data=False)
        d = openml.datasets.get_dataset(task.dataset_id, download_data=False)
        indices = task.get_train_test_split_indices()
        nobs = len(indices[0]) + len(indices[1])
        nfeat = len(d.features)
        name = latex_str.format(task.openml_url, d.name)
        dfs.append(pd.DataFrame([[name, tid, nobs, nfeat]], columns=cols))

    df = pd.concat(dfs)

    latex = df.to_latex(index=False).replace("\\{", "{").replace("\\}", "}")
    latex = latex.replace("\\textbackslash ", "\\")
    print(latex)
