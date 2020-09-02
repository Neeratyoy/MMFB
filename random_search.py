import numpy as np
from matplotlib import pyplot as plt

from benchmark import RandomForestBenchmark


def random_search(benchmark, iterations, verbose=False):
    results = []
    inc_config = None
    inc_fidelity = None
    inc_score = np.inf

    for i in range(1, iterations + 1):
        if verbose:
            print("Run {:<3}/{:<3}:".format(i, iterations), end=" ")
        config = benchmark.get_config()
        fidelity = benchmark.get_fidelity()
        result = benchmark.objective(config, fidelity)
        if result['function_value'] < inc_score:
            inc_score = result['function_value']
            inc_config = config
            inc_fidelity = fidelity
        results.append(result)
        if verbose:
            print(inc_score)

    return inc_score, inc_config, inc_fidelity, results


if __name__ == "__main__":
    # hepatitis: https://www.openml.org/t/54
    rf_bench = RandomForestBenchmark(task_id=54)
    rf_bench.load_data_automl()

    inc_score, inc_config, inc_fidelity, results = random_search(rf_bench, 1000, verbose=True)
    print(results[-3:], '\n')
    print("Best score: {}".format(inc_score))
    print("Incumbent: \n{}".format(inc_config))
    print("Best fidelity: \n{}".format(inc_fidelity))

    trajectory = []
    inc = np.inf
    for i, res in enumerate(results):
        if res['function_value'] < inc:
            inc = res['function_value']
        trajectory.append(inc)

    plt.plot(trajectory)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('# function evaluations')
    plt.ylabel('loss')
    plt.title('incumbent trace')
    plt.show()
