from matrix_multiplication import binet, strassen, get_and_reset_flops
import numpy as np
from matplotlib import pyplot as plt
import time

EPSILON = 10 ** -8


def is_same(A1, A2):
    """
    Checks if A1 contains the same numbers as A2 (with possible error = EPSILON) on every number.
    """
    for r_id, row in enumerate(A1):
        for el_id, el in enumerate(row):
            if abs(el - A2[r_id, el_id]) > EPSILON:
                return False

    return True


def perform_tests(dims, *args):
    """
    Perform  time and correct test on every function from *args.
    Return tuples (flops, times) where flops is number of floating point operation in A * B and time is whole time
    for A * B operation
    """
    times = {f.__name__: [] for f in args}
    flops = {f.__name__: [] for f in args}
    for dim in dims:
        A = np.random.rand(dim, dim)
        B = np.random.rand(dim, dim)
        for f in args:
            time_f = time.time()
            if not is_same(f(A, B), A @ B):
                print(A @ B, f(A, B), sep="\n____________________________\n")
                return False

            times[f.__name__].append(time.time() - time_f)
            flops[f.__name__].append(get_and_reset_flops(f.__name__))

    return flops, times


def plot(y, x, xlabel, ylabel, title):
    plt.plot(x, y['binet'], label="Binet Method", color='blue')
    plt.plot(x, y['strassen'], label="Strassen Method", color='red')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)

    plt.show()


dims = [2, 4, 8, 16, 32, 64, 128]
flops, times = perform_tests(dims, binet, strassen)
x = [el * el for el in dims]

plot(flops, x, 'Number of elements', 'Flops', 'Flops comparison between Binet and Strassen')
plot(times, x, 'Number of elements', 'Time[s]', 'Time comparison between Binet and Strassen')
