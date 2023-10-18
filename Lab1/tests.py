from matrix_multiplication import binet, strassen, alpha_tensor, get_and_reset_flops
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


def perform_tests(dims):
    """
    Perform  time and correct test on every function from dims.keys().
    Return tuples (flops, times) where flops is number of floating point operation in A * B and time is whole time
    for A * B operation
    """
    times = {f.__name__: [] for f in dims.keys()}
    flops = {f.__name__: [] for f in dims.keys()}

    for f, dim_list in dims.items():
        for dim1, dim2 in dim_list:
            A = np.random.rand(dim1[0], dim1[1])
            B = np.random.rand(dim2[0], dim2[1])

            time_f = time.time()
            if not is_same(f(A, B), A @ B):
                print()
                return False

            times[f.__name__].append(time.time() - time_f)
            flops[f.__name__].append(get_and_reset_flops(f.__name__))

    return flops, times


def plot(y, x, xlabel, ylabel, title):
    plt.plot(x['binet'], y['binet'], label="Binet Method", color='blue')
    plt.plot(x['strassen'], y['strassen'], label="Strassen Method", color='red')
    plt.plot(x['alpha_tensor'], y['alpha_tensor'], label="Alpha Tensor method", color='green')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)

    plt.show()


binet_strassen_dims = [[(2, 2), (2, 2)], [(4, 4), (4, 4)], [(8, 8), (8, 8)], [(16, 16), (16, 16)],
                       [(32, 32), (32, 32)], [(64, 64), (64, 64)], [(128, 128), (128, 128)]]
alpha_tensor_dims = [[(4, 5), (5, 5)], [(16, 25), (25, 25)], [(64, 125), (125, 125)]]

products_binet_strassen = list(map(lambda sub_arr: [x * y for x, y in sub_arr], binet_strassen_dims))

binet_strassen_elements = list(map(lambda sub_arr: sub_arr[0] * sub_arr[1], products_binet_strassen))

products_alpha_tensor = list(map(lambda sub_arr: [x * y for x, y in sub_arr], alpha_tensor_dims))
alpha_tensor_elemets = list(map(lambda sub_arr: sub_arr[0] * sub_arr[1], products_alpha_tensor))

dims = {binet: binet_strassen_dims,
        strassen: binet_strassen_dims,
        alpha_tensor: alpha_tensor_dims}
flops, times = perform_tests(dims)

number_of_elements = {binet.__name__: binet_strassen_elements,
                      strassen.__name__: binet_strassen_elements,
                      alpha_tensor.__name__: alpha_tensor_elemets}

plot(flops, number_of_elements, 'Number of elements', 'Flops', 'Flops comparison between Matrix Multiplication Methods')
plot(times, number_of_elements, 'Number of elements', 'Time[s]',
     'Time comparison between Matrix Multiplication Methods')
