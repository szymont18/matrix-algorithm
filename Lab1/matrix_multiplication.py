import numpy as np

binet_flops = 0
strassen_flops = 0


def divide_matrix(M, n):
    A11 = M[0: n // 2, 0: n // 2]
    A12 = M[0: n // 2, n // 2: n]
    A21 = M[n // 2: n, 0: n // 2]
    A22 = M[n // 2: n, n // 2: n]

    return A11, A12, A21, A22


def binet(A: np.ndarray, B: np.ndarray):
    """
    :return: C = A * B where A, B, C are matrixs
    We can assume that dimensions of A and B are 2 ^ p.
    """
    global binet_flops

    n, m = len(A), len(A[0])
    p, q = len(B), len(B[0])

    if (n == 1 and m == 1) or (p == 1 and q == 1):
        binet_flops += 1
        return A * B

    A11, A12, A21, A22 = divide_matrix(A, n)
    B11, B12, B21, B22 = divide_matrix(B, n)

    C11 = binet(A11, B11) + binet(A12, B21)
    C12 = binet(A11, B12) + binet(A12, B22)
    C21 = binet(A21, B11) + binet(A22, B21)
    C22 = binet(A21, B12) + binet(A22, B22)

    return np.vstack(
        (np.hstack((C11, C12)),
         np.hstack((C21, C22)))
    )


def strassen(A, B):
    """
    :return: C = A * B where A, B, C are matrixs
    We can assume that dimensions of A and B are 2 ^ p.
    """
    global strassen_flops
    n = len(A)

    if n == 1:
        strassen_flops += 1
        return A * B

    A11, A12, A21, A22 = divide_matrix(A, n)
    B11, B12, B21, B22 = divide_matrix(B, n)

    P1 = strassen(A11 + A22, B11 + B22)
    P2 = strassen(A21 + A22, B11)
    P3 = strassen(A11, (B12 - B22))
    P4 = strassen(A22, (B21 - B11))
    P5 = strassen(A11 + A12, B22)
    P6 = strassen(A21 - A11, B11 + B12)
    P7 = strassen(A12 - A22, B21 + B22)

    return np.vstack(
        (np.hstack((P1 + P4 - P5 + P7, P3 + P5)),
         np.hstack((P2 + P4, P1 - P2 + P3 + P6)))
    )


def get_and_reset_flops(function_name):
    """
    Returns number of flops depends on function_name
    """
    global strassen_flops, binet_flops
    to_return = None

    if function_name == strassen.__name__:
        to_return = strassen_flops
        strassen_flops = 0

    elif function_name == binet.__name__:
        to_return = binet_flops
        binet_flops = 0

    return to_return
