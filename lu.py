import numpy as np


def lu(A):
    P = None
    L = None
    U = None
    return P, L, U


if __name__ == '__main__':
    A = np.matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    P, L, U = lu(A)
