import numpy as np


def permutate(A, index_origin, index_target, axis=0):
    if axis == 0:
        C = A[:, index_target].copy()
        A[:, index_target] = A[:, index_origin]
        A[:, index_origin] = C
        return A
    elif axis == 1:
        R = A[index_target, :].copy()
        A[index_target, :] = A[index_origin, :]
        A[index_origin, :] = R
        return A
    else:
        raise ValueError('Invalid selection of axis')


def multiply(A, index, coef, axis=0):
    if axis == 0:
        A[:, index] = coef * A[:, index]
        return A
    elif axis == 1:
        A[index, :] = coef * A[index, :]
        return A
    else:
        raise ValueError('Invalid selection of axis')


def add(A, index_origin, coef, index_target, axis=0):
    if axis == 0:
        A[:, index_target] = A[:, index_target] + coef * A[:, index_origin]
        return A
    elif axis == 1:
        A[index_target, :] = A[index_target, :] + coef * A[index_origin, :]
        return A
    else:
        raise ValueError('Invalid selection of axis')


def get_non_zero_diagonal(A, index, Ainv):
    if A[index, index] != 0:
        return A, Ainv
    elif A.shape == (1, 1):
        raise ValueError("Singular matrix")
    else:
        n = A.shape[0]
        # TODO: select the largest, element in the row
        for i in range(index + 1, n):
            if A[index, i] != 0:
                A = permutate(A, index_origin=index, index_target=i, axis=0)

                Ainv = permutate(Ainv, index_origin=index, index_target=i, axis=0)
                return A, Ainv
        raise ValueError("Singular matrix")


def get_zero_column(A, index, Ainv, descending=True):
    n = A.shape[0]
    start, stop = [index + 1, n] if descending else [0, index]
    for i in range(start, stop):
        p = -A[i, index] / A[index, index]
        A = add(A, index_origin=index, coef=p, index_target=i, axis=1)
        Ainv = add(Ainv, index_origin=index, coef=p, index_target=i, axis=1)

    return A, Ainv


def inverse(matrix):
    A = matrix.copy()
    if A.shape[0] != A.shape[1]:
        raise ValueError('Matrix is not square')
    n = A.shape[0]
    if n == 1:
        return np.matrix([[1 / A[0, 0]]])
    Ainv = np.identity(n)

    for i in range(n - 1):
        A, Ainv = get_non_zero_diagonal(A, i, Ainv)
        A, Ainv = get_zero_column(A, i, Ainv)

    for i in range(n - 1):
        A, Ainv = get_zero_column(A, n - 1 - i, Ainv, descending=False)

    for i in range(n):
        Ainv = multiply(Ainv, index=i, coef=1 / A[i, i], axis=1)
        A = multiply(A, index=i, coef=1 / A[i, i], axis=1)
    return Ainv, A


if __name__ == "__main__":
    A = np.matrix([[1, 2, 3], [0, 1, 2], [0, 0, 1]])
    A = np.matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    A = np.matrix([[1, 2, 3], [1, 1, 1], [0, 4, 5]])

    Ainv, _ = inverse(A)
    print(Ainv)
    print(np.linalg.inv(A))
