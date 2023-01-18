import numpy as np


def cholesky(A0, i=0):
    A = A0.copy()
    n = A.shape[0]
    if (A.T != A).any():
        raise ValueError("Input matrix is not symetrical")
    if A[i, i] < 0:
        raise ValueError("Input matrix is not SDP")
    if i == n - 1:
        L = np.identity(n)
        L[n - 1, n - 1] = A[n - 1, n - 1] ** 0.5
        return L
    else:
        #Compute L
        L = np.concatenate([
            np.concatenate([np.identity(i), np.zeros((i, n - i))], axis=1),
            np.concatenate([np.zeros((1, i)), np.matrix(A[i, i] ** 0.5), np.zeros((1, n - i - 1))], axis=1),
            np.concatenate(
                [np.zeros((n - i - 1, i)), A[i + 1:, i] / A[i, i] ** 0.5, np.identity(n - i - 1)],
                axis=1)
        ], axis=0)
        #Update A
        A = np.concatenate([
            np.concatenate([np.identity(i), np.zeros((i, n - i))], axis=1),
            np.concatenate([np.zeros((1, i)), np.matrix(1), np.zeros((1, n - i - 1))], axis=1),
            np.concatenate(
                [np.zeros((n - i - 1, i + 1)), A[i + 1:, i + 1:] - A[i + 1:, i].dot(A[i + 1:, i].T) / A[i, i]],
                axis=1)
        ], axis=0)
        return L.dot(cholesky(A, i=i + 1))


if __name__ == '__main__':
    A = np.matrix([[2, 1, 1], [1, 2, 1], [1, 1, 2]])  # A is SDP
    print(np.linalg.det(A))

    L = cholesky(A)
    print(L)
    L = np.linalg.cholesky(A)
    print(L)
    print(L.dot(L.T))
