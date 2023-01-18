import numpy as np
import scipy.linalg


def permutate_non_zero_diagonal(A):
    P = np.identity(A.shape[0])
    for n in range(A.shape[0] - 1):
        imax = np.argmax(abs(A[n + 1:, n]))
        P[imax, imax] = 0
        P[n, n] = 0
        P[imax, n] = 1
        P[n, imax] = 1
    return P.dot(A), P


def lu(A):
    # Doolittle decomposition (L has 1 on the diagonal)
    Ap = A.copy()
    N = Ap.shape[0]
    L = np.matrix(np.identity(Ap.shape[0]))
    Ap, P = permutate_non_zero_diagonal(Ap)
    for n in range(N - 1):
        Ln = np.matrix(np.identity(N))
        Ln[n + 1:, n] = -Ap[n + 1:, n] / Ap[n, n]
        Ap = Ln.dot(Ap)
        L[n + 1:, n] = -Ln[n + 1:, n]
    U = Ap
    return P, L, U


def lu_doolittle(A):
    if not isinstance(A, np.matrix):
        raise ValueError("Input is not a numpy.matrix object.")
    if A.shape[0] != A.shape[1]:
        raise ValueError("Input matrix is not square.")
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    for i in range(n):
        # Update U first
        for j in range(i, n):
            if i == 0:
                U[i, j] = A[i, j]
            else:
                U[i, j] = A[i, j] - np.dot(L[i, :], U[:, j])
        for j in range(i + 1):
            if j == 0:
                L[i, j] = A[j, j] / U[j, j]
            elif j == i:
                L[i, j] = 1
            else:
                if U[j, j] != 0:
                    L[i, j] = (A[i, j] - np.dot(L[i, :], U[:, j])) / U[j, j]
    P = np.identity(n)
    return P, L, U


if __name__ == '__main__':
    A = np.matrix([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
    A = np.matrix([[2, -1, -2],
                   [-4, 6, 3],
                   [-4, -2, 8]])
    # A = np.matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    # A = np.matrix([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])

    P, L, U = lu(A)
    print('UDF')
    # print(P)
    print(L)
    print(U)
    print(P.dot(L).dot(U))
    Pref, Lref, Uref = scipy.linalg.lu(A)
    print("Reference")
    # print(Pref)
    print(Lref)
    print(Uref)
    print(Pref.dot(Lref).dot(Uref))
    # print(P.dot(A))
    # print(Pref.dot(A))
