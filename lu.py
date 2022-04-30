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


if __name__ == '__main__':
    A = np.matrix([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
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
