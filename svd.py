import numpy as np
import matplotlib.pyplot as plt


def givens_matrix(n, i, j, theta):
    G = np.identity(n)
    G[i, i] = np.cos(theta)
    G[i, j] = -np.sin(theta)
    G[j, i] = np.sin(theta)
    G[j, j] = np.cos(theta)
    return G


def compute_theta(A, i, j):
    if (A[i, i] == A[j, j]).all():
        return np.pi / 4
    else:
        return np.arctan(2 * A[i, j]) / (A[j, j] - A[i, i]) / 2


def offdiagsum(A):
    return np.sum(A * A - np.diag(A * A)) ** 2


def index_max(A):
    A0 = abs(A).copy()
    #Exclude diagonal elements
    for i in range(A0.shape[0]):
        A0[i, i] = 0
    index = np.where(A0 == np.amax(A0))
    return index


def condition(n_iteration, A):
    return (n_iteration > 0)


def eigen_decomposition(A0, n_sweep=10):
    n = A0.shape[0]
    N = int(n * (n - 1) / 2)
    n_iteration = n_sweep * N
    A = A0.copy()
    H = np.identity(n)
    offdiagsum_hist = [offdiagsum(A)]
    while condition(n_iteration, A):
        i, j = index_max(A)
        theta = compute_theta(A, i, j)
        Hn = givens_matrix(n, i, j, theta)
        H = Hn.dot(H)
        A = Hn.dot(A).dot(Hn.T)
        offdiagsum_hist.append(offdiagsum(A))
        n_iteration -= 1

    return np.diag(A), A, offdiagsum_hist


def svd(A, verbose=False, tolerance=1e-3):
    # U, _ = eigen_decomposition(A.dot(A.T))
    # V, _ = eigen_decomposition(A.T.dot(A))
    lambda_u, U = np.linalg.eig(A.dot(A.T))
    lambda_v, V = np.linalg.eig(A.T.dot(A))
    # Sort by descending order eigenvalues
    idx_u = lambda_u.argsort()[::-1]
    idx_v = lambda_v.argsort()[::-1]
    lambda_u = lambda_u[idx_u]
    lambda_v = lambda_v[idx_v]
    delta = abs(lambda_v - lambda_u).max()
    if (delta > tolerance) and verbose:
        print(f"Exceed tolerance: {delta} vs. {tolerance}")
    U = U[:, idx_u]
    V = V[:, idx_v]
    print(lambda_u, lambda_v)
    Sigma = U.T.dot(A).dot(V)
    return U, np.diag(Sigma), V


if __name__ == '__main__':
    A = np.matrix([[1, 2, 3], [0, 1, 2], [0, 0, 1]])
    # A = np.matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

    # U_ref, D_ref, V_ref = np.linalg.svd(A)
    # U, D, V = svd(A)
    # print(U)
    # print(U_ref)
    #
    # print(V)
    # print(V_ref)
    # # print(D)
    #
    # print(np.diag(D_ref))
    #
    # print(A)
    # print(U.dot(np.diag(D)).dot(V.T))
    # print(U_ref.dot(np.diag(D_ref)).dot(V_ref))
    #
    v_ref, H_ref = np.linalg.eig(A)

    v, H, dec = eigen_decomposition(A)
    plt.plot(dec)
    plt.show()
    print(v_ref)
    print(v)
    print(H_ref)
    print(H)
