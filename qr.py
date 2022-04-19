import numpy as np


def compute_householder(C):
    m = C.shape[0]
    if max(abs(C[1:, 0])) == 0:
        return np.identity(m)
    eps = 1 if C[0, 0] < 0 else -1  # TODO: this is for numerical stability (to investigate)
    alpha = eps * np.linalg.norm(C)
    e = np.zeros((m, 1))
    e[0, 0] = 1
    u = C - alpha * e
    v = u / np.linalg.norm(u)
    return np.identity(m) - 2 * v.dot(v.T)


def compute_HH(A):
    return compute_householder(A[:, 0])


def is_upper_matrix(A):
    for j in range(A.shape[1] - 1):
        if max(abs(A[j + 1:, j])) > 0:
            return False
    return True


def QR_decomposition_HH(A, Q=None, R=None, N0=None):
    # TODO: corriger
    if Q is None:
        Q = np.identity(A.shape[0])
    if R is None:
        R = A.copy()
    if N0 is None:
        N0 = A.shape[0]
    if A.shape[0] <= 1:
        return np.identity(1), Q, R, N0
    if is_upper_matrix(R):
        return A, Q, R, N0
    H = compute_HH(A)

    n = H.shape[0]
    I = np.identity(N0 - n)
    z = np.zeros((N0 - n, n))
    Qn = np.concatenate(
        [np.concatenate([I, z], axis=1),
         np.concatenate([z.T, H], axis=1)],
        axis=0
    )
    R = Qn.dot(R)
    Q = Qn.dot(Q)
    A = R[n + 1:, n + 1:]
    return QR_decomposition_HH(A, Q, R, N0)


def proj(x, u):
    return (x.T.dot(u) / u.T.dot(u))[0, 0] * u


def QR_decomposition_GS(A0):
    A = A0.copy()
    n = A.shape[0]
    Q = np.zeros((n, n))
    U = []
    for j in range(n):
        u = A[:, j]
        for u_k in U:
            u = u - proj(A[:, j], u_k)
        U.append(u)
        Q[:, j] = np.array(u / np.linalg.norm(u))[:, 0]
    # Q = Q.T
    R = Q.T.dot(A)
    return Q, R


def QR(A, method="GS"):
    """wrapper"""
    A0 = A.copy()
    if method == "HH":
        _, Q, R, _ = QR_decomposition_HH(A0)
    elif method == "GS":
        Q, R = QR_decomposition_GS(A0)
    return Q, R




if __name__ == '__main__':
    A = np.matrix([[1, 2, 3], [0, 1, 2], [0, 0, 1]])
    A = np.matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

    # v=np.zeros((1,2))
    # print(np.concatenate([v,v],axis=0))

    print(A)
    Q, R = QR(A)
    # Q, R = np.linalg.qr(A)
    print(Q)
    # print(Q.T.dot(Q))
    print(R)
    # print(is_upper_matrix(R))
    # print(Q.dot(R))
    Q, R = np.linalg.qr(A)
    print(Q)
    # print(Q.T.dot(Q))

    print(R)
    # print(Q.dot(R))
    # print(is_upper_matrix(R))
