import numpy as np
from numpy.random.mtrand import randint
from utils import *


def reconstruct(u, s, vt, k=0):
    if k != 0:
        return u[:, :k].dot(s[:k, np.newaxis] * vt[:k, :])
    else:
        return u.dot(s[:, np.newaxis] * vt)


def r_svd(A, k=0, kernel="gaussian", return_matrices=False, power_iteration=0):
    """Computes a randomized SVD for A using the uniform law

    Args:
        A (np.array): The matrix which we want to compute SVD
        k (int): Number of columns projection used to estimate A
        kernel (str): what method to use

    Returns:
        np.array: The reconstructed matrix U.S.Vt
    """    
    #Number of rows in A
    m, n = A.shape
    
    #We create a matrix Omega which has k vector of size n filled with random numbers
    if (kernel == "uniform"):
        Omega = np.random.rand(n, k)
    elif (kernel == "gaussian"):
        Omega = gaussianMatrixGenerator(n,k)
    elif (kernel == "colsampling"):
        Omega = np.zeros(shape=(n,k))
        #In each column, we put a 1 at a random position, 
        #this will lead in choosing random columns of A to project  
        for c in Omega :
            random_int = randint(0, k)
            c[random_int] = 1
    elif (kernel == "SRHT"):
        Omega = hadamard_random_matrix(n, k)
    elif (kernel == "DCT"):
        Omega = DCT_random_matrix(n, k)
    else:
        raise ValueError("kernel must be either of 'uniform', 'gaussian', 'colsampling', 'SRHT', 'DCT'.")


    #We randomly project k columns of A and create a Y matrix
    Y = A @ Omega
    #Y = Y / np.linalg.norm(Y, axis=0)

    for i in range(power_iteration):
        Y = A.T @ Y
        Y = Y / np.linalg.norm(Y, axis=0)
        Y = A @ Y 
        Y = Y / np.linalg.norm(Y, axis=0)

    #We compute QR on Y because it is small (m,k)
    Q,R = np.linalg.qr(Y)
    
    #We project A into the smaller orthnonormal space Q to create the matrix B
    B = Q.T @ A

    #We can now compute a fast SVD on B
    U_tilde, S_tilde, Vt_tilde = np.linalg.svd(B, full_matrices=False)

    #We reconstruct the left singular vectors of by re-projecting by Q (before it was Qt)
    reconstructed_U = Q @ U_tilde
    
    #And we return the A_tilde, our reconstructed matrix    
    if return_matrices:
        return reconstructed_U , S_tilde, Vt_tilde
    else:
        return reconstruct(reconstructed_U , S_tilde, Vt_tilde)


def svd_regular(A, k=0, return_matrices=False):
    """Computes the regular SVD for A and reconstructs A with U, S, and V

    Args:
        A (np.array): The matrix which we want to compute SVD

    Returns:
        np.array: The reconstructed matrix U.S.Vt
    """    
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    if return_matrices:
        return U, S, Vt
    else:
        return reconstruct(U, S, Vt, k=k)

