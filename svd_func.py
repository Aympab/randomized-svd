import numpy as np
from numpy.random.mtrand import randint
from utils import *

def svd_regular(A):
    """Computes the regular SVD for A and reconstructs A with U, S, and V

    Args:
        A (np.array): The matrix which we want to compute SVD

    Returns:
        np.array: The reconstructed matrix U.S.Vt
    """    
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    return U@np.diag(S)@Vt


def svd_rand_uniform(A, k):
    """Computes a randomized SVD for A using the uniform law

    Args:
        A (np.array): The matrix which we want to compute SVD
        k (int): Number of columns projection used to estimate A

    Returns:
        np.array: The reconstructed matrix U.S.Vt
    """    
    #Number of rows in A
    m, n = A.shape
    
    #We create a matrix Omega which has k vector of size n filled with random numbers
    Omega = np.random.rand(n, k)
    
    #We randomly project k columns of A and create a Y matrix
    Y = A@Omega

    #We compute QR on Y because it is small (m,k)
    Q,R = np.linalg.qr(Y)
    
    #We project A into the smaller orthnonormal space Q to create the matrix B
    B = Q.T @ A

    #We can now compute a fast SVD on B
    U_tilde, S_tilde, Vt_tilde = np.linalg.svd(B, full_matrices=False)

    #We reconstruct the left singular vectors of by re-projecting by Q (before it was Qt)
    reconstructed_U = Q @ U_tilde
    
    #And we return the A_tilde, our reconstructed matrix
    return reconstructed_U @ np.diag(S_tilde)@Vt_tilde


def svd_rand_columns(A, k):
    """Computes a randomized SVD for A using the random columns method

    Args:
        A (np.array): The matrix which we want to compute SVD
        k (int): Number of columns projection used to estimate A

    Returns:
        np.array: The reconstructed matrix U.S.Vt
    """    
    m, n = A.shape
    
    #We create a matrix full of zeros
    Omega = np.zeros(shape=(n,k))

    #In each column, we put a 1 at a random position, 
    #this will lead in choosing random columns of A to project  
    for c in Omega :
        random_int = randint(0, k)
        c[random_int] = 1
        

    Y = A@Omega
    Q,R = np.linalg.qr(Y)   
     
    B = Q.T @ A    
    U_tilde, S_tilde, Vt_tilde = np.linalg.svd(B, full_matrices=False)
    reconstructed_U = Q @ U_tilde
    
    return reconstructed_U @ np.diag(S_tilde)@Vt_tilde


def svd_rand_gaussian(A, k):
    """Computes a randomized SVD for A using a gaussian law

    Args:
        A (np.array): The matrix which we want to compute SVD
        k (int): Number of columns projection used to estimate A

    Returns:
        np.array: The reconstructed matrix U.S.Vt
    """ 
    m, n = A.shape
    
    #This time Omega is filled with random values generated from a standard normal distribution law
    Omega = gaussianMatrixGenerator(n,k)

    Y = A@Omega
    Q,R = np.linalg.qr(Y)
    B = Q.T @ A

    U_tilde, S_tilde, Vt_tilde = np.linalg.svd(B, full_matrices=False)
    reconstructed_U = Q @ U_tilde


    return reconstructed_U @ np.diag(S_tilde)@Vt_tilde