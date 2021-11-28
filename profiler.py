import matplotlib.pyplot as plt
import numpy as np
from numpy.random.mtrand import randint
from time import perf_counter
from utils import *
from svd_func import *


#RAJOUTER LERREUR / ERREUR TOTALE
def execute_t_times(t, svd_func, A, k):
    """Executes t times a randomized svd function and plots duration and error for k ascending 

    Args:
        t (int): number of time to execute the function, k will increase each time by k/t, starting at k/t
        svd_func (function svd_rand): the svd to execute
        A (np.array): The matrix using to compute svd
        k (int): number max of columns to use in the randomized svd

    Returns:
        null : plots the array
    """    
    step = int(k/t)

    
    tab_k = [i for i in range(step, k - 1, step)]
    
    avg_error, avg_magn, avg_duration = 0, 0, 0
    
    error = []
    duration = []
    i = 1
    
    # print(tab_k)
    for m_k in tab_k:
        print("({}/{}) Running SVD with k = {}...".format(i, len(tab_k),m_k))
        start = perf_counter()  
        A_tilde = svd_func(A,m_k) #compute A_tild using randomized svd
        end = perf_counter()
    
        #get the sum of all error for each composant
        rms = abs(np.linalg.norm(A - A_tilde))
    
        #add this sum the the avg_summ
        error.append(rms)
        duration.append(end-start)
        i += 1
    

        
        
    plt.figure(figsize=(5, 5))
    plt.title("SVD compute duration with k ascending")
    plt.plot(tab_k, duration, color='b', label='duration')
    plt.xlabel('k')
    plt.ylabel('t (sec)')
    
    plt.figure(figsize=(5, 5))
    plt.title("SVD exact error with k ascending")
    plt.plot(tab_k, error, color='r', label='error')
    plt.xlabel('k')
    plt.ylabel('total error residual')
    
    plt.show()
    
    return


def erreurs_rangFixe(sizes, k=50):
    """Executes exact SVD and all randomized SVD on different sized generated square matrices, computes the error

    Args:
        sizes (list of strings): square matrix sizes to try
        k (int): number max of columns to use in the randomized svd

    Returns:
        errors_exactsvd: list of approximation error of exact svd
        errors_rsvd_gauss: list of approximation error of r-svd with gaussian RM
        errors_rsvd_uni: list of approximation error of r-svd with uniform RM
        errors_rsvd_col: list of approximation error of r-svd with column sampling
    """
    errors_exactsvd = []
    errors_rsvd_gauss = []
    errors_rsvd_uni = []
    errors_rsvd_col = []
    for size in sizes:
        M = rankk_random_matrix(size, size, k)
        errors_exactsvd.append(np.linalg.norm(svd_regular(M, k=k) - M))

        errors_rsvd_gauss.append(np.linalg.norm(svd_rand_gaussian(M, k) - M))
        errors_rsvd_uni.append(np.linalg.norm(svd_rand_uniform(M, k) - M))
        errors_rsvd_col.append(np.linalg.norm(svd_rand_columns(M, k) - M))
    
    return errors_exactsvd, errors_rsvd_gauss, errors_rsvd_uni, errors_rsvd_col


def erreurs_rangFixe_photos(photos, k=100): # En chantier (je m'appelle teuse)
    """Executes exact SVD and all randomized SVD on different sized photos, computes the error

    Args:
        photos (list of strings): photo paths
        k (int): number max of columns to use in the randomized svd

    Returns:
        errors_exactsvd: list of approximation error of exact svd
        errors_rsvd_gauss: list of approximation error of r-svd with gaussian RM
        errors_rsvd_uni: list of approximation error of r-svd with uniform RM
        errors_rsvd_col: list of approximation error of r-svd with column sampling
        sizes: list of matrix sizes
    """
    errors_exactsvd = []
    errors_rsvd_gauss = []
    errors_rsvd_uni = []
    errors_rsvd_col = []
    sizes = []
    for img in photos:
        M = toGrayScale(getColouredImage(img))
        shape = M.shape
        if shape[0] < shape[1]:
            M = M.transpose()
        size = shape[0] * shape[1] 

        errors_exactsvd.append(np.linalg.norm(svd_regular(M, k=k) - M))

        errors_rsvd_gauss.append(np.linalg.norm(svd_rand_gaussian(M, k) - M))
        errors_rsvd_uni.append(np.linalg.norm(svd_rand_uniform(M, k) - M))
        errors_rsvd_col.append(np.linalg.norm(svd_rand_columns(M, k) - M))
    
    return errors_exactsvd, errors_rsvd_gauss, errors_rsvd_uni, errors_rsvd_col, sizes


def print_result(error, magnitude, duration, name):
    print("\n####################################################################")
    print(name)
    print(">>> Duration : {:.5f} sec.".format(duration))
    print(">>> Error : 10e" + str(magnitude))
    print(">>> Exact error RMS : " + str(error))
    print("####################################################################")
    
    return
