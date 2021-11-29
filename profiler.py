import matplotlib.pyplot as plt
import numpy as np
from numpy.random.mtrand import randint
from math import ceil, log10
from time import perf_counter
from utils import *
from svd_func import *


#RAJOUTER LERREUR / ERREUR TOTALE
def execute_t_times(t, svd_func, A, k, verbose=1):
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

    m, n = A.shape
    size = m*n
    
    tab_k = [i for i in range(step, k - 1, step)]
        
    error = []
    duration = []
    i = 1
    
    # print(tab_k)
    for m_k in tab_k:
        if(verbose == 1) :
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
    

    # Plot compressed and original image
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10/2.1))

    axes[0].set_title('SVD compute duration with k ascending')
    axes[0].plot(tab_k, duration, color='b', label='duration')
    axes[1].set_title("truncated SVD error (size = {})".format(size))
    axes[1].plot(tab_k, error, color='r', label='error')  
    
    axes[0].grid()
    axes[1].grid()
    plt.show()
    
    return

def fixed_rank_errors_random_matrixes(sizes, k=50):    
    duration_exactsvd = []
    duration_rsvd_gauss = []
    duration_rsvd_uni = []
    duration_rsvd_col = []
    
    errors_exactsvd = []
    errors_rsvd_gauss = []
    errors_rsvd_uni = []
    errors_rsvd_col = []
    for size in sizes:
        M = rankk_random_matrix_generator(size, size, k)

        s = perf_counter()
        errors_exactsvd.append(np.linalg.norm(svd_regular(M, k=k) - M))
        e = perf_counter()
        duration_exactsvd.append(e - s)

        s = perf_counter()
        errors_rsvd_gauss.append(np.linalg.norm(svd_rand_gaussian(M, k) - M))
        e = perf_counter()
        duration_rsvd_gauss.append(e - s)
        
        s = perf_counter()
        errors_rsvd_uni.append(np.linalg.norm(svd_rand_uniform(M, k) - M))
        e = perf_counter()
        duration_rsvd_uni.append(e - s)
        
        s = perf_counter()
        errors_rsvd_col.append(np.linalg.norm(svd_rand_columns(M, k) - M))
        e = perf_counter()
        duration_rsvd_col.append(e - s)
        

    # Plot compressed and original image

    errors = [errors_exactsvd, errors_rsvd_gauss, errors_rsvd_uni, errors_rsvd_col]
    durations = [duration_exactsvd, duration_rsvd_gauss, duration_rsvd_uni, duration_rsvd_col]
        
    plt.figure()
    plt.title("SVD error with matrix approximation of rank K = {}".format(50))
    plt.plot(sizes, errors[0], label = "exact SVD")
    plt.plot(sizes, errors[1], label = "Random SVD (Gauss)")
    plt.plot(sizes, errors[2], label = "Random SVD (Uniform)")
    plt.plot(sizes, errors[3], label = "Random SVD (Col sampling)")
    plt.xlabel("Size of Square Matrix N")
    plt.ylabel("Reconstruction error of svd")
    plt.grid()
    plt.legend(loc='center right', bbox_to_anchor=(2, 0.5))

    plt.figure()
    plt.title("SVD compute duration")
    plt.plot(sizes, durations[0], label = "exact SVD")
    plt.plot(sizes, durations[1], label = "Random SVD (Gauss)")
    plt.plot(sizes, durations[2], label = "Random SVD (Uniform)")
    plt.plot(sizes, durations[3], label = "Random SVD (Col sampling)")
    plt.xlabel("Size of Square Matrix N")
    plt.ylabel("Compute time")
    plt.grid()
    # plt.legend()
    
    plt.show()

    return 
