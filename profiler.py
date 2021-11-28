import matplotlib.pyplot as plt
import numpy as np
from numpy.random.mtrand import randint
from math import ceil, log10
from time import perf_counter
from utils import *
from svd_func import *
import os


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
        
        # os.system('cls' if os.name == 'nt' else "printf '\033c'")
        # print("({}/{}) Running SVD with k = {}...".format(i, len(tab_k),m_k))
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
        

    return [errors_exactsvd, errors_rsvd_gauss, errors_rsvd_uni, errors_rsvd_col], [duration_exactsvd, duration_rsvd_gauss, duration_rsvd_uni, duration_rsvd_col]
