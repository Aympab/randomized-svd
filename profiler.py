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




def erreurs_rangFixe(k=50):
    plt.figure(figsize=(10, 6))
    plt.title("SVD error with matrix size ascending")
    
    sizes = [100, 500, 1000]#, 5000, 10000]
    errors_exactsvd = []
    for size in sizes:
        M = rankk_random_matrix(size, size, k)
        errors_exactsvd.append(np.linalg.norm(svd_regular(M) - M))
    
    errors_rsvd = []
    for size in sizes: 
        M = rankk_random_matrix(size, size, k)
        errors_rsvd.append(np.linalg.norm(svd_rand_gaussian(M, k), M))
    
    return errors_exactsvd, errors_rsvd


def print_result(error, magnitude, duration, name):
    print("\n####################################################################")
    print(name)
    print(">>> Duration : {:.5f} sec.".format(duration))
    print(">>> Error : 10e" + str(magnitude))
    print(">>> Exact error RMS : " + str(error))
    print("####################################################################")
    
    return
