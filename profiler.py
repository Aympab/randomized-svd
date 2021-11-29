import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

def cache_svd_photo(path, M, k):
    directory = "./resources/precomputed"
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)
    path = path.split("/")[-1].split(".")[0]
    if os.path.isfile(directory + "/" + path + "_u.npy") \
        & os.path.isfile(directory + "/" + path + "_s.npy") \
        & os.path.isfile(directory + "/" + path + "_vt.npy") \
        & os.path.isfile(directory + "/" + "durations.csv"):
        u = np.load(directory + "/" + path + "_u.npy")
        s = np.load(directory + "/" + path + "_s.npy")
        vt = np.load(directory + "/" + path + "_vt.npy")
        reconstructed = reconstruct(u, s, vt, k=k)
        durations = pd.read_csv(directory + "/" + "durations.csv")
    else:
        if (os.path.isfile(directory + "/" + "durations.csv")):
            durations = pd.read_csv(directory + "/" + "durations.csv")
        else: 
            durations = pd.DataFrame(columns = ["path", "duration"])
        s = perf_counter()
        U, S, Vt = svd_regular(M, k=k, return_matrices=True)
        reconstructed = reconstruct(U, S, Vt, k=k)
        e = perf_counter()
        durations = durations.append(pd.Series(index = ["path", "duration"], data = [path, e - s]), \
                    ignore_index=True) 
        np.save(directory + "/" + path + "_u.npy", U)
        np.save(directory + "/" + path + "_s.npy", S)
        np.save(directory + "/" + path + "_vt.npy", Vt)
        durations.to_csv(directory + "/" + "durations.csv", index=False)

    
    error = np.linalg.norm(reconstructed - M)
    duration = durations[durations.path == path].duration.values[0]
    return error, duration


def fixed_rank_errors_photos(paths, k=100):    
    duration_exactsvd = []
    duration_rsvd_gauss = []
    duration_rsvd_uni = []
    duration_rsvd_col = []
    
    errors_exactsvd = []
    errors_rsvd_gauss = []
    errors_rsvd_uni = []
    errors_rsvd_col = []

    sizes = []
    for path in paths:
        M = np.asarray(toGrayScale(getColouredImage(path)))
        shape = M.shape
        if shape[0] < shape[1]:
            M = M.transpose()
        sizes.append(shape[0] * shape[1])

        error, duration = cache_svd_photo(path, M, k)
        errors_exactsvd.append(error)
        duration_exactsvd.append(duration)

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
        

    sizes = np.asarray(sizes)
    ordre = np.argsort(sizes)
    sizes = sizes[ordre]
    errors_exactsvd = np.asarray(errors_exactsvd)[ordre]
    errors_rsvd_gauss = np.asarray(errors_rsvd_gauss)[ordre]
    errors_rsvd_uni = np.asarray(errors_rsvd_uni)[ordre]
    errors_rsvd_col = np.asarray(errors_rsvd_col)[ordre]
    duration_exactsvd = np.asarray(duration_exactsvd)[ordre]
    duration_rsvd_gauss = np.asarray(duration_rsvd_gauss)[ordre]
    duration_rsvd_uni = np.asarray(duration_rsvd_uni)[ordre]
    duration_rsvd_col = np.asarray(duration_rsvd_col)[ordre]
    # Plot compressed and original image

    errors = [errors_exactsvd, errors_rsvd_gauss, errors_rsvd_uni, errors_rsvd_col]
    durations = [duration_exactsvd, duration_rsvd_gauss, duration_rsvd_uni, duration_rsvd_col]
        
    plt.figure(figsize=(10, 6))
    plt.title("SVD error with matrix approximation of rank K = {}".format(k))
    plt.plot(sizes, errors[0], label = "truncated SVD")
    plt.plot(sizes, errors[1], linestyle = "solid", label = "Random SVD (Gauss)")
    plt.plot(sizes, errors[2], linestyle = "dashed", label = "Random SVD (Uniform)")
    plt.plot(sizes, errors[3], linestyle = "dotted", label = "Random SVD (Col sampling)")
    plt.xlabel("log(Size of Square Matrix N)")
    plt.ylabel("Reconstruction error of svd")
    plt.xscale("log")
    plt.grid()
    plt.legend()

    plt.figure(figsize=(10, 6))
    plt.title("SVD compute duration")
    plt.plot(sizes, durations[0], label = "truncated SVD")
    plt.plot(sizes, durations[1], linestyle="solid", label = "Random SVD (Gauss)")
    plt.plot(sizes, durations[2], linestyle="dashed", label = "Random SVD (Uniform)")
    plt.plot(sizes, durations[3], linestyle="dotted", label = "Random SVD (Col sampling)")
    plt.legend()
    plt.yscale("log")
    plt.xlabel("Size of Square Matrix N")
    plt.ylabel("log(Compute time)")
    plt.grid()
    plt.legend()
    
    plt.show()

