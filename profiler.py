import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
from numpy.random.mtrand import randint
from math import ceil, log10
from time import perf_counter
from utils import *
from svd_func import *


#RAJOUTER LERREUR / ERREUR TOTALE
def execute_t_times(t, svd_func, A, k, kernel, verbose=1):
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
        A_tilde = svd_func(A, m_k, kernel=kernel) #compute A_tild using randomized svd
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
    """ Profiles error and duration for fixed rank random matrices

    Args:
        sizes (list of int) : Sizes to test the different
        k (int): number max of columns to use in the randomized svd and the rank of the matrix

    Returns:
        null : used only for plotting
    """  
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
        errors_rsvd_gauss.append(np.linalg.norm(r_svd(M, k, kernel="gaussian") - M))
        e = perf_counter()
        duration_rsvd_gauss.append(e - s)
        
        s = perf_counter()
        errors_rsvd_uni.append(np.linalg.norm(r_svd(M, k, kernel="uniform") - M))
        e = perf_counter()
        duration_rsvd_uni.append(e - s)
        
        s = perf_counter()
        errors_rsvd_col.append(np.linalg.norm(r_svd(M, k, kernel="colsampling") - M))
        e = perf_counter()
        duration_rsvd_col.append(e - s)

        

    # Plot compressed and original image

    errors = [errors_exactsvd, errors_rsvd_gauss, errors_rsvd_uni, errors_rsvd_col]#, errors_rsvd_SRHT]
    durations = [duration_exactsvd, duration_rsvd_gauss, duration_rsvd_uni, duration_rsvd_col]#, duration_rsvd_SRHT]
        
    plt.figure()
    plt.title("SVD error with matrix approximation of rank K = {}".format(k))
    plt.plot(sizes, errors[0], label = "exact SVD")
    plt.plot(sizes, errors[1], label = "Random SVD (Gauss)")
    plt.plot(sizes, errors[2], label = "Random SVD (Uniform)")
    plt.plot(sizes, errors[3], label = "Random SVD (Col sampling)")
    #plt.plot(sizes, errors[4], label = "Random SVD (SRHT)")
    plt.xlabel("Size of Square Matrix N")
    plt.ylabel("Reconstruction error of svd")
    plt.grid()
    plt.legend()

    plt.figure()
    plt.title("SVD compute duration")
    plt.plot(sizes, durations[0], label = "exact SVD")
    plt.plot(sizes, durations[1], label = "Random SVD (Gauss)")
    plt.plot(sizes, durations[2], label = "Random SVD (Uniform)")
    plt.plot(sizes, durations[3], label = "Random SVD (Col sampling)")
    #plt.plot(sizes, durations[4], label = "Random SVD (SRHT)")
    plt.xlabel("Size of Square Matrix N")
    plt.ylabel("Compute time")
    plt.grid()
    plt.legend()
    
    plt.show()

    return 

def cache_svd_photo(path, M, k):
    """ caches the SVD to avoid computing it multiple times

    Args:
        path (string) : path of image
        M (numpy 2D array) : Matrix
        k (int): number of columns to use in truncated svd

    Returns:
        null : matrix error and compute duration
    """  
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
    """ Launches r-svd for for different photos and plots errors and durations

    Args:
        paths (list of string) : paths of images
        k (int): number of k to keep in randomized svd

    Returns:
        null : used only for plotting
    """   
    duration_exactsvd = []
    duration_rsvd_gauss = []
    duration_rsvd_uni = []
    duration_rsvd_col = []
    duration_rsvd_SRHT = []
    duration_rsvd_DCT = []
    
    errors_exactsvd = []
    errors_rsvd_gauss = []
    errors_rsvd_uni = []
    errors_rsvd_col = []
    errors_rsvd_SRHT = []
    errors_rsvd_DCT = []

    sizes = []
    size_text_labels = []
    svd_ratio = []
    for path in paths:
        M = np.asarray(toGrayScale(getColouredImage(path)))
        shape = M.shape
        if shape[0] < shape[1]:
            M = M.transpose()
        sizes.append(shape[0] * shape[1])
        size_text_labels.append(str(int(shape[0]/1000)) + "kx" + str(int(shape[1]/1000)) + "k")

        error, duration = cache_svd_photo(path, M, k)
        errors_exactsvd.append(error)
        duration_exactsvd.append(duration)
        svd_ratio.append(error / (shape[0] * shape[1]))

        s = perf_counter()
        errors_rsvd_gauss.append(np.linalg.norm(r_svd(M, k, kernel="gaussian") - M))
        e = perf_counter()
        duration_rsvd_gauss.append(e - s)
        
        s = perf_counter()
        errors_rsvd_uni.append(np.linalg.norm(r_svd(M, k, kernel="uniform") - M))
        e = perf_counter()
        duration_rsvd_uni.append(e - s)
        
        s = perf_counter()
        errors_rsvd_col.append(np.linalg.norm(r_svd(M, k, kernel="colsampling") - M))
        e = perf_counter()
        duration_rsvd_col.append(e - s)

        s = perf_counter()
        errors_rsvd_SRHT.append(np.linalg.norm(r_svd(M, k, kernel="SRHT") - M))
        e = perf_counter()
        duration_rsvd_SRHT.append(e - s)

        s = perf_counter()
        errors_rsvd_DCT.append(np.linalg.norm(r_svd(M, k, kernel="DCT") - M))
        e = perf_counter()
        duration_rsvd_DCT.append(e - s)
        
        

    sizes = np.asarray(sizes)
    ordre = np.argsort(sizes)
    sizes = sizes[ordre]
    size_text_labels = np.asarray(size_text_labels)[ordre]
    errors_exactsvd = np.asarray(errors_exactsvd)[ordre]
    errors_rsvd_gauss = np.asarray(errors_rsvd_gauss)[ordre] / errors_exactsvd
    errors_rsvd_uni = np.asarray(errors_rsvd_uni)[ordre] / errors_exactsvd
    errors_rsvd_col = np.asarray(errors_rsvd_col)[ordre] / errors_exactsvd
    errors_rsvd_SRHT = np.asarray(errors_rsvd_SRHT)[ordre] / errors_exactsvd
    errors_rsvd_DCT = np.asarray(errors_rsvd_DCT)[ordre] / errors_exactsvd
    duration_exactsvd = np.asarray(duration_exactsvd)[ordre]
    duration_rsvd_gauss = np.asarray(duration_rsvd_gauss)[ordre]
    duration_rsvd_uni = np.asarray(duration_rsvd_uni)[ordre]
    duration_rsvd_col = np.asarray(duration_rsvd_col)[ordre]
    duration_rsvd_SRHT = np.asarray(duration_rsvd_SRHT)[ordre]
    duration_rsvd_DCT = np.asarray(duration_rsvd_DCT)[ordre]
    # Plot compressed and original image

    errors = [np.ones(shape=errors_exactsvd.shape), errors_rsvd_gauss, errors_rsvd_uni, \
        errors_rsvd_col, errors_rsvd_SRHT, errors_rsvd_DCT]
    durations = [duration_exactsvd, duration_rsvd_gauss, duration_rsvd_uni, duration_rsvd_col, \
        duration_rsvd_SRHT, duration_rsvd_DCT]
        
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.title("R-SVD error with matrix approximation of rank K = {}, \
        with respect to truncated SVD error".format(k))
    plt.plot(sizes, errors[0], label = "truncated SVD")
    plt.plot(sizes, errors[1], linestyle = "solid", label = "Random SVD (Gauss)")
    plt.plot(sizes, errors[2], linestyle = "dashed", label = "Random SVD (Uniform)")
    plt.plot(sizes, errors[3], linestyle = "dashed", label = "Random SVD (Col sampling)")
    plt.plot(sizes, errors[4], linestyle = "dotted", label = "Random SVD (SRHT)")
    plt.plot(sizes, errors[5], linestyle = "dotted", label = "Random SVD (DCT)")

    for i, txt in enumerate(size_text_labels):
        ax.annotate(txt, (sizes[i], np.max(np.asarray(errors)[:, i])))

    plt.xlabel("log(Size of Matrix N)")
    plt.ylabel("Reconstruction error of r-svd respective to truncated svd")
    plt.xscale("log")
    plt.grid()
    plt.legend()

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    plt.title("SVD compute duration")
    plt.plot(sizes, durations[0], label = "truncated SVD")
    plt.plot(sizes, durations[1], linestyle="solid", label = "Random SVD (Gauss)")
    plt.plot(sizes, durations[2], linestyle="dashed", label = "Random SVD (Uniform)")
    plt.plot(sizes, durations[3], linestyle="dashed", label = "Random SVD (Col sampling)")
    plt.plot(sizes, durations[4], linestyle="dotted", label = "Random SVD (SRHT)")
    plt.plot(sizes, durations[5], linestyle="dotted", label = "Random SVD (DCT)")
    
    for i, txt in enumerate(size_text_labels):
        ax2.annotate(txt, (sizes[i], np.max(np.asarray(durations)[:, i])))

    plt.legend()
    plt.yscale("log")
    plt.xlabel("Size of Matrix N")
    plt.ylabel("log(Compute time)")
    plt.grid()
    plt.legend()
    
    plt.show()



def fixed_rank_error_power_iteration(path, k=100):
    """ Plots the effect of power iteration in relative error

    Args:
        path (list of string) : path of image
        k (int) : number of k to keep in randomized svd

    Returns:
        null : used only for plotting
    """   
    errors_rsvd_gauss = []
    errors_rsvd_uni = []
    errors_rsvd_cols = []
    errors_rsvd_DCT = []
    errors_rsvd_SRHT = []

    svd_ratio = []
    
    M = np.asarray(toGrayScale(getColouredImage(path)))
    shape = M.shape
    if shape[0] < shape[1]:
        M = M.transpose()

    error, _ = cache_svd_photo(path, M, k)
    svd_ratio.append(error / (shape[0] * shape[1]))

    for i in range(0, 3):
        errors_rsvd_gauss.append(np.linalg.norm(r_svd(M, k, kernel="gaussian", power_iteration=i) - M) / error)
        errors_rsvd_uni.append(np.linalg.norm(r_svd(M, k, kernel="uniform", power_iteration=i) - M) / error)
        errors_rsvd_cols.append(np.linalg.norm(r_svd(M, k, kernel="colsampling", power_iteration=i) - M) / error)
        errors_rsvd_DCT.append(np.linalg.norm(r_svd(M, k, kernel="DCT", power_iteration=i) - M) / error)
        errors_rsvd_SRHT.append(np.linalg.norm(r_svd(M, k, kernel="SRHT", power_iteration=i) - M) / error)
        
        
    plt.subplots(figsize=(10, 6))
    plt.plot(range(3), np.ones(shape=np.arange(3).shape), label="Truncated SVD")
    plt.plot(range(3), errors_rsvd_gauss, linestyle="dashed", label="R-SVD (Gaussian)")
    plt.plot(range(3), errors_rsvd_uni, linestyle="dashed", label="R-SVD (Uniform)")
    plt.plot(range(3), errors_rsvd_cols, linestyle="dashed", label="R-SVD (Colsampling)")
    plt.plot(range(3), errors_rsvd_DCT, linestyle="dotted", label="R-SVD (DCT)")
    plt.plot(range(3), errors_rsvd_SRHT, linestyle="dotted", label="R-SVD (SRHT)")
    plt.legend()
    plt.title("Power iteration effect on respective error for q in {1, 2, 3}")
    plt.xlabel("Power iteration q ")
    plt.ylabel("Respective error with truncated SVD")
    plt.grid()
    plt.legend()
    
    plt.show()