import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from math import ceil, log10

figsize=(6866/500, 2386/500)
# figsize = (6946/700,3906/700)
# figsize = (12000/1000,14000/1000) ##fat

def getColouredImage(path):
    img_colour = Image.open(path)
    plt.figure(figsize=figsize)
    plt.imshow(img_colour)
    plt.title("Original coloured image")
    return img_colour

def toGrayScale(img):
    img_bw = img.convert('L')
    # plt.figure(figsize=figsize)
    # plt.imshow(img_bw, cmap='gray')
    # plt.title("Black and white image")    
    # print("Image array shape: {}".format(img_bw.size))
    return img_bw

def gaussianMatrixGenerator(m, n, param = (0, 1)):
    omega = []
    for i in range(n):
        omega.append(np.random.normal(param[0], param[1], size=m))
    omega = np.asarray(omega).transpose()
    shape = omega.shape
    if ((shape[0] == 1) or (shape[1] == 1)):
        omega = omega.flatten()
    return omega

def rankk_random_matrix_generator(m, n, rank):
    mat = np.empty((0,m))
    for i in range(rank):
        mat = np.append(mat, [np.random.normal(100, 75, size=m)], axis=0)
        
    for i in range(rank, n):
        shape = mat.shape
        w = np.random.uniform(low = 0, high = 1, size = shape[1])
        mat = np.append(mat, [w + np.random.normal(0, 0.00025, size=m)], axis=0)

    return np.asarray(mat).transpose()

def compute_error(A, A_tilde):
    """Computes the exact error and RMS' magnitude for two matrices. 

    Args:
        A (np.array): The original matrix 
        A_tilde (np.array): The approximated matrix with which we want to compute the error

    Returns:
        rsse : the exact error overall for the matrix (summ of each component's errors)
        rmse magnitude: the exact error divided by the size of the matrix, to the log10
    """    
    m,n  = A.shape
    size = m*n
    exact_error = abs(np.linalg.norm(A - A_tilde))
    rms_magnitude = ceil(log10(exact_error/size))
    return exact_error, rms_magnitude

def print_result(error, magnitude, duration, name):
    print("\n####################################################################")
    print(name)
    print(">>> Duration : {:.5f} sec.".format(duration))
    print(">>> Rooted mean squared Error (RMSE) : 10e" + str(magnitude))
    print(">>> Rooted sum squared Erros (RSSE)  : " + str(error))
    print("####################################################################")
    
    return


