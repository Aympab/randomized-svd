import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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