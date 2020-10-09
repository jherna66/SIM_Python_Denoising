## Display functions

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def print_image(image, mod=0):
    dpi = 80 # dots per inch
    # show image
    height = int(image.shape[0])
    width = int(image.shape[1])
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    if mod == 1:
        plt.imshow(np.log(np.abs(image)), cmap="gray")
    else:
        plt.imshow(np.absolute(image), cmap="gray")
    #plt.axis('off')
    plt.axis('equal')
    plt.show()
    
def print_raw_image(image):
    dpi = 80 # dots per inch
    # show image
    figsize = image.shape[1] / float(dpi), image.shape[0] / float(dpi)
    fig = plt.figure(figsize=figsize)
    plt.imshow(image, cmap="gray")
    plt.axis('off')
    plt.axis('equal')
    plt.show()

def print_mesh(image, mod=1):
    # Mesh
    d = image.shape[0]
    n = np.arange(0, d, 1)
    x, y = np.meshgrid(n, n)
    if mod == 1:
        fig1 = plt.figure()
        ax1 = fig1.gca(projection='3d')
        ax1.plot_surface(x, y, image, cmap='plasma')
        #fig2 = plt.figure()
        #ax2 = fig2.gca(projection='3d')
        #ax2.plot_surface(nx, ny, np.power(np.log(np.absolute(np.fft.fftshift(np.imag(image)))),2), cmap='plasma')
    else:
        fig1 = plt.figure()
        ax1 = fig1.gca(projection='3d')
        ax1.dist = 11
        ax1.plot_surface(x, y, np.real(image), cmap='plasma')
        fig2 = plt.figure()
        ax2 = fig2.gca(projection='3d')
        ax2.dist = 11
        ax2.plot_surface(x, y, np.imag(image), cmap='plasma')
    plt.show()
    
def print_plot(image, mod=1):
    size = int(image.shape[0])
    peaks = np.flip(np.fft.fftshift(image[size//2 + 1]).argsort())[0:3]
    print('Peaks = 1st: ', (size/2) - peaks[0], ', 2nd: ', (size/2) - peaks[1], ', 3rd: ', (size/2) - peaks[2])
    x = np.linspace(0,size-1,size, dtype=np.uint)
    r = (size/2)-x
    fig = plt.figure()
    if mod == 1:
        plt.plot(r, np.log(np.abs(image[size//2 + 1])**2))
    else:
        plt.plot(r, np.absolute(image[size//2 + 1]))
    plt.axis('tight')
    plt.show()
    
def print_raw_plot(image, mod=1):
    size = int(image.shape[0])
    peaks = np.flip(image[0]).argsort()[0:3]
    print('Peaks = 1st: ', (size/2) - peaks[0], ', 2nd: ', (size/2) - peaks[1], ', 3rd: ', (size/2) - peaks[2])
    x = np.linspace(0,image.shape[0]-1,size, dtype=np.uint)
    r = (size/2)-x
    fig = plt.figure()
    plt.plot(r, np.fft.fftshift(image[0]))
    