import numpy as np

def fft(image):
    if (len(image.shape) > 2):
        spectrum = np.fft.fftshift(np.fft.fft2(image),axes=(1,2))
    else:
        spectrum = np.fft.fftshift(np.fft.fft2(image))
    return spectrum

def ifft(spectrum):
    if (len(spectrum.shape) > 2):
        image = np.fft.ifft2(np.fft.ifftshift(spectrum,axes=(1,2)))
    else:
        image = np.fft.ifft2(np.fft.ifftshift(spectrum))
    return image

def max_value_index(matrix):
    '''Returns a tuple which are the index of the max of the matrix'''
    
    # flatten matrix
    maxElement = np.amax(np.ravel(matrix))
    x, y = np.where(matrix == maxElement)
    return x[0], y[0]

def pad_image(image, target_width, pad_value):
    size = image.shape[0]
    pad_size = (target_width - size)//2
    pad_image = np.pad(image, pad_size, 'constant', constant_values=pad_value)
    return pad_image