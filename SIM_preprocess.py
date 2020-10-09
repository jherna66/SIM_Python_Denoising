import numpy as np
from scipy import signal
from scipy import special
import common_functions as common

def PSF_OTF(size, a, style):
    # Mesh
    n = np.arange(0, size, 1)
    x, y = np.meshgrid(np.fft.ifftshift(n - size//2),np.fft.ifftshift(n - size//2))
    r = np.sqrt(np.power(np.minimum(x, np.abs(x - size)),2) + np.power(np.minimum(y, np.abs(y - size)),2))
    eps = np.finfo(np.float64).eps # So that zero does not produce NaN
    R = r + eps
    if style == 'gaussian':
        sigma_sq = np.power(1.25/a,2)
        psf = np.exp(-np.power((r),2)/(2*sigma_sq))
        otf = np.fft.fft2(psf)
    elif style == 'bessel':
        psf = np.power(np.abs(2 * special.jv(1, a*R) / R),2)
        otf = np.fft.fft2(psf)
    elif style == 'butterworth':
        fs = size//2
        pre_otf = np.zeros((size,size))
        pre_otf[r < 175] = 1
        b, t = signal.butter(1, 10/fs, 'lowpass')
        otf = signal.filtfilt(b, t, pre_otf)
        otf = signal.filtfilt(b, t, otf.transpose())
        psf = np.abs(np.fft.ifft2(otf))
    elif style=='delta':
        psf = np.zeros((size,size))
        psf[0,0] = 1
        otf = np.fft.fft2(psf)
    scale = np.amax(np.abs(otf))
    otf = otf/scale
    psf = psf/scale
    
    return np.fft.fftshift(psf), np.fft.fftshift(otf)

def OTF_butterworth(size, a):
    fs = size//2
    n = np.arange(0, size, 1)
    x, y = np.meshgrid(np.fft.ifftshift(n - size//2),np.fft.ifftshift(n - size//2))
    r = np.sqrt(np.power(np.minimum(x, np.abs(x - size)),2) + np.power(np.minimum(y, np.abs(y - size)),2))
    circle = np.zeros((size,size))
    circle[r < a] = 1
    # make filter
    b, a = signal.butter(1, 10/fs, 'lowpass')
    w, h = signal.freqs(b, a)
    y = signal.filtfilt(b, a, np.fft.fftshift(circle))
    y = signal.filtfilt(b, a, y.transpose())
    return y
    
# Widefield Image
def WF_image(image, otf):
    image_FT = common.fft(image)
    widefield_FT = image_FT * otf
    widefield = common.ifft(widefield_FT)
    return widefield, widefield_FT

# Raw SIM images
def raw_SIM_images(image, otf, mod, freq, angles, phases):
    # modulation
    size = image.shape[0]
    n = np.arange(0, size, 1)
    x, y = np.meshgrid(n, n)
    
    an = np.repeat(angles, 3, axis=0) # [0 60 120]
    ph = np.tile(phases, 3) # [0 120 240]
    I_0 = 0.5 # mean value of illumination
    
    
    p = freq/size
    k = p * np.array([np.cos(an), np.sin(an)]).transpose()
    
    i = np.repeat(np.zeros((size,size))[np.newaxis], 9, axis=0)
    for t in range(9):
        i[t] = I_0 * (1 + mod * np.cos(2 * np.pi * (k[t,0] * x + k[t,1] * y) + ph[t]))
    
    image_SI = i * image
    image_SI_FT = common.fft(image_SI)
    rawSIM_FT = image_SI_FT * otf
    rawSIM = common.ifft(rawSIM_FT)
    return np.abs(rawSIM), k

# Raw SIM images
def raw_SIM_images_w_noise(image, otf, mod, freq, angles, phases, noise):
    # modulation
    size = image.shape[0]
    n = np.arange(0, size, 1)
    x, y = np.meshgrid(n, n)
    
    an = np.repeat(angles, 3, axis=0) # [0 60 120]
    ph = np.tile(phases, 3) # [0 120 240]
    I_0 = 0.5 # mean value of illumination
    
    
    p = freq/size
    k = p * np.array([np.cos(an), np.sin(an)]).transpose()
    
    i = np.repeat(np.zeros((size,size))[np.newaxis], 9, axis=0)
    for t in range(9):
        i[t] = I_0 * (1 + mod * np.cos(2 * np.pi * (k[t,0] * x + k[t,1] * y) + ph[t]))
    
    image_SI = i * image
    image_SI_FT = common.fft(image_SI)
    rawSIM_FT = image_SI_FT * otf
    rawSIM = common.ifft(rawSIM_FT)
    return np.abs(rawSIM).astype(np.uint16) + np.random.normal(0,noise,(size,size)), k
