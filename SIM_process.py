import numpy as np
import common_functions as common

from scipy.signal import correlate

def pad_image_sequence(image, target_width, pad_value):
    # image - expected to have shape (9, size, size) where size is the dimesion of the a single raw image
    size = image.shape[1]
    pad_width = (target_width - size)//2
    padded_image =  np.repeat(np.ones((target_width,target_width), dtype=np.complex128)[np.newaxis], 9, axis=0) * pad_value
    padded_image[:,pad_width:-pad_width,pad_width:-pad_width] = image
    return padded_image

#################### Phase ####################

# Wicker 2013 Autocorrelation


def separate_freq_comp(spectrum, mod, phases):
    # in:
    # spectrum - spectrum of raw SIM images (array of 9 images)
    # mod - modulation factor
    # phases - phases in each frequency
    size = spectrum.shape[1]
    M = 0.5 * np.array([[1, (mod/2)*np.exp(1j*phases[0]), (mod/2)*np.exp(-1j*phases[0])],
                        [1, (mod/2)*np.exp(1j*phases[1]), (mod/2)*np.exp(-1j*phases[1])],
                        [1, (mod/2)*np.exp(1j*phases[2]), (mod/2)*np.exp(-1j*phases[2])]], dtype=np.complex128)

    M_inv = np.linalg.inv(M)
    
    sep_comp = np.repeat(np.zeros((size,size), dtype=np.complex128)[np.newaxis], 9, axis=0)
    
    for t in range(3):
        sep_comp[3*t] = M_inv[0,0] * spectrum[3*t] + M_inv[0,1] * spectrum[3*t+1] + M_inv[0,2] * spectrum[3*t+2] # S(k)H(k)
        sep_comp[3*t+1] = M_inv[1,0] * spectrum[3*t] + M_inv[1,1] * spectrum[3*t+1] + M_inv[1,2] * spectrum[3*t+2] # S(k - p)H(k)
        sep_comp[3*t+2] = M_inv[2,0] * spectrum[3*t] + M_inv[2,1] * spectrum[3*t+1] + M_inv[2,2] * spectrum[3*t+2] # S(k + p)H(k)
        
    return sep_comp

def shift_spectrum(spectrum, freq, angles, round_factor = 100):
    # in:
    # sprectrum - spectrum to be shifted
    # freq - amount to be shifted by
    # angles - direction to be shift
    
    # out:
    # spectrum_s - shifted spectrum
    
    size = spectrum.shape[1]
    hsize = size//2
    n = np.arange(0, size, 1)
    x, y = np.meshgrid(n, n)
    
    an = np.repeat(angles, 3, axis=0) # [0 60 120]
    
    p = freq/size
    k = p * np.array([np.cos(an), np.sin(an)]).transpose()
    
    
    spectrum_s = np.repeat(np.zeros((size,size), dtype=np.complex128)[np.newaxis], 9, axis=0)
    
    for t in range(3):
        spectrum_s[3*t] = spectrum[3*t] #S(k)H(k - p)
        spectrum_s[3*t+1] = np.fft.fft2(np.fft.ifft2(spectrum[3*t+1]) * np.exp(-1j * 2 * np.pi * (k[3*t+1,0] * x + k[3*t+1,1] * y))) #S(k)H(k - p)
        spectrum_s[3*t+2] = np.fft.fft2(np.fft.ifft2(spectrum[3*t+2]) * np.exp(1j * 2 * np.pi * (k[3*t+2,0] * x + k[3*t+2,1] * y))) #S(k)H(k + p)
    
    return spectrum_s

def raised_cosine_filter(size, signal_freq, fvector, angles):
    nyquist_freq = 2*signal_freq
    excess_freq = size - nyquist_freq
    #beta = excess_freq/nyquist_freq
    beta = 2*(1/size)*excess_freq
    angle_arr = fvector * np.repeat(angles, 3, axis=0)
    sign_factor = np.array([1, 1, -1, 1, 1, -1, 1, 1, -1])
    print(angle_arr)
    RCF = np.repeat(np.zeros((size,size),dtype=np.complex128)[np.newaxis], 9, axis=0)
    n = np.arange(0, size, 1)

    for t in range(9):
        xshift = fvector * np.cos(angle_arr[t])
        yshift = fvector * np.sin(angle_arr[t])
        if (t%3 == 0):
            print("I'm here")
            # Don't shift the center of RC filter
            x, y = np.meshgrid(np.abs(n - size//2),np.abs(n - size//2))
            #print_image(x,0)
            #print_image(y,0)
            r = np.sqrt(np.power(x,2) + np.power(y,2))
            #print_image(r)
        elif (t%3 == 2):
            # Shift to the right
            x, y = np.meshgrid(np.abs(n - size//2 + sign_factor[t]*xshift),np.abs(n - size//2 + sign_factor[t]*yshift))
            r = np.sqrt(np.power(x,2) + np.power(y,2))
        elif (t%3 == 1):
            # Shift to the left
            x, y = np.meshgrid(np.abs(n - size//2 + sign_factor[t]*xshift),np.abs(n - size//2 + sign_factor[t]*yshift))
            r = np.sqrt(np.power(x,2) + np.power(y,2))
    
        RCF[t,r <= (1-beta)*signal_freq/2] = 1
    
        for v in range(size):
            for u in range(size):
                if (r[v,u] > (1-beta)*signal_freq/2) and (r[v,u] <= (1+beta)*signal_freq/2):
                    RCF[t,v,u] = 0.5*(1 + np.cos( (np.pi/(signal_freq*beta)) * (r[v,u] - (1-beta)*signal_freq/2) ))
    return RCF

def resize_otf(otf, new_size):
    pad_size = (new_size - otf.shape[1]) // 2
    return np.fft.ifftshift(np.pad(np.fft.fftshift(otf), pad_size, 'edge'))
    

def wiener_filter(spectrum, Hc, Ha, w = 0):
    # in:
    # spectrum - spectrum to be filtered
    # Hc - conjugate of the OTF
    # Ha = absolute of the OTF
    # w = wiener parameter
    
    # out: wiener-filtered spectrum
    
    wiener_num = (Hc * spectrum).sum(axis=0)
    wiener_denom = np.power(Ha,2).sum(axis=0) + w**2
    return wiener_num / wiener_denom

def wiener_filter_per_component(spectrum, Hc, Ha, w=0):
    top = Hc * spectrum
    bottom = np.power(Ha,2) + w**2
    return top/bottom
    

def apodization(size, threshold, fvector, angles):
    ADF_s = np.repeat(np.zeros((size,size),dtype=np.complex128)[np.newaxis], angles.size * 3, axis=0)
    ADF = np.zeros((size,size),dtype=np.complex128)
    n = np.arange(0, size, 1)
    for t in range(angles.size * 3):
        # Create blank canvas for apod_fun
        
        # Create the vector for angles
        angle_arr = np.repeat(angles, 3, axis=0)
        xshift = fvector * np.cos(angle_arr[t])
        yshift = fvector * np.sin(angle_arr[t])
        if (t%3 == 0):
            x, y = np.meshgrid(np.abs(n - size//2),np.abs(n - size//2))
            r = np.sqrt(np.power(x,2) + np.power(y,2))
            #ADF[r < threshold] = 1
            ADF_s[t,r < threshold] = 1
        elif (t%3 == 1):
            # Shift to the right
            x, y = np.meshgrid(np.abs(n - size//2 + xshift),np.abs(n - size//2 + yshift))
            r = np.sqrt(np.power(x,2) + np.power(y,2))
            #ADF[r < threshold] = 1
            ADF_s[t,r < threshold] = 1
        elif (t%3 == 2):
            # Shift to the left
            x, y = np.meshgrid(np.abs(n - size//2 - xshift),np.abs(n - size//2 - yshift))
            r = np.sqrt(np.power(x,2) + np.power(y,2))
            #ADF[r < threshold] = 1
            ADF_s[t,r < threshold] = 1
        
    return ADF_s
    