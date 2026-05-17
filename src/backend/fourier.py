# fourier.py manages the mathematical backend.
# receive audio input (in bit64) -> obtain FFT -> return the FFT result (in bit64)

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

def fourier_transform(img):
    """
    receives a 2D image (in bit64) and returns its Fourier Transform (in bit64).
    """
    img_array = np.array(img, dtype=np.float64)

    # fourier Transform
    fourier_result = np.fft.fft2(img_array)

    # shift the zero frequency component to the center of the spectrum
    fourier_result_shifted = np.fft.fftshift(fourier_result)

    return fourier_result_shifted

def signal_sample(img):
    """
    samples a 2D image in the time domain for appropiate
    frequency analysis. This is done by applying a window function to the image.
    """
    img_array = np.array(img, dtype=np.float64)

    # applying a Hanning window to the image
    window = np.hanning(img_array.shape[0])[:, None] * np.hanning(img_array.shape[1])
    sampled_img = img_array * window

    return sampled_img

def tuning(freq, fft):
    """
    receives a frequency and a Fourier Transform (in bit64), thus
    comparing the magnitude of the Fourier Transform at the given frequency to a threshold value. If the magnitude exceeds the threshold,
    it indicates that the frequency is present in the signal (in-tune!).
    """
    # Calculate the magnitude of the Fourier Transform
    magnitude = np.abs(fft)

    # Define a threshold value (this can be adjusted based on your requirements)
    threshold = 0.1 * np.max(magnitude)

    # Check if the magnitude at the given frequency exceeds the threshold
    if magnitude[freq] > threshold:
        return True  # Frequency is present (in-tune)
    else:
        return False  # Frequency is not present (out-of-tune)