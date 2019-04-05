"""
Simple functions for my Fourier notebooks.
"""


# From notebook 3

def ft1d(func):
    ft = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(func)))
    return ft


def ift1d(func):
    ift = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(func)))
    return ift
