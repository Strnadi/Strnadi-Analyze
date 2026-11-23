"""Random smooth spectral gain curve (+-6 dB knots plus a tilt): a cheap stand-in for unknown microphone /
device frequency responses. Origin: research session (`ssl_pretrain.py::rand_eq`), used for SSL views."""
import numpy as np


def random_eq(y):
    Y = np.fft.rfft(y); f = np.linspace(0, 1, len(Y))
    g_db = np.interp(f, np.linspace(0, 1, 6), np.random.uniform(-6, 6, 6)) + np.random.uniform(-6, 6) * (f - 0.5)
    return np.fft.irfft(Y * 10 ** (g_db / 20), n=len(y)).astype(np.float32)
