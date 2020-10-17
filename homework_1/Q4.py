#coding:utf8
import numpy as np


def gaussKernel(sig, m=0):
    """
    Gaussian kernel implementation
    :param sig: sigma parameter in Gaussian
    :param m: kernel size
    :return: Gaussian kernel
    """
    if m < 3 and m != 0:
        print("The kernel size is small, please input again!")
    elif m == 0:
        m = int(np.ceil(6 * sig)+1)
        if m % 2 == 0:
            m += 1
        kernel = np.zeros((m, m))
        half_size = int(m / 2)
        for i in range(-half_size, half_size + 1):
            for j in range(-half_size, half_size + 1):
                kernel[i + half_size, j + half_size] = np.exp(-(i ** 2 + j ** 2) / (2 * sig * sig))
        kernel /= kernel.sum()
        return kernel
    else:
        kernel = np.zeros((m, m))
        half_size = int(m / 2)
        for i in range(-half_size, half_size + 1):
            for j in range(-half_size, half_size + 1):
                kernel[i + half_size, j + half_size] = np.exp(-(i ** 2 + j ** 2) / (2 * sig * sig))
        kernel /= kernel.sum()
        return kernel


if __name__ == '__main__':
    kernel = gaussKernel(1)
    print(kernel)
