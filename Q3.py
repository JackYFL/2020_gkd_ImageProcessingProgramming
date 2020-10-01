import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def conv2d(padimg, kernel):
    """
    Implement 2dconv
    :param padimg: The padding image
    :param kernel: The conv kernel
    :return: Image after conv
    """
    w, h = kernel.shape
    W, H = padimg.shape
    convimg = np.zeros((W - 2 * int(w / 2), H - 2 * int(h / 2)))
    for ii in range(int(w / 2), W - int(w / 2)):
        for jj in range(int(h / 2), H - int(h / 2)):
            window = padimg[(ii - int(w / 2)):(ii + int(w / 2) + 1), (jj - int(h / 2)):(jj + int(h / 2) + 1)] * kernel
            convimg[ii - int(w / 2), jj - int(h / 2)] = window.sum()
    return convimg


def twodConv(f, kernel, padding='zero'):
    """
    2d conv for f
    :param f: The gray image
    :param kernel: conv kernel
    :return: g: output image
    """
    img = f
    W, H = img.shape
    w, h = kernel.shape
    w_pad = int((w - 1) / 2)
    h_pad = int((h - 1) / 2)
    if padding == 'zero':
        img_pad = np.zeros((W + 2 * w_pad, H + 2 * h_pad))
        img_pad[w_pad:w_pad + W, w_pad:w_pad + H] = img
        img_pad = np.array(img_pad, dtype=np.uint8)
        img_conv = conv2d(img_pad, kernel)
        return img_conv
    elif padding == 'replicate':  # if the padding mode is replicate
        img_pad = np.pad(img, ((w_pad, w_pad), (h_pad, h_pad)), 'edge')
        img_pad = np.array(img_pad, dtype=np.uint8)
        img_conv = conv2d(img_pad, kernel)
        return img_conv
    else:
        print('The padding mode is invalid, please input again!')


if __name__ == '__main__':
    img = np.arange(1, 7).reshape(2, 3)
    w = np.ones((5, 5))*0.1
    conv1 = twodConv(f=img, kernel=w, padding='replicate')
    conv2 = twodConv(f=img, kernel=w)
    print(conv1)
    print(conv2)
