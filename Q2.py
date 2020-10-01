import cv2 as cv
import os
import numpy as np
from matplotlib import pyplot as plt

def rgb1gray(f, method='NTFC',showflag=True):
    """
    Turn rgb img to gray img
    :param f: image path
    :param method: 'average' or 'NTFC'(defaut)
    return gray image
    """
    if os.path.exists(f):
        rgbimg = cv.imread(f)
        rgbimg=np.array(rgbimg,dtype=np.float32)
        if method == 'average':
            grayimg = (rgbimg[:, :, 0] + rgbimg[:, :, 1] + rgbimg[:, :, 2]) / 3
            grayimg = np.around(grayimg)
            grayimg = np.array(grayimg,dtype=np.uint8)
            if showflag:
                plt.figure()
                plt.imshow(grayimg,'gray')
                plt.title('Average')
            return grayimg
        elif method == 'NTFC':
            grayimg = (0.2989 * rgbimg[:, :, 0] + 0.5870 * rgbimg[:, :, 1] + 0.1140 * rgbimg[:, :, 2])
            grayimg = np.around(grayimg)
            grayimg = np.array(grayimg,dtype=np.uint8)
            if showflag:
                plt.figure()
                plt.imshow(grayimg,'gray')
                plt.title('NTFC')
            return grayimg
        else:
            print('The method parameter is invalid, please input again!')
    else:
        print('Imgae path is invalid, please input again!')


if __name__ == '__main__':
    rgb1gray(f='mandril_color.tif', method='average')
    rgb1gray(f='mandril_color.tif')
    rgb1gray(f='lena512color.tiff', method='average')
    rgb1gray(f='lena512color.tiff')
    plt.show()
