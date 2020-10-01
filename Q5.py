from Q2 import rgb1gray
from Q3 import twodConv
from Q4 import gaussKernel
from matplotlib import pyplot as plt
import cv2 as cv

if __name__ == '__main__':
    cameraman_img = rgb1gray(f='cameraman.tif', showflag=False)
    einstein_img = rgb1gray(f='einstein.tif', showflag=False)
    lena_img = rgb1gray(f='lena512color.tiff', showflag=False)
    mandril_img = rgb1gray(f='mandril_color.tif', showflag=False)
    for i in [1, 2, 3, 5]:
        kernel = gaussKernel(i)
        conv_cameraman = twodConv(cameraman_img, kernel, 'replicate')
        conv_einstein = twodConv(einstein_img, kernel, 'replicate')
        conv_lena = twodConv(lena_img, kernel, 'replicate')
        conv_mandril = twodConv(mandril_img, kernel, 'replicate')
        plt.figure()
        plt.imshow(conv_cameraman, 'gray')
        plt.title('cameraman conv sigma=%d' % i)
        plt.figure()
        plt.imshow(conv_einstein, 'gray')
        plt.title('einstein conv sigma=%d' % i)
        plt.figure()
        plt.imshow(conv_lena, 'gray')
        plt.title('lena conv sigma=%d' % i)
        plt.figure()
        plt.imshow(conv_mandril, 'gray')
        plt.title('mandril conv sigma=%d' % i)
    kernel = gaussKernel(1)
    conv_cameraman = twodConv(cameraman_img, kernel, 'replicate')
    conv_cv = cv.GaussianBlur(cameraman_img, (0, 0), sigmaX=1, sigmaY=1)
    plt.figure()
    plt.imshow(conv_cameraman, 'gray')
    plt.title('My Gauss(sigma=1)')
    plt.figure()
    plt.imshow(conv_cv, 'gray')
    plt.title('opencv gauss(sigma=1)')
    diff = conv_cv - conv_cameraman
    print(diff.sum())
    plt.show()

