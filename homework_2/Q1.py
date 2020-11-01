import cv2
import numpy as np
import matplotlib.pyplot as plt


def showimg(f, title=None, save=None):
    plt.figure()
    plt.imshow(f, 'gray')
    plt.axis('off')
    if title:
        plt.title(title)
    if save:
        plt.savefig(save, dpi=800, bbox_inches='tight')


def dft2D(imgPath):
    """
    a fast 2D fourier transformation for image
    :param imgPath: the path of the image
    :return: image after 2D-fft
    """
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.array(img, 'float')
    (H, W) = f.shape
    fft2d = np.zeros(f.shape, 'complex128')
    # fft2dtest = np.zeros(f.shape, 'complex128')
    for h in range(H):
        for w in range(W):
            f[h, w] = f[h, w] * ((-1) ** (w + h))
    for h in range(H):
        fft2d[h, :] = np.fft.fft((f[h, :]))
    for w in range(W):
        fft2d[:, w] = np.fft.fft((fft2d[:, w]))
    # fft2dtest = np.fft.fft2(f)
    # fft2dtest = np.fft.fftshift(fft2dtest)
    # fft2dtest = np.log(np.abs(fft2dtest))
    fft2d = np.log(np.abs(fft2d)+1)
    # fft2d = np.abs(fft2d)
    # showimg(fft2dtest, title='fftstandard')
    # fft2d = np.array(fft2d, 'uint8')
    showimg(fft2d, title='fft2d', save='figures/fft2d_%s.png' % (imgPath.split('.')[0]))
    # diff = abs(fft2dtest - fft2d).sum()
    return fft2d


if __name__ == '__main__':
    dft2D('rose512.tif')
    plt.show()
