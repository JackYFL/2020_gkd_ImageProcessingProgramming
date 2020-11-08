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


def dft2D(img, frequency=False, shift=True):
    """
    a fast 2D fourier transformation for image
    :param f: gray image
    :return: image after 2D-fft
    """
    f = np.array(img, 'complex128')
    (H, W) = f.shape
    fft2d = np.zeros(f.shape, 'complex128')
    if shift:
        for h in range(H):
            for w in range(W):
                f[h, w] = f[h, w] * ((-1) ** (w + h))
    for h in range(H):
        fft2d[h, :] = np.fft.fft((f[h, :]))
    for w in range(W):
        fft2d[:, w] = np.fft.fft((fft2d[:, w]))
    if frequency:
        fft2d = np.log(np.abs(fft2d))
    # fft2d = np.array(fft2d, 'uint8')
    return fft2d


def idft2D(F):
    """
    Inverse fourier transformation of the image
    :param F: the fourier transformation of the image
    :return: the inverse of the image
    """
    Fconjugate = F.conjugate()
    (W, H) = F.shape
    fdft = dft2D(img=Fconjugate, shift=False)
    idft = (fdft / (W * H)).conjugate()

    return idft


if __name__ == '__main__':
    img = cv2.imread('rose512.tif')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    showimg(img, title='gray image', save='figures/rose_gray.png')
    f = ((img - img.min()) / (img.max() - img.min()))
    F = dft2D(f, shift=False)
    g = idft2D(F)
    d = 255*(f - g)
    d = np.array(d, 'float')
    diff = abs(d).sum()
    showimg(d, title='The difference of two images', save='figures/difference_rose512_nounit8.png')
    d = np.array(d, 'uint8')
    diff = abs(d).sum()
    showimg(d, title='The difference of two images', save='figures/difference_rose512.png')
    plt.show()
