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


def create_rectangle(H, W, normalize=False):
    img = np.zeros((512, 512))
    img[(256 - int(H / 2)):(256 + int(H / 2)), (256 - int(W / 2)):(256 + int(W / 2))] = np.ones((H, W)) * 255
    if normalize:
        f = ((img - img.min()) / (img.max() - img.min()))
    return f


def dft2D(img, frequency=False, shift=True, type=False):
    """
    a fast 2D fourier transformation for image
    :param f: gray image
    :param frequency: whether calculate the log abs image
    :param shift: whether centering the image
    :param type: whether get the abs image
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
        fft2d = np.log(np.abs(fft2d)+1)
    if type:
        fft2d = np.array(np.abs(fft2d), 'float')
    return fft2d


if __name__ == '__main__':
    img = create_rectangle(60, 10, normalize=True)
    dft2dimg_noshift = dft2D(img, shift=False, frequency=False, type=True)
    dft2dimg_shift = dft2D(img, shift=True, frequency=False, type=True)
    dft2dimg_shift_log = dft2D(img, shift=True, frequency=True, type=True)
    showimg(img, title='source img', save='figures/src_img.png')
    showimg(dft2dimg_noshift, title='frequency no shifting img', save='figures/frequency_img.png')
    showimg(dft2dimg_shift, title='frequency shifting img', save='figures/frequency_shift_img.png')
    showimg(dft2dimg_shift_log, title='frequency log shifting img', save='figures/frequency_log_shift_img.png')

    plt.show()