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


def pad_img(img):
    (W, H) = img.shape
    WW = 2 ** np.ceil(np.log2(W))
    HH = 2 ** np.ceil(np.log2(H))
    # img_pad = np.zeros((2 ** int(Wlog), 2 ** int(Hlog)))
    # img_pad = np.pad(img, (
    #     (int((WW - W) / 2), int(WW - int((WW - W) / 2) - W)), (int((HH - H) / 2), int(HH - int((HH - H) / 2) - H))),
    #                  constant_values=0)
    img_pad = np.pad(img, (
        (int((WW - W) / 2), int(WW - int((WW - W) / 2) - W)), (int((HH - H) / 2), int(HH - int((HH - H) / 2) - H))),
                     'edge')
    return img_pad


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
        fft2d = np.log(np.abs(fft2d) + 1)
    if type:
        fft2d = np.array(np.abs(fft2d), 'float')
    return fft2d


def show_dft_img(img_path, save_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_pad = pad_img(img)
    img_dft2d = dft2D(img=img_pad, frequency=True, shift=True, type=False)

    (H, W) = img.shape
    (HH, WW) = img_dft2d.shape
    showimg(f=img_dft2d[int((HH - H) / 2):int((HH - H) / 2 + H), int((WW - W) / 2):int((WW - W) / 2 + W)],
            save=save_path)


if __name__ == '__main__':
    show_dft_img(img_path='house02.tif', save_path='figures/house02_dft2d.png')
    show_dft_img(img_path='Characters_test_pattern.tif', save_path='figures/charactors_dft2d.png')
    show_dft_img(img_path='house.tif', save_path='figures/house_dft2d.png')
    show_dft_img(img_path='lena_gray_512.tif', save_path='figures/lena_gray512_dft2d.png')
    show_dft_img(img_path='lunar_surface.tif', save_path='figures/lunar_surface_dft2d.png')

    plt.show()
