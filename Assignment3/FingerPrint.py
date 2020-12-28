# coding:utf8
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def ReadFingerprint(path):
    img = cv.imread(path)
    b, g, r = cv.split(img)  # 拆分通道
    # rgbimg = cv.merge([r, g, b])  # 合并通道
    grayimg = np.round(r * 0.2989 + g * 0.587 + b * 0.114)
    grayimg = np.array(grayimg, dtype=np.uint8)
    return grayimg


def showimg(img, name=None, save=None):
    plt.figure()
    plt.imshow(img, 'gray')
    plt.axis('off')
    # if name != 'None':
    #     plt.title(name)
    if save:
        plt.savefig(save, dpi=800, bbox_inches='tight')


def skeleton_extract_erosion(binary):
    # kernel = np.ones((3, 3), np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    skeleton = np.zeros(binary.shape, np.uint8)
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
    for i in range(1, 100):
        erosion = cv.erode(binary, kernel, iterations=1)
        dilation = cv.dilate(erosion, kernel, iterations=1)
        subtract = (binary - dilation)
        # showimg(subtract, name='%d' % i, save=False)
        skeleton = cv.bitwise_or(skeleton, subtract)
        s = cv.countNonZero(erosion)
        if s == 0:
            break
        binary = erosion.copy()
    return skeleton


def skeleton_distance(binary_img):
    # kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    kernel = np.ones((7, 7))
    # erosion = cv.erode(binary_img, kernel, iterations=1)
    #####get outer boundary######
    dilation = cv.dilate(binary_img, kernel, iterations=2)
    outer_boundary = dilation - binary_img

    #####distance transform######
    dist = cv.distanceTransform(binary_img, cv.DIST_L2, 3)

    #####get local maximum value#####
    # skeleton_distance=cv.threshold(dist, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # skeleton_distance = np.array(dist, dtype=np.uint8)
    # skeleton_distance = cv.erode(dist, kernel, iterations=1)
    h = dist.shape[0]
    w = dist.shape[1]
    # kernel_size = 5
    # half_size = int(kernel_size / 2)
    # dist_pad = np.pad(dist, ((half_size, half_size), (half_size, half_size)),
    #                   'constant', constant_values=(0, 0))
    # dist_mask = np.full_like(dist, False, dtype=np.bool)
    # for i in range(half_size, h + half_size):
    #     for j in range(half_size, w + half_size):
    #         if (dist_pad[i,j]!=0):
    #             window = dist_pad[(i - half_size):(i + half_size+1),
    #                      (j - half_size):(j + half_size+1)]
    #             if (np.max(window) == dist_pad[i, j]):
    #                 dist_mask[i-half_size, j-half_size] = True
    laplace_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    # xkernel = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]])
    # ykernel = xkernel.T
    grad = cv.filter2D(dist, -1, laplace_kernel, borderType=cv.BORDER_REPLICATE)
    # gradx = cv.filter2D(dist, -1, xkernel, borderType=cv.BORDER_REPLICATE)
    # grady = cv.filter2D(dist, -1, ykernel, borderType=cv.BORDER_REPLICATE)
    # sobelx = cv.Sobel(dist, cv.CV_32F, 1, 0, ksize=5)  # 1，0参数表示在x方向求一阶导数
    #
    # sobely = cv.Sobel(dist, cv.CV_64F, 0, 1, ksize=5)  # 0,1参数表示在y方向求一阶导数

    # x_mask = (gradx > 1.8)
    # y_mask = (grady > 1.8)
    mask = grad > 2.8

    # mask = (x_mask + y_mask)

    skeleton_distance = binary_img * mask

    return dist, skeleton_distance


def crop(f):
    row, col = f.shape
    f1 = f.copy()
    # crop kernel
    A = np.array([0, -1, -1, 1, 1, -1, 0, -1, -1]).reshape(3, 3)
    A1 = np.rot90(A)
    A2 = np.rot90(A1)
    A3 = np.rot90(A2)
    B = np.array([1, -1, -1, -1, 1, -1, -1, -1, -1]).reshape(3, 3)
    B1 = np.rot90(B)
    B2 = np.rot90(B1)
    B3 = np.rot90(B2)
    maskList = [A, A1, A2, A3, B, B1, B2, B3]
    # 细化
    for k in range(2):
        for m in maskList:
            f1_hitmiss = cv.morphologyEx(f1, cv.MORPH_HITMISS, m)
            f1 = cv.bitwise_and(f1, 255 - f1_hitmiss)
    f2 = np.zeros(f.shape, dtype=np.uint8)
    # 复原
    for m in maskList:
        f2_hitmiss = cv.morphologyEx(f1, cv.MORPH_HITMISS, m)
        f2 = cv.bitwise_or(f2, f2_hitmiss)
    # 膨胀端点
    H = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]).reshape([3, 3])
    f3 = cv.dilate(f2, H, iterations=1)
    f3 = cv.bitwise_and(f3, f)
    # 求并
    f4 = cv.bitwise_or(f1, f3)
    return f4


if __name__ == '__main__':
    path = 'fingerprint.jpg'
    gray = ReadFingerprint(path)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    binary = 255 - binary
    showimg(gray, name='Gray Image', save='figures/gray_image.jpg')
    showimg(binary, name='Binary Image', save='figures/binary_image.jpg')
    skeleton_morphology = skeleton_extract_erosion(binary)
    showimg(skeleton_morphology, name='Skeleton by Morphology', save='figures/morph_skeleton.jpg')
    dist, skeleton_dis = skeleton_distance(binary)
    showimg(dist, name='distance', save='figures/distance.jpg')
    showimg(skeleton_dis, name='Skeleton by Distance', save='figures/distance_skeleton.jpg')

    # kernel = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]).reshape([3, 3])
    skeleton_morph_cut = crop(skeleton_morphology)
    # skeleton_morph_cut = cv.morphologyEx(skeleton_morph_cut, cv.MORPH_CLOSE, kernel)
    showimg(skeleton_morph_cut, name='Skeleton by Distance', save='figures/morph_skeleton_cut.jpg')

    skeleton_dis_cut = crop(skeleton_dis)
    # skeleton_dis_cut = cv.morphologyEx(skeleton_dis_cut, cv.MORPH_CLOSE, kernel)
    showimg(skeleton_dis_cut, name='Skeleton by Distance', save='figures/distance_skeleton_cut.jpg')
    plt.show()
