import cv2 as cv
import os
import matplotlib.pyplot as plt


def scanLine4e(f, I, loc):
    """
    :param f: the gray-scale map
    :param I: integer (row if loc=='row'; column if loc=='column')
    :param loc: string ('row' or 'column')
    :return: s: pixel vector of one row or column
    """
    if os.path.exists(f):
        img = cv.imread(f, 0)  # Read image and turn rgb 2 gray
        # cv.namedWindow('input image')
        # cv.imshow('input image', img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        if I < img.shape[0] and I < img.shape[1] and I > 0:
            if loc == 'row':
                s = img[I]  # extract the I-th row vector
                plt.figure()
                ######bar graph######
                # plt.bar(s, label='row pixel vector')
                # plt.xlabel('pixel index', fontsize=17)
                # plt.ylabel('value', fontsize=17)
                # plt.title(u'The I-th row pixel vector bar', fontsize=20)
                # plt.show()
                ######histogram######
                plt.hist(s, bins=18, facecolor='red', edgecolor='blue',alpha=0.6)
                plt.xlabel('pixel index', fontsize=17)
                plt.ylabel('value',fontsize=17)
                plt.title(u'The I-th row pixel vector hist', fontsize=20)
            elif loc == 'col':
                s = img[:][I - 1]  # extract the I-th col vector
                plt.figure()
                # plt.bar(x=range(1, s.shape[0] + 1), height=s.tolist(), label='col pixel vector')
                # plt.xlabel('pixel index')
                # plt.ylabel('value')
                # plt.title(u'The I-th col pixel vector', fontsize=20)
                plt.hist(s, bins=18, facecolor='red', edgecolor='blue',alpha=0.6)
                plt.xlabel('pixel index', fontsize=17)
                plt.ylabel('value',fontsize=17)
                plt.title(u'The I-th col pixel vector hist', fontsize=20)
            else:
                print('The loc parameter is invalid, please input again!')
        else:
            print('The pixel vector index is invalid, please input again!')
    else:
        print('The path is invalid, please input again!')


if __name__ == '__main__':
    scanLine4e(f='cameraman.tif', I=128, loc='row')
    scanLine4e(f='einstein.tif', I=128, loc='col')
    plt.show()
