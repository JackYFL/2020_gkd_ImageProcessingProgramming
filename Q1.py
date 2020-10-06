#coding:utf8
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
                ######bar graph######
                plt.figure()
                plt.plot(range(1, 1 + s.shape[0]), s, linewidth=3)
                plt.xlabel('pixel index', fontsize=20)
                plt.ylabel('value', fontsize=20)
                plt.title(u'The I-th row pixel vector', fontsize=20)
                plt.savefig('figures/The_I-th_row_pixel_vector_plot.png', dpi=800, bbox_inches='tight')
                # ######histogram######
                # plt.figure()
                # plt.hist(s, bins=18, facecolor='red', edgecolor='blue', alpha=0.6)
                # plt.xlabel('pixel index', fontsize=20)
                # plt.ylabel('value', fontsize=20)
                # plt.title(u'The I-th row pixel vector hist', fontsize=20)
                # plt.savefig('figures/The_I-th_row_pixel_vector_hist.png', dpi=800, bbox_inches='tight')
            elif loc == 'col':
                s = img[:,I]  # extract the I-th col vector
                plt.figure()
                plt.plot(range(1, 1 + s.shape[0]), s, linewidth=3)
                plt.xlabel('pixel index', fontsize=20)
                plt.ylabel('value', fontsize=20)
                plt.title(u'The I-th col pixel vector', fontsize=20)
                plt.savefig('figures/The_I-th_col_pixel_vector_plot.png', dpi=800, bbox_inches='tight')
                # plt.figure()
                # plt.hist(s, bins=18, facecolor='red', edgecolor='blue', alpha=0.6)
                # plt.xlabel('pixel index', fontsize=20)
                # plt.ylabel('value', fontsize=20)
                # plt.title(u'The I-th col pixel vector hist', fontsize=20)
                # plt.savefig('figures/The_I-th_col_pixel_vector_hist.png', dpi=800, bbox_inches='tight')
            else:
                print('The loc parameter is invalid, please input again!')
        else:
            print('The pixel vector index is invalid, please input again!')
    else:
        print('The path is invalid, please input again!')


if __name__ == '__main__':
    scanLine4e(f='cameraman.tif', I=128, loc='row')
    scanLine4e(f='cameraman.tif', I=128, loc='col')
    plt.show()
