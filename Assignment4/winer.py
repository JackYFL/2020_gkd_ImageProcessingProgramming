import cv2
from matplotlib import pyplot as plt
import numpy as np
import pywt


# print(pywt.families())
# ['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus', 'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor']

def guass_noise(pic, SNR=1):
    # SNR为信噪比
    pic = np.array(pic, dtype=float)
    SNR = 10 ** (SNR / 10)
    row, col = np.shape(pic)
    pic_power = np.sum(pic * pic) / (row * col)
    noise_power = pic_power / SNR
    noise = np.random.randn(row, col) * np.sqrt(noise_power)
    pic = (noise + pic)
    pic = np.where(pic <= 0, 0, pic)
    pic = np.where(pic > 255, 255, pic)
    return np.uint8(pic)


def wiener_filter(pic, HH):
    # r padding半径
    row, col = np.shape(pic)
    noise_std = (np.median(np.abs(HH)) / 0.6745)
    noise_var = noise_std ** 2
    ############################
    # noise_var=50
    # filter
    # mysize = 3
    # step = 3
    # ans = np.zeros([row, col])
    # for i in range(0, row - mysize, step):
    #     for j in range(0, col - mysize, step):
    #         var = (1 / (mysize * mysize)) * np.sum(pic[i:(i + mysize), j:j + (mysize)]) - noise_var
    #         ans[i:(i + mysize), j:(j + mysize)] = pic[i:(i + mysize), j:(j + mysize)] * var / (var + noise_var)
    ############################
    var = 1 / (row * col) * np.sum(pic * pic) - noise_var
    ans = pic * var / (var + noise_var)
    return ans


def wiener_dwt(pic, index=1):
    # index 为进行几层分解与重构
    pic = np.array(pic, dtype=float)
    coeffs = pywt.dwt2(pic, 'bior4.4')
    LL, (LH, HL, HH) = coeffs

    # LL为低频信号 LH为水平高频 HL为垂直高频  HH为对角线高频信号

    # 维纳滤波
    LH = wiener_filter(LH, HH)
    HL = wiener_filter(HL, HH)
    HH = wiener_filter(HH, HH)

    # 重构
    if index > 1:
        LL = wiener_dwt(LL, index - 1)
        # bior4.4小波重构可能会改变矩阵维数，现统一矩阵维数
        row, col = np.shape(LL)
        d1 = row - np.shape(HH)[0]
        d2 = col - np.shape(HH)[1]
        if d1 > 0 or d2 > 0:
            d1 = row - np.arange(d1) - 1
            d2 = col - np.arange(d2) - 1
            LL = np.delete(LL, d1, axis=0)
            LL = np.delete(LL, d2, axis=1)
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(LL, cmap='gray')
    plt.axis('off')
    # plt.title('The ' + str(time - index + 1) + 'th dwt----' + 'LL')
    plt.subplot(2, 2, 2)
    plt.imshow(LH, cmap='gray')
    plt.axis('off')
    # plt.title('The ' + str(time - index + 1) + 'th dwt----' + 'LH')
    plt.subplot(2, 2, 3)
    plt.imshow(HL, cmap='gray')
    plt.axis('off')
    # plt.title('The ' + str(time - index + 1) + 'th dwt----' + 'HL')
    plt.subplot(2, 2, 4)
    plt.imshow(HH, cmap='gray')
    plt.axis('off')
    plt.tight_layout()  # 调整整体空白
    plt.subplots_adjust(wspace=0, hspace=0.1)  # 调整子图间距
    # plt.title('The ' + str(time - index + 1) + 'th dwt----' + 'HH')
    plt.savefig('figures/the_%d_th_dwt.jpg' % (time - index + 1), dpi=800, bbox_inches='tight')
    plt.show()
    pic_ans = pywt.idwt2((LL, (LH, HL, HH)), 'bior4.4')
    # pic_ans = np.where(pic_ans <= 0, 0, pic_ans)
    # pic_ans = np.where(pic_ans > 255, 255, pic_ans)
    return pic_ans


def run(filename):
    SNR = 20
    global time
    time = 2  # 分解次数
    f = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    f_noise = guass_noise(f, SNR)
    f_process = wiener_dwt(f_noise, time)
    f_process = np.where(f_process <= 0, 0, f_process)
    f_process = np.where(f_process > 255, 255, f_process)
    f_process = np.uint8(f_process)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(f, cmap='gray')
    plt.axis('off')
    # plt.title('original image')
    plt.subplot(1, 3, 2)
    plt.imshow(f_noise, cmap='gray')
    plt.axis('off')
    # plt.title('polluted image ----SNR = ' + str(SNR))
    plt.subplot(1, 3, 3)
    plt.imshow(f_process, cmap='gray')
    plt.axis('off')
    # plt.title('polluted image after wiener_dwt')
    plt.tight_layout()  # 调整整体空白
    plt.savefig('figures/orig_noisy_denoise.jpg', dpi=800, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    run('lena512color.tiff')
