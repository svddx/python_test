
import cv2 as cv
import numpy as np

import time
from matplotlib import pyplot as plt

print(cv.__version__)

# if __name__ == '__main__':
#
#     # cv.NamedWindow("camera", 1)
#     # 开启ip摄像头
#     video = "http://admin:admin@192.168.1.215:8081/"
#     capture = cv.VideoCapture(video)
#     print(capture.get(5))
#     num = 0
#     while(capture.isOpened()):
#         ret, frame = capture.read()
#         resultFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#         cv.imshow("camera", resultFrame)
#         if (cv.waitKey(62) & 0xFF == ord('q')):
#             break
#
# #最后记得释放捕捉
# capture.release()
# cv.destroyAllWindows()


def equalHist(image):
    """直方图均衡化，图像增强的一个方法"""
    # 彩色图片转换为灰度图片
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 直方图均衡化，自动调整图像的对比度，让图像变得清晰
    dst = cv.equalizeHist(gray)
    cv.imshow("equalHist", dst)


def clahe(image):
    """
    局部直方图均衡化
    把整个图像分成许多小块（比如按8*8作为一个小块），
    那么对每个小块进行均衡化。
    这种方法主要对于图像直方图不是那么单一的（比如存在多峰情况）图像比较实用
    """
    # 彩色图片转换为灰度图片
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # cliplimit：灰度值
    # tilegridsize：图像切割成块，每块的大小
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    dst = clahe.apply(gray)
    cv.imshow("clahe", dst)




#split(): split image into patches, and save them
#  return: null
# img: image matrix
# ratio : #patch_length/image_length
# n number of patches per line
# dstPath
def split(img1, m, n, dstPath):

    img = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

    height = img.shape[0]
    width = img.shape[1]
    # cv.imshow('img', img)
    # cv.waitKey(1)
    pHeight = m
    pHeightInterval = m
    # print('pHeight: %d\n' %pHeight)
    # print('pHeightInterval: %d\n' %pHeightInterval)
    pWidth = n
    pWidthInterval = n
    # print('pWidth: %d\n' %pWidth)
    # print('pWidthInterval: %d\n' %pWidthInterval)
    cnt = 1
    count = np.zeros(int((height-pHeight)/m) * int((width-pWidth)/n))
    for i in range(int((height-pHeight)/m)):
        for j in range(int((width-pWidth)/n)):
            y = int(pHeightInterval) * i
            x = int(pWidthInterval) * j
            # print('x: %d\n' %(x+pWidth))
            # print('y: %d\n' %(y+pHeight))
            patch = img[y:y+pHeight, x:x+pWidth]
            # cv.imshow('patch', patch)
            # cv.waitKey(1)
            hist_full = cv.calcHist([patch], [0], None, [256], [0, 256])
            # print('cnt: %d' %cnt)
            # print((np.linalg.norm(hist_full[:, 0], ord=0)))
            count[cnt - 1] = np.linalg.norm(hist_full[:, 0], ord=0)
            # cv.imwrite(dstPath + '_%d' %cnt + '.jpg', patch)
            cnt += 1
    sum1 = count.sum()
    mean = sum1 / cnt
    var = count * count
    sum2 = var.sum()

    var = sum2 / cnt - mean ** 2
    if (mean > 30 and mean < 50 and var < 90 ):
        cv.imshow('rain', img1)
        cv.waitKey(1)
    else:
        cv.imshow('other', img1)
        cv.waitKey(1)

    print("mean : ", mean, "var :", var)
    # plt.hist(count, bins=255)
    # plt.show()


src = cv.imread('0.jpg')
path ="D:\\picture_tmp\\"
# cv.imshow('pic', src)
# cv.waitKey(1)

split(src, 8,  8, path)
print("exit")

# img = np.array(src)
# mean = np.mean(img)
# img = img - mean
# img = img + mean * 0.7 #修对比度和亮度
# img = img/255. #非常关键，没有会白屏
# cv.imshow('pic', img)
# cv.waitKey(1)
# print('finish')
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# split(gray, 8,  8, path)


#
# height = src.shape[0]
# width = src.shape[1]
#
# ceil_height = (int)(height / 8)
# ceil_width = (int)(width / 8)
#
#
# # 图一
# cv.imshow("yt", src)
# cv.waitKey(1)
# # 图二
# equalHist(src)
# cv.waitKey(1)
# # 图三
# clahe(src)
# cv.waitKey(1)
#
# cv.imshow("yt", src)
# cv.waitKey(1)