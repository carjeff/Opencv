import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
def blur(image):      #均值模糊  去随机噪声有很好的去燥效果
    dst = cv.blur(image, (1, 15))    #（1, 15）是垂直方向模糊，（15， 1）是水平方向模糊
    cv.imshow("meanBlur", dst)
    cv.waitKey(0)


def medianBlur(image):    # 中值模糊  对椒盐噪声有很好的去燥效果
    dst = cv.medianBlur(image, 5)
    cv.imshow("medianBlur", dst)
    cv.waitKey(0)

def GaussianBlur(image):
    dst = cv.GaussianBlur(image, (7, 7), 0)
    cv.imshow("GaussianBlur", dst)
    cv.waitKey(0)

def bilateralFilter(image):
    dst = cv.bilateralFilter(image, 9, 5, 5)
    cv.imshow("bilateralFilter", dst)
    cv.waitKey(0)

def sharpening(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
    dst = cv.filter2D(image, -1, kernel=kernel)
    cv.imshow("sharpening", dst)
    cv.waitKey(0)

def sobel(image):
    sobelx = cv.Sobel(image,cv.CV_64F, 1, 0, ksize=3)
    # 利用Sobel方法可以进行sobel边缘检测
    # img表示源图像，即进行边缘检测的图像
    # cv2.CV_64F表示64位浮点数即64float。
    # 这里不使用numpy.float64，因为可能会发生溢出现象。用cv的数据则会自动
    # 第三和第四个参数分别是对X和Y方向的导数（即dx,dy），对于图像来说就是差分，这里1表示对X求偏导（差分），0表示不对Y求导（差分）。其中，X还可以求2次导。
    # 注意：对X求导就是检测X方向上是否有边缘。
    # 第五个参数ksize是指核的大小。
    sobely = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)
    # 与上面不同的是对y方向进行边缘检测

    sobelXY = cv.Sobel(image, cv.CV_64F, 1, 1, ksize=3)
    # 这里对两个方向同时进行检测，则会过滤掉仅仅只是x或者y方向上的边缘
    plt.subplot(2, 1, 1)
    plt.imshow(image, 'gray')
    # 其中gray表示将图片用灰度的方式显示，注意需要使用引号表示这是string类型。
    # 可以用本行命令显示'gray'的类型：print(type('gray'))
    plt.title('image')
    plt.subplot(2, 2, 2)
    plt.imshow(sobelx, 'gray')
    plt.title('sobelX')
    plt.subplot(2, 2, 3)
    plt.imshow(sobely, 'gray')
    plt.title('sobelY')
    plt.subplot(2, 2, 4)
    plt.imshow(sobelXY, 'gray')
    plt.title('sobelXY')
    plt.show()


image = cv.imread("./image/lena.jpg")
gimage = cv.imread("./image/lena.jpg",0)
cv.namedWindow("image", cv.WINDOW_NORMAL)
cv.imshow("image", image)

cv.waitKey(0)
blur(image)
medianBlur(image)
GaussianBlur(image)
bilateralFilter(image)
sharpening(image)
sobel(gimage)