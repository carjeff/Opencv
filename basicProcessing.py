
import numpy as np
import cv2 as cv

image = cv.imread("./image/lena.jpg")
cv.namedWindow("Image", cv.WINDOW_NORMAL)
cv.imshow("Image", image)


M = np.float32([[1, 0, 25], [0, 1, 50]]) #平移矩阵1：向x正方向平移25，向y正方向平移50
shifted = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
cv.imshow("Image", shifted)
cv.waitKey(2000)

M = np.float32([[1, 0, -50], [0, 1, -90]])#平移矩阵2：向x负方向平移-50，向y负方向平移-90
shifted = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
cv.imshow("Image", shifted)
cv.waitKey(2000)


#读取image的中心center
(h,w) = image.shape[:2]
center = (h / 2, w / 2)

#旋转45度，缩放0.75
M = cv.getRotationMatrix2D(center, 45, 0.75)#旋转缩放矩阵：(旋转中心，旋转角度，缩放因子)
rotated = cv.warpAffine(image, M, (h, w))
cv.imshow("Image", rotated)
cv.waitKey(2000)

#旋转-45度，缩放1.25
M = cv.getRotationMatrix2D(center, -45, 1.25)#旋转缩放矩阵：(旋转中心，旋转角度，缩放因子)
rotated = cv.warpAffine(image, M, (h, w))
cv.imshow("Image", rotated)
cv.waitKey(2000)


#图像水平翻转
flipped = cv.flip(image, 1)
cv.imshow("Image", flipped)
cv.waitKey(2000)

#图像垂直翻转
flipped = cv.flip(image, 0)
cv.imshow("Image", flipped)
cv.waitKey(2000)

#图像水平垂直翻转
flipped = cv.flip(image, -1)
cv.imshow("Image", flipped)
cv.waitKey(2000)

cv.destroyAllWindows()




