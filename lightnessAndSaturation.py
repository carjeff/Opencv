import cv2
import numpy as np

def s_and_b(arg):
    lsImg = np.zeros(image.shape, np.float32)
    hlsCopy = np.copy(hlsImg)
    l = cv2.getTrackbarPos('Lightness', 'lightness and saturation')
    s = cv2.getTrackbarPos('Saturation', 'lightness and saturation')
    # 1.调整亮度饱和度(线性变换)、 2.将hlsCopy[:,:,1]和hlsCopy[:,:,2]中大于1的全部截取
    hlsCopy[:, :, 1] = (1.0 + l / float(MAX_VALUE)) * hlsCopy[:, :, 1]
    hlsCopy[:, :, 1][hlsCopy[:, :, 1] > 1] = 1
    # HLS空间通道2是饱和度，对饱和度进行线性变换，且最大值在255以内，这归一化了，所以应在1以内
    hlsCopy[:, :, 2] = (1.0 + s / float(MAX_VALUE)) * hlsCopy[:, :, 2]
    hlsCopy[:, :, 2][hlsCopy[:, :, 2] > 1] = 1
    # HLS2BGR
    lsImg = cv2.cvtColor(hlsCopy, cv2.COLOR_HLS2BGR)
    # 显示调整后的效果
    cv2.imshow("lightness and saturation", lsImg)


image = cv2.imread('./image/lena.jpg', 1)
# 图像归一化，且转换为浮点型, 颜色空间转换 BGR转为HLS
new_Img = image.astype(np.float32)
new_Img = new_Img / 255.0
# HLS空间，三个通道分别是: Hue色相、lightness亮度、saturation饱和度
# 通道0是色相、通道1是亮度、通道2是饱和度
hlsImg = cv2.cvtColor(new_Img, cv2.COLOR_BGR2HLS)

l, s, MAX_VALUE = 100, 100, 200
cv2.namedWindow("lightness and saturation", cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar("Lightness", "lightness and saturation", l, MAX_VALUE, s_and_b)
cv2.createTrackbar("Saturation", "lightness and saturation", s, MAX_VALUE, s_and_b)

s_and_b(0)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()