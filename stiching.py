from Stitcher import Stitcher
import cv2

# 读取拼接图片
imageA = cv2.imread("image/left_01.png")
imageB = cv2.imread("image/right_01.png")

# 把图片拼接成全景图
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB])

# 显示所有图片
cv2.imshow("Matches", vis)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()