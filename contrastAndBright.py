import cv2
import numpy as np

alpha = 0
beta = 0
img_path = "./image/lena.jpg"
img = cv2.imread(img_path)
new_img = cv2.imread(img_path)

def updateAlpha(x):
    global alpha, img, new_img
    alpha = cv2.getTrackbarPos('Contrast', 'image')
    alpha = alpha * 0.01
    new_img = np.uint8(np.clip(((alpha+0.2) * img + beta), 0, 255))

def updateBeta(x):
    global beta, img, new_img
    beta = cv2.getTrackbarPos('Bright', 'image')
    new_img = np.uint8(np.clip((alpha * img + beta), 0, 255))
  

# 创建窗口
cv2.namedWindow('image')
cv2.createTrackbar('Contrast', 'image', 0, 300, updateAlpha)
cv2.createTrackbar('Bright', 'image', 0, 255, updateBeta)
cv2.setTrackbarPos('Contrast', 'image', 90)
cv2.setTrackbarPos('Bright', 'image', 10)
 


while(True):
    cv2.imshow('image', new_img)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()