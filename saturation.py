import cv2 as cv
import numpy as np

img_path = "./image/lena.jpg"
img = cv.imread(img_path)
new_img = cv.imread(img_path)


def saturation(x):
    global sat, img
    img = img.astype(np.float32)
    img = img /255.0
    new_img = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    sat = cv.getTrackbarPos("Saturation", "image")
    new_img[ : , :, 2] = sat

        
cv.namedWindow("image", cv.WINDOW_AUTOSIZE)
cv.createTrackbar("Saturation","image",0,255,saturation)
cv.setTrackbarPos("Saturation","image",200)
new_img = cv.cvtColor(new_img, cv.COLOR_HLS2BGR)



while(True):
    cv.imshow('image', new_img)
    if cv.waitKey(1) == ord('q'):  
        break

cv.destroyAllWindows()