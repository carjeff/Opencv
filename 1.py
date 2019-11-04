import numpy as np
import glob

from PIL import Image

f = glob.glob('./image/lena.jpg')

PicData = np.zeros((1000, 98306))

for i in range(1000):
    img = Image.open(f[i]).convert('L')
    img = Image.open(f[i])
    img = img.resize((256,256))
    img = np.array(img)
    m, n = img.shape
    im = img.reshape((-1, 1))
    PicData[i, :98304] = im[:, 0]
    PicData[i, 98304] = m
    PicData[i, 98305] = n

print(PicData.shape)
# print(PicData[:,1])
np.save('PicData', PicData)