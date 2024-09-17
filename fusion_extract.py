import glob

import pywt
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
from PIL import Image

def chTranform(ch1,ch2,shape):
    cooef1 = pywt.dwt2(ch1,'db5',mode = 'periodization')
    cooef2 = pywt.dwt2(ch2, 'db5',mode = 'periodization')
    cA1, (cH1,cV1,cD1) = cooef1
    cA2, (cH2,cV2,cD2) = cooef2

    cA = (cA1 + cA2)/2
    cH = (cH1 + cH2)/2
    cV = (cV1 + cV2)/2
    cD = (cD1 + cD2)/2

    fincoC = cA,(cH,cV,cD)
    outImgC = pywt.idwt2(fincoC, 'db5',mode = 'periodization')
    outImgC = cv2.resize(outImgC, (shape[0],shape[1]))
    return outImgC


def fusion(img1, img2):


    I1 = img1
    I2 = img2


    I2 = cv2.resize(I2, (I1.shape[1], I1.shape[0]))

    # print(I1.shape)
    # print(I2.shape)

    iR1 = I1.copy()
    iR1[:,:,1] = iR1[:,:,2] = 0
    iR2 = I2.copy()
    iR2[:,:,1] = iR2[:,:,2] = 0


    iG1 = I1.copy()
    iG1[:,:,0] = iG1[:,:,2] = 0
    iG2 = I2.copy()
    iG2[:,:,0] = iG2[:,:,2] = 0

    iB1 = I1.copy()
    iB1[:,:,0] = iB1[:,:,1] = 0
    iB2 = I2.copy()
    iB2[:,:,0] = iB2[:,:,1] = 0

    shape = (I1.shape[1],I2.shape[0])
    # print(shape)

    outImageR = chTranform(iR1,iR2, shape)
    outImageG = chTranform(iG1,iG2, shape)
    outImageB = chTranform(iB1, iB2, shape)


    outImage = I1.copy()
    outImage[:,:,0] = outImage[:,:,1] = outImage[:,:,2] = 0
    outImage[:,:,0] = outImageR[:,:,0]
    outImage[:,:,1] = outImageG[:,:,1]
    outImage[:,:,2] = outImageB[:,:,2]

    outImage = np.multiply(np.divide(outImage - np.min(outImage),(np.max(outImage) - np.min(outImage))),255)
    outImage = outImage.astype(np.uint8)

    return outImage



# path1 = glob.glob('test/ss.jpg_HH.jpg')
# path2 = glob.glob('test/ss.jpg_HL.jpg')
# path3 = glob.glob('test/ss.jpg_LH.jpg')


p1 = 'HH.jpg'
p2 = 'HL.jpg'
p3 = 'LH.jpg'


img1 = cv2.imread(p1)
img2 = cv2.imread(p2)
img3 = cv2.imread(p3)

outImage = fusion(img1, img2)
outImage = fusion(outImage, img3)

cv2.imwrite('_fusion2.jpg', outImage)




# for n,m,z in zip(path1,path2,path3):
#     img1 = cv2.imread(n)
#     img2 = cv2.imread(m)
#     img3 = cv2.imread(z)
#
#     outImage = fusion(img1,img2)
#     outImage = fusion(outImage,img3)
#
#     if cv2.imwrite(str(n) + '_fusion.jpg', outImage):
#         print(n + ' wrote')
#     else:
#         print('failed')



