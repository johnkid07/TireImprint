import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pywt
from rlbp_wlg_python import  lbp_torch
def lbp(img):
    img = np.asarray(img)
    [R,C] = img.shape
    img = torch.from_numpy(img.reshape(1,R,C).astype('uint8'))


    val = lbp_torch(img)

    return val.numpy(), R, C



img = cv2.imread("QK02-KH-OB3.JPG", cv2.IMREAD_GRAYSCALE)

dwt_photo = pywt.dwt2(img, 'bior1.3')
LL, (LH, HL, HH) = dwt_photo
grayLH, Rows, Cols = lbp(LH)
grayHL, Rows, Cols = lbp(HL)
grayHH, Rows, Cols = lbp(HH)

grayLH = grayLH.reshape(Rows, Cols).astype('uint8')
grayHL = grayHL.reshape(Rows, Cols).astype('uint8')
grayHH = grayHH.reshape(Rows, Cols).astype('uint8')

# arr , R, C = lbp(img)
# imlbp, r, c = lbp(img)
# arr = imlbp.reshape(r, c).astype('uint8')

cv2.imwrite('LH.jpg', grayLH)
cv2.waitKey(0)
cv2.destroyAllWindows()
