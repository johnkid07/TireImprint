import torch

import numpy as np
import torch.nn.functional as F


def lbp_torch(x):
    # pad image for 3x3 mask size

    x = F.pad(input=x, pad=[1, 1, 1, 1], mode='constant')
    b = x.shape
    M = b[1]
    N = b[2]
    # print(x)

    y = x
    # select elements within 3x3 mask
    # y00  y01  y02
    # y10  y11  y12
    # y20  y21  y22

    # select elements within 5x5 mask
    # y00  y01  y02  y03  y04
    # y10  y11  y12  y13  y14
    # y20  y21  y22  y23  y24
    # y30  y31  y32  y33  y34
    # y40  y41  y42  y43  y44

    # y00 = y[:, 0:M - 4, 0:N - 4]
    # y01 = y[:, 0:M - 4, 1:N - 3]
    # y02 = y[:, 0:M - 4, 2:N - 2]
    # y03 = y[:, 0:M - 4, 3:N - 1]
    # y04 = y[:, 0:M - 4, 4:N]
    #
    # y10 = y[:, 1:M - 3, 0:N - 4]
    # y11 = y[:, 1:M - 3, 0:N - 3]
    # y12 = y[:, 1:M - 3, 0:N - 2]
    # y13 = y[:, 1:M - 3, 0:N - 1]
    # y14 = y[:, 1:M - 3, 0:N]
    #
    # y20 = y[:, 2:M - 2, 0:N - 4]
    # y21 = y[:, 2:M - 2, 0:N - 3]
    # y22 = y[:, 2:M - 2, 0:N - 2]
    # y23 = y[:, 2:M - 2, 0:N - 1]
    # y24 = y[:, 2:M - 2, 0:N]
    #
    # y30 = y[:, 3:M - 1, 0:N - 4]
    # y31 = y[:, 3:M - 1, 0:N - 3]
    # y32 = y[:, 3:M - 1, 0:N - 2]
    # y33 = y[:, 3:M - 1, 0:N - 1]
    # y34 = y[:, 3:M - 1, 0:N]
    #
    # y40 = y[:, 4:M - 0, 0:N - 4]
    # y41 = y[:, 4:M - 1, 0:N - 3]
    # y42 = y[:, 4:M - 2, 0:N - 2]
    # y43 = y[:, 4:M - 3, 0:N - 1]
    # y44 = y[:, 4:M - 4, 0:N]

    # define ALG = [sigma(i, 24) = gi + g] / 25

    y00 = y[:, 0:M - 2, 0:N - 2]
    y01 = y[:, 0:M - 2, 1:N - 1]
    y02 = y[:, 0:M - 2, 2:N]
    #
    y10 = y[:, 1:M - 1, 0:N - 2]
    y11 = y[:, 1:M - 1, 1:N - 1]
    y12 = y[:, 1:M - 1, 2:N]
    #
    y20 = y[:, 2:M, 0:N - 2]
    y21 = y[:, 2:M, 1:N - 1]
    y22 = y[:, 2:M, 2:N]


    p = 8
    # WLGc
    wlg01 = (p*y11 + y01)/8 + p
    wlg02 = (p*y11 + y02)/8 + p
    wlg00 = (p*y11 + y00)/8 + p
    wlg10 = (p*y11 + y10)/8 + p
    wlg11 = (p*y11 + y11)/8 + p
    wlg12 = (p*y11 + y12)/8 + p
    wlg20 = (p*y11 + y20)/8 + p
    wlg21 = (p*y11 + y21)/8 + p
    wlg22 = (p*y11 + y22)/8 + p





    # Comparisons
    # 1 ---------------------------------
    # print("alg : ", (y11 + y01)/9)
    # print("y11 = " , y11)

    bit = torch.le(y00, wlg00)
    tmp = torch.mul(bit, torch.tensor(256))

    # 2 ---------------------------------
    bit = torch.le(y01, wlg01)
    val = torch.mul(bit, torch.tensor(128))
    val = torch.add(val, tmp)
    # print(val)

    # 3 ---------------------------------
    bit = torch.le(y02, wlg02)
    tmp = torch.mul(bit, torch.tensor(64))
    val = torch.add(val, tmp)

    # 4 ---------------------------------
    bit = torch.le(y10, wlg10)
    tmp = torch.mul(bit, torch.tensor(32))
    val = torch.add(val, tmp)

    # 5 ---------------------------------
    bit = torch.le(y11, wlg11)
    tmp = torch.mul(bit, torch.tensor(16))
    val = torch.add(val, tmp)

    # 6 ---------------------------------
    bit = torch.le(y12, wlg12)
    tmp = torch.mul(bit, torch.tensor(8))
    val = torch.add(val, tmp)

    # 7 ---------------------------------
    bit = torch.le(y20, wlg20)
    tmp = torch.mul(bit, torch.tensor(4))
    val = torch.add(val, tmp)

    # 8 ---------------------------------
    bit = torch.le(y21, wlg21)
    tmp = torch.mul(bit, torch.tensor(2))
    val = torch.add(val, tmp)

    bit = torch.le(y22, wlg22)
    tmp = torch.mul(bit, torch.tensor(1))
    val = torch.add(val, tmp)
    return val


# print('Random test numbers:')
# imgs=np.random.randint(0,256,(7,7))
# print(imgs)
#
# # Compute using pytorch
# y1=lbp_torch(torch.from_numpy(imgs.reshape(1,7,7)))
# # Compute using python
#
# print('Python computation result:')
# print('PyTorch computation result:')
# print(y1.numpy().reshape(7,7).astype('uint8'))
