import cv2
from matplotlib import pyplot as plt
import medpy.metric as mm
import numpy as np
# 计算DICE系数，即DSI
def calDSI(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    DSI_s, DSI_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                DSI_s += 1
            if binary_GT[i][j] == 255:
                DSI_t += 1
            if binary_R[i][j] == 255:
                DSI_t += 1
    DSI = 2 * DSI_s / DSI_t
    # print(DSI)
    return DSI


# 计算VOE系数，即VOE
def calVOE(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    VOE_s, VOE_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255:
                VOE_s += 1
            if binary_R[i][j] == 255:
                VOE_t += 1
    VOE = 2 * (VOE_t - VOE_s) / (VOE_t + VOE_s)
    return VOE


# 计算RVD系数，即RVD
def calRVD(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    RVD_s, RVD_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255:
                RVD_s += 1
            if binary_R[i][j] == 255:
                RVD_t += 1
    RVD = RVD_t / RVD_s - 1
    return RVD

def calASSD(binary_GT, binary_R):
    return mm.obj_assd(binary_GT,binary_R)

#此条存疑，自己写的
def calRMSD(binary_GT, binary_R):
    return np.sqrt( np.mean(  (np.square(mm.obj_asd(binary_GT, binary_R)), np.square(mm.obj_asd(binary_R, binary_GT)) )  ) )

def calMaxD(binary_GT, binary_R):
    return np.max( (mm.obj_asd(binary_GT, binary_R), mm.obj_asd(binary_R, binary_GT)) )

# 计算Prevision系数，即Precison
def calPrecision(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    P_s, P_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                P_s += 1
            if binary_R[i][j] == 255:
                P_t += 1

    Precision = P_s / P_t
    return Precision


# 计算Recall系数，即Recall
def calRecall(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    R_s, R_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                R_s += 1
            if binary_GT[i][j] == 255:
                R_t += 1

    Recall = R_s / R_t
    return Recall


if __name__ == '__main__':
    img_GT = cv2.imread('./preds/1_gt.png', 0)
    img_R = cv2.imread('./preds/1_pred.png', 0)
    ret_GT, binary_GT = cv2.threshold(img_GT, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ret_R, binary_R = cv2.threshold(img_R, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    plt.figure()
    plt.subplot(121), plt.imshow(binary_GT), plt.title('真值图')
    plt.axis('off')
    plt.subplot(122), plt.imshow(binary_R), plt.title('分割图')
    plt.axis('off')
    plt.show()

    print('{0:.4}'.format(calDSI(binary_GT, binary_R)))  # 保留四位有效数字

    print('{0:.4}'.format(calVOE(binary_GT, binary_R)))

    print('{0:.4}'.format(calRVD(binary_GT, binary_R)))

    print('{0:.4}'.format(calASSD(binary_GT, binary_R)))

    print('{0:.4}'.format(calRMSD(binary_GT, binary_R)))
    print('{0:.4}'.format(calMaxD(binary_GT, binary_R)))