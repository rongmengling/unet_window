import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import ImageDataGenerator
from HDF5DatasetWriter import HDF5DatasetWriter
from HDF5DatasetGenerator import HDF5DatasetGenerator
from utils import *
from tqdm import tqdm
import cv2
import glob
# 可以在part1之前设定好（即循环外）
seed=1
# data_gen_args = dict(rotation_range=3,
#                     width_shift_range=0.01,
#                     height_shift_range=0.01,
#                     shear_range=0.01,
#                     zoom_range=0.01,
#                     fill_mode='nearest')

image_datagen = ImageDataGenerator()
mask_datagen = ImageDataGenerator()
print('build ImageDataGenerator finished.')

# 可以在part1之前设定好（即循环外）
# 这儿的数量需要提前写好，感觉很不方便，但不知道怎么改，我是先跑了之前的程序，计算了一共有多少
# 张图片后再写的，但这样明显不是好的解决方案
outputPath = "./data_train/val_liver.h5"

if os.path.exists(outputPath):
  os.remove(outputPath)

windowW = 280
windowC = -10
print('build HDF5DatasetWriter finished')

full_images2 = []
full_livers2 = []
for i in range(18,21):#后3个人作为测试样本
    # 注意不同的系统，文件分割符的区别we
    label_path = './3Dircadb/3Dircadb1.'
    files = glob.glob(label_path + str(i) + '/MASKS_DICOM/livertumor' + '*' + '/' + '*')
    liver_slices = [pydicom.dcmread(file) for file in files]
    # 注意需要排序，即使文件夹中显示的是有序的，读进来后就是随机的了
    liver_slices.sort(key=lambda x: int(x.InstanceNumber))
    # s.pixel_array 获取dicom格式中的像素值
    if len(liver_slices) == 0:
        break
    livers = np.stack([s.pixel_array for s in liver_slices])

    #肝的原分割图
    image_slices = [pydicom.dcmread(s) for s in glob.glob(label_path + str(i) + '/MASKS_DICOM/liver/' + '*')]
    image_slices.sort(key=lambda x: int(x.InstanceNumber))
    # ================================================
    images = get_pixels_hu(image_slices)
    images[images <= 0] = 0
    images[images > 0] = 1

    #原图
    image_yuan = [pydicom.dcmread(s) for s in glob.glob(label_path + str(i) + '/PATIENT_DICOM/' + '*')]
    image_yuan.sort(key=lambda x: int(x.InstanceNumber))
    # ================================================
    images_yuan = get_pixels_hu(image_yuan)
    images_yuan = transform_ctdata(images_yuan, windowW, windowC)

    images = np.multiply(images_yuan,images)

    start, end = getRangImageDepth(livers)
    images = clahe_equalized(images,start,end)
    images /= 255.
    images = images[start+5:end-5]
    print("%d person, images.shape:(%d,)"%(i,images.shape[0]))
    livers[livers>0] = 1
    livers = livers[start+5:end-5]

    full_images2.append(images)
    full_livers2.append(livers)
    
full_images2 = np.vstack(full_images2)
full_images2 = np.expand_dims(full_images2,axis=-1)
full_livers2 = np.vstack(full_livers2)
full_livers2 = np.expand_dims(full_livers2,axis=-1)

dataset = HDF5DatasetWriter(image_dims=(full_images2.shape[0], full_images2.shape[1], full_images2.shape[2], 1),
                            mask_dims=(full_images2.shape[0], full_images2.shape[1], full_images2.shape[2], 1),
                            outputPath=outputPath)


dataset.add(full_images2, full_livers2)

print("total images in val ",dataset.close())
