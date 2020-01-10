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
data_gen_args = dict(rotation_range=3,
                    width_shift_range=0.01,
                    height_shift_range=0.01,
                    shear_range=0.01,
                    zoom_range=0.01,
                    fill_mode='nearest')

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)
outputPath = "./data_train/train_liver.h5"

if os.path.exists(outputPath):
  os.remove(outputPath)
dataset = HDF5DatasetWriter(image_dims=(1148, 512, 512, 1),
                            mask_dims=(1148, 512, 512, 1),
                            outputPath=outputPath)

windowW = 280
windowC = -10
#part1
for i in range(1,18): # 前17个人作为测试集
   # 注意不同的系统，文件分割符的区别
   label_path = './3Dircadb/3Dircadb1.'
   files = glob.glob(label_path +str(i)+'/MASKS_DICOM/livertumor'+'*'+'/'+'*')
   liver_slices = [pydicom.dcmread(file)for file in files]
   # 注意需要排序，即使文件夹中显示的是有序的，读进来后就是随机的了
   liver_slices.sort(key = lambda x: int(x.InstanceNumber))
   # s.pixel_array 获取dicom格式中的像素值
   if len(liver_slices)==0:
      break
   livers = np.stack([s.pixel_array for s in liver_slices])

   # 肝的原分割图
   image_slices = [pydicom.dcmread(s) for s in glob.glob(label_path + str(i) + '/MASKS_DICOM/liver/' + '*')]
   image_slices.sort(key=lambda x: int(x.InstanceNumber))
   # ================================================
   images = get_pixels_hu(image_slices)
   images[images <= 0] = 0
   images[images > 0] = 1

   # 原图
   image_yuan = [pydicom.dcmread(s) for s in glob.glob(label_path + str(i) + '/PATIENT_DICOM/' + '*')]
   image_yuan.sort(key=lambda x: int(x.InstanceNumber))
   # ================================================
   images_yuan = get_pixels_hu(image_yuan)
   images_yuan = transform_ctdata(images_yuan, windowW, windowC)

   images = np.multiply(images_yuan, images)


   for i in range(int(livers.shape[0]/129)):
      full_images = []  # 后面用来存储目标切片的列表
      full_livers = []  # 功能同上
      new_liver = livers[i*129:i*129+129]
      start, end = getRangImageDepth(new_liver)

      new_images = clahe_equalized(images, start, end)

      new_images /= 255.

      total = (end - 4) - (start + 4) + 1
      print("%d person, total slices %d" % (i, total))
      # 首和尾目标区域都太小，舍弃
      new_images = new_images[start + 5:end - 5]
      print("%d person, images.shape:(%d,)" % (i, new_images.shape[0]))
      # if new_images.shape[0]==0:
      #    break
      new_liver[new_liver > 0] = 1

      new_liver = new_liver[start + 5:end - 5]

      # =================================================
      # full_images= np.concatenate(full_images,images)
      # full_livers= np.concatenate(full_livers,livers)
      full_images.append(new_images)
      full_livers.append(new_liver)

      full_images = np.vstack(full_images)
      full_images = np.expand_dims(full_images, axis=-1)
      full_livers = np.vstack(full_livers)
      full_livers = np.expand_dims(full_livers, axis=-1)

      image_datagen.fit(full_images, augment=True, seed=seed)
      mask_datagen.fit(full_livers, augment=True, seed=seed)
      image_generator = image_datagen.flow(full_images, seed=seed)
      mask_generator = mask_datagen.flow(full_livers, seed=seed)

      train_generator = zip(image_generator, mask_generator)
      x = []
      y = []
      i = 0

      try:
         for x_batch, y_batch in train_generator:
            i += 1
            x.append(x_batch)
            y.append(y_batch)
            if i >= 2:  # 因为我不需要太多的数据
               break
      except ZeroDivisionError:
         break
      else:
         x = np.vstack(x)
         y = np.vstack(y)
         # ===================================================
         # part4 接part3
         dataset.add(full_images, full_livers)
         dataset.add(x, y)
         print('add once finished.')


dataset.close()





