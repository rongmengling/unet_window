import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from HDF5DatasetWriter import HDF5DatasetWriter
from HDF5DatasetGenerator import HDF5DatasetGenerator
from skimage import io
import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt

val_outputPath = 'D:/BaiduNetdiskDownload/Unet/Unet/data_train/val_liver.h5'
bag_data = './bag_data'
bag_data_msk = './bag_data_msk'
image_outputPath = 'D:/BaiduNetdiskDownload/Unet/Unet/data_train/train_liver.h5'

test_reader = HDF5DatasetGenerator(dbPath=val_outputPath, batchSize=152)
test_iter = test_reader.generator()
fixed_test_images, fixed_test_masks = test_iter.__next__()

num = 0
for i in range(152):
    temp_images = fixed_test_images[i]
    temp_masks = fixed_test_masks[i]
    image2 = (temp_images[:, :, 0] * 255.).astype(np.uint8)
    gt = (temp_masks[ :, :, 0] * 255.).astype(np.uint8)
    io.imsave(os.path.join(bag_data, str(num) + '.png'), image2)
    io.imsave(os.path.join(bag_data_msk, str(num) + '.png'), gt)
    num += 1

#2934

test_reader2 = HDF5DatasetGenerator(dbPath=image_outputPath, batchSize=2782)
test_iter2 = test_reader2.generator()
fixed_test_images2, fixed_test_masks2 = test_iter2.__next__()
for i in range(2782):
    temp_images = fixed_test_images2[i]
    temp_masks = fixed_test_masks2[i]
    image2 = (temp_images[:, :, 0] * 255.).astype(np.uint8)
    gt = (temp_masks[ :, :, 0] * 255.).astype(np.uint8)
    io.imsave(os.path.join(bag_data, str(num) + '.png'), image2)
    io.imsave(os.path.join(bag_data_msk, str(num) + '.png'), gt)
    num += 1
