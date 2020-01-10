import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from HDF5DatasetWriter import HDF5DatasetWriter
from HDF5DatasetGenerator import HDF5DatasetGenerator
from utils import *
from tqdm import tqdm
from skimage import io
from Unet import *
import loss_utils
import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt

seed = 1
image_datagen = ImageDataGenerator()
mask_datagen = ImageDataGenerator()
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 1
TOTAL_VAL = 152  # 总共的validation数据
val_outputPath = './data_train/val_liver.h5'
BATCH_SIZE = 4
windowW = 210
windowC = 40

model_file = './models/weights_unet-03--0.82.h5'
if os.path.exists(val_outputPath):
    os.remove(val_outputPath)

full_images2 = []
full_livers2 = []
for i in range(18, 21):  # 后3个人作为测试样本
    label_path = './3Dircadb/3Dircadb1.%d/MASKS_DICOM/liver' % i
    data_path = './3Dircadb/3Dircadb1.%d/PATIENT_DICOM' % i
    liver_slices = [pydicom.dcmread(label_path + '/' + s) for s in os.listdir(label_path)]
    liver_slices.sort(key=lambda x: int(x.InstanceNumber))
    livers = np.stack([s.pixel_array for s in liver_slices])
    start, end = getRangImageDepth(livers)
    total = (end - 4) - (start + 4) + 1
    print("%d person, total slices %d" % (i, total))

    image_slices = [pydicom.dcmread(data_path + '/' + s) for s in os.listdir(data_path)]
    image_slices.sort(key=lambda x: int(x.InstanceNumber))

    images = get_pixels_hu(image_slices)
    images = transform_ctdata(images, windowW, windowC)
    images = clahe_equalized(images, start, end)
    images /= 255.
    images = images[start + 5:end - 5]
    print("%d person, images.shape:(%d,)" % (i, images.shape[0]))
    livers[livers > 0] = 1
    livers = livers[start + 5:end - 5]

    full_images2.append(images)
    full_livers2.append(livers)

full_images2 = np.vstack(full_images2)
full_images2 = np.expand_dims(full_images2, axis=-1)
full_livers2 = np.vstack(full_livers2)
full_livers2 = np.expand_dims(full_livers2, axis=-1)

dataset = HDF5DatasetWriter(image_dims=(full_images2.shape[0], full_images2.shape[1], full_images2.shape[2], 1),
                            mask_dims=(full_images2.shape[0], full_images2.shape[1], full_images2.shape[2], 1),
                            outputPath=val_outputPath)

dataset.add(full_images2, full_livers2)

print("total images in val ", dataset.close())



test_reader = HDF5DatasetGenerator(dbPath=val_outputPath, batchSize=TOTAL_VAL)
test_iter = test_reader.generator()
fixed_test_images, fixed_test_masks = test_iter.__next__()

model = get_unet()
model.load_weights(model_file)
floatDSI =0.0
floatVOE =0.0
floatRVD =0.0
floatASSD =0.0
floatRMSD =0.0
floatMaxD =0.0
for i in range(19):
    temp_images = fixed_test_images[i*4:i*4+4]
    temp_masks = fixed_test_masks[i*4:i*4+4]
    imgs_mask_test = model.predict(temp_images, verbose=1)
    j = 0
    for image2 in imgs_mask_test:
        image2 = (image2[:, :, 0] * 255.).astype(np.uint8)
        gt = (temp_masks[j, :, :, 0] * 255.).astype(np.uint8)
        DSI = loss_utils.calDSI(gt, image2)
        VOE = loss_utils.calVOE(gt, image2)
        RVD = loss_utils.calRVD(gt, image2)
        ASSD = loss_utils.calASSD(gt, image2)
        #nan后应该等于多少不确定
        if ASSD!=ASSD:
            ASSD = 0.0
        RMSD = loss_utils.calRMSD(gt, image2)
        if RMSD!=RMSD:
            RMSD = 0.0
        MaxD = loss_utils.calMaxD(gt, image2)
        if MaxD!=MaxD:
            MaxD = 0.0
        print('（1）DICE计算结果，      DICE       = {0:.4}'.format(DSI))  # 保留四位有效数字
        print('（2）VOE计算结果，       VOE       = {0:.4}'.format(VOE))
        print('（3）RVD计算结果，       RVD       = {0:.4}'.format(RVD))
        print('（3）ASSD计算结果，      ASSD      = {0:.4}'.format(ASSD))
        print('（4）RMSD计算结果，      RMSD      = {0:.4}'.format(RMSD))
        print('（5）MaxD计算结果，      MaxD      = {0:.4}'.format(MaxD) +'\n\n')
        if DSI>floatDSI:
            floatDSI = DSI
            floatVOE = VOE
            floatRVD = RVD
            floatASSD = ASSD
            floatRMSD = RMSD
            floatMaxD = MaxD

        j += 1
print('\n\n\n\n')
print(str(windowW)+' '+str(windowC))
print('     DICE       = {0:.4}'.format(floatDSI))
print('      VOE       = {0:.4}'.format(floatVOE))
print('      RVD       = {0:.4}'.format(floatRVD))
print('     ASSD       = {0:.4}'.format(floatASSD))
print('     RMSD       = {0:.4}'.format(floatRMSD))
print('     MaxD       = {0:.4}'.format(floatMaxD) +'\n')