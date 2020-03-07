# Unet_Window
This network model contains UNet code, which was proposed by Ronneberger et al. In “U-Net: Convolutional Networks for Biomedical Image Segmentation”, which has been developed and widely used. This network model is introduced in “Intensity windowing of CT images for automatic liver and tumor segmentation”.
This is a network model for segmenting the liver and its lesions. On the basis of UNet, a window technology is added. The 3DIRCADb data set is segmented step by step and the best CT value ranges of the liver and its lesions are obtained. Generate _ *. Py completes all the routine data extraction and data preprocessing work (including grayscale, equalization, normalization, etc.), where adjustment of the window width and window level is the focus of improvement. Unet.py: Network structure definition of unet. Test_and_loss.py: Unet test evaluation.

# Method 
Step one: apply different intensity windows to the original image to find more possible organization details as preprocessing, adjust the window width and window level to segment the liver and perform contrast experiments to study the segmentation effect of CT values in various ranges on UNet. Step two: the liver region segmented in the previous step is mapped to the original image, and then different intensity windows are used for tumor detection and segmentation, a comparative experiment similar to liver segmentation is done. The optimal segmentation CT value range and corresponding optimal segmentation results of liver and lesion were obtained. This preprocessing is aimed at the original format of medical images, and uses as much of the original information of 16 bits images as possible. And taking into account the difference between the CT value of the liver and the lesion, different pretreatments are performed for the liver and the lesion to achieve segmentation, which can improve the accuracy of the segmentation.

# Dataset
The 3DIRCADb dataset includes 20 venous phase enhanced CT volumes from various European hospitals with different CT scanners. And the dataset is stored in DICOM format. Available on http://ircad.fr/research/3d-ircadb-01 


# Result
UNet applied with this preprocessing gets higher DICE scores than the original segmentation result,and the experiment obtained the optimal CT value range based on 3DIRCADb. The optimal window width range for liver segmentation is 200HU~240HU and the window level range is 30HU~50HU, so the optimal CT value range is -90HU~170HU.The optimal window width for lesion segmentation is 130HU~150HU and the window level is 50HU~70HU, so the optimal CT value range is -25HU~145HU.

### **预测结果展示**	

> 注: 	  		gt 为 groundTruth,  		  pred 为 模型预测结果


