import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from BagData import test_dataloader
import loss_utils
from PIL import Image

model_file = './checkpoints/fcn_model_95.pt'

full_images2 = []
full_livers2 = []
show_vgg_params=False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# vgg_model = VGGNet(requires_grad=True, show_params=show_vgg_params)
# fcn_model = FCNs(pretrained_net=vgg_model, n_class=2)
# fcn_model = fcn_model.to(device)
# fcn_model.load_state_dict(torch.load(model_file))
fcn_model = torch.load(model_file)
fcn_model.eval()

criterion = nn.BCELoss().to(device)
optimizer = optim.SGD(fcn_model.parameters(), lr=1e-2, momentum=0.7)

num1 = 25
num2 = 60
num3 = 101
num4 = 119

# dcm_img = Image.fromarray(img_label_dcm)
# dcm_img = dcm_img.convert('L')
# dcm_img.save('temp' + str(time.time()) + '.jpg')
# dcm_img.show()

def msk(bag, bag_msk1,fcn_model,num):
    bag = bag.to(device)
    bag_msk1 = bag_msk1.to(device)
    optimizer.zero_grad()
    output1 = fcn_model(bag)
    output_np = output1.cpu().detach().numpy().copy()
    output = np.uint8((np.uint8(output_np[0][0]))<130)*255
    out_img = Image.fromarray(np.uint8(output))
    out_img = out_img.convert('L')
    out_img.save('out_' + str(num) + '.jpg')
    out_img.show()
    bag_msk_np = bag_msk1.cpu().detach().numpy().copy()
    # bag_msk = np.squeeze(np.argmin(bag_msk_np, axis=1))
    bag_msk = (bag_msk_np[0][0])*255
    msk_img = Image.fromarray(np.uint8(bag_msk))
    msk_img = msk_img.convert('L')
    msk_img.save('mask_' + str(num) + '.jpg')
    msk_img.show()
    pass


for index, (bag, bag_msk1) in enumerate(test_dataloader):
    if index ==  num1:
        msk(bag,bag_msk1,fcn_model,num1)

    if index == num2:
        msk(bag,bag_msk1,fcn_model,num2)

    if index == num3:
        msk(bag,bag_msk1,fcn_model,num3)

    if index == num4:
        msk(bag,bag_msk1,fcn_model,num4)