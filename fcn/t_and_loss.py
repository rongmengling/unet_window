import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from BagData import test_dataloader
import loss_utils


floatDSI =0.0
floatVOE =0.0
floatRVD =0.0
floatASSD =0.0
floatRMSD =0.0
floatMaxD =0.0

minDSI =0.0
minVOE =0.0
minRVD =0.0
minASSD =0.0
minRMSD =0.0
minMaxD =0.0

totalDSI =0.0
totalVOE =0.0
totalRVD =0.0
totalASSD =0.0
totalRMSD =0.0
totalMaxD =0.0
num = 0
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

for index, (bag, bag_msk1) in enumerate(test_dataloader):
    bag = bag.to(device)
    bag_msk1 = bag_msk1.to(device)
    optimizer.zero_grad()
    output1 = fcn_model(bag)
    # output = torch.sigmoid(output)  # output.shape is torch.Size([4, 2, 160, 160])
    # loss = criterion(output, bag_msk)
    # iter_loss = loss.item()
    # test = output.data.cpu().numpy()
    # test2 = np.squeeze(test)
    output_np = output1.cpu().detach().numpy().copy()
    output = np.uint8((np.uint8(output_np[0][0]))<130)*255
    bag_msk_np = bag_msk1.cpu().detach().numpy().copy()
    bag_msk = (bag_msk_np[0][0])*255

    DSI = loss_utils.calDSI(bag_msk, output)
    VOE = loss_utils.calVOE(bag_msk, output)
    RVD = loss_utils.calRVD(bag_msk, output)
    ASSD = loss_utils.calASSD(bag_msk, output)
    # nan后应该等于多少不确定
    if ASSD != ASSD:
        ASSD = 0.0
    RMSD = loss_utils.calRMSD(bag_msk, output)
    if RMSD != RMSD:
        RMSD = 0.0
    MaxD = loss_utils.calMaxD(bag_msk, output)
    if MaxD != MaxD:
        MaxD = 0.0
    print('（1）DICE计算结果，      DICE       = {0:.4}'.format(DSI))  # 保留四位有效数字
    print('（2）VOE计算结果，       VOE       = {0:.4}'.format(VOE))
    print('（3）RVD计算结果，       RVD       = {0:.4}'.format(RVD))
    print('（3）ASSD计算结果，      ASSD      = {0:.4}'.format(ASSD))
    print('（4）RMSD计算结果，      RMSD      = {0:.4}'.format(RMSD))
    print('（5）MaxD计算结果，      MaxD      = {0:.4}'.format(MaxD) + '\n\n')
    if DSI > floatDSI:
        floatDSI = DSI
        floatVOE = VOE
        floatRVD = RVD
        floatASSD = ASSD
        floatRMSD = RMSD
        floatMaxD = MaxD
    if DSI < minDSI and DSI > 0:
        minDSI = DSI
        minVOE = VOE
        minRVD = RVD
        minASSD = ASSD
        minRMSD = RMSD
        minMaxD = MaxD
    totalDSI += DSI
    totalVOE += VOE
    totalRVD += RVD
    totalASSD += ASSD
    totalRMSD += RMSD
    totalMaxD += MaxD
    num += 1
print('\n\n\n\n最大最小以 DSI为对比标准\n')
print('    max DICE       = {0:.4}'.format(floatDSI))
print('    max  VOE       = {0:.4}'.format(floatVOE))
print('    max  RVD       = {0:.4}'.format(floatRVD))
print('    max ASSD       = {0:.4}'.format(floatASSD))
print('    max RMSD       = {0:.4}'.format(floatRMSD))
print('    max MaxD       = {0:.4}'.format(floatMaxD) + '\n')

print('    min DICE       = {0:.4}'.format(minDSI))
print('    min  VOE       = {0:.4}'.format(minVOE))
print('    min  RVD       = {0:.4}'.format(minRVD))
print('    min ASSD       = {0:.4}'.format(minASSD))
print('    min RMSD       = {0:.4}'.format(minRMSD))
print('    min MaxD       = {0:.4}'.format(minMaxD) + '\n')

print('    average DICE       = {0:.4}'.format(totalDSI / num))
print('    average  VOE       = {0:.4}'.format(totalVOE / num))
print('    average  RVD       = {0:.4}'.format(totalRVD / num))
print('    average ASSD       = {0:.4}'.format(totalASSD / num))
print('    average RMSD       = {0:.4}'.format(totalRMSD / num))
print('    average MaxD       = {0:.4}'.format(totalMaxD / num) + '\n')