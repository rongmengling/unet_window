import numpy as np
import torch
import torch.nn.functional as F

def get_acc(logit, target):
    '''
    :param logit: (batch size, channel, width, height). Channel represent onehot
    :param target: (batch size, channel, width, height)
    :return: acc
    '''

    bs, c, w, h = logit.size()
    num_classes = c

    logit = F.softmax(logit)
    res = (logit * target).sum()
    print(res)

    return res


if __name__ == '__main__':

    logit = torch.randn(2,2,2,2)
    target = torch.randn(2,2,2,2)

    # print('logit:', logit)
    # print('target:', target)
    
    get_acc(logit, target)

