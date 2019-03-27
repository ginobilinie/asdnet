#!/usr/bin/env python
"""
The Residual FCN is implemented, and this is the calling main method.
By Dong Nie
07/22/2017
"""

from __future__ import absolute_import


import sys
import time


import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from mySegNet import *
from torch.autograd import Variable
from utils import *


def main():
    nclasses = 15
    train = True
    niter = 100
    times = torch.FloatTensor(niter)

    batch_size = 1
    nchannels = 3
    height = 360
    width = 480
    
    path_test='/home/dongnie/warehouse/BrainEstimation'
    path_patients_h5='/home/dongnie/warehouse/pelvicSeg/pelvicH5'
    batch_size=10
    data_generator = Generator_2D_slices(path_patients_h5,batch_size,inputKey='dataMR2D',outputKey='dataSeg2D')

    model = ENet(nclasses)
    loss = nn.NLLLoss2d()
    softmax = nn.Softmax()

    model.cuda()
    loss.cuda()
    softmax.cuda()

    optimizer = torch.optim.Adam(model.parameters())

    for i in range(niter):
#         x = torch.FloatTensor(
#             torch.randn(batch_size, nchannels, height, width))
#         y = torch.LongTensor(batch_size, height, width)
#         y.random_(nclasses)
# 
# #pinned memory is page-locked memory. It is easy for users to shoot themselves in the foot 
# #if they enable page-locked memory for everything, because it cant be pre-empted. 
#         x.pin_memory()
#         y.pin_memory()
# 
#         input = Variable(x.cuda(async=True))
#         target = Variable(y.cuda(async=True))
# 
        sys.stdout.write('\r{}/{}'.format(i, niter))

        print('iter %d'%i)
        running_loss=0.0
        inputs,labels=data_generator.next()
        print('inputs shape is', inputs.shape)
        print('labels shape is', labels.shape)
        #inputs=inputs[:,:,:,2]
        #inputs=inputs[:,:,:,2]
        
        labels=np.squeeze(labels)
        labels=labels.astype(int)
        #inputs=np.expand_dims(inputs,axis=1)
        #labels=np.expand_dims(labels,axis=1)
        print('inputs shape is', inputs.shape)
        print('labels shape is', labels.shape)
        inputs=torch.from_numpy(inputs)
        labels=torch.from_numpy(labels)
        inputs=inputs.cuda()
        labels=labels.cuda()

        #we should consider different data to train
        #wrap them into Variable
        inputs,labels=Variable(inputs),Variable(labels)

        start = time.time()

        if train:
            optimizer.zero_grad()
            model.train()
        else:
            model.eval()

        outputs = model(inputs)

        if train:
            loss_ = loss.forward(outputs, labels)
            loss_.backward()
            optimizer.step()
        else:
            output_2d = outputs.view(height * width, nclasses)
            pred = softmax(output_2d).view(outputs.size())

        times[i] = time.time() - start
    sys.stdout.write('\n')

    print('average time per image {:04f} sec'.format(
        times.mean()))

if __name__ == '__main__':
    main()
