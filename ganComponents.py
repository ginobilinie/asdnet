'''
    Basic Components for GAN: Regressor,Segmentor, Discriminator
    Dong Nie
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nnBuildUnits import *
from utils import weights_init


class Segmentor(nn.Module):
    def __init__(self):
        super(Segmentor,self).__init__()
        
'''
  Basic UNet arch
'''
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, TModule=None, FModule=False, nd=2):
        super(UNet, self).__init__()
        #self.imsize = imsize
        assert nd==1 or nd==2 or nd==3, 'nd is not correctly specified!!!!, it should be {1,2,3}'
        self.activation = F.relu
        
        self.pool1 = maxPool23DUinit(kernel_size=3,stride=2,padding=1, dilation=1, nd=nd)
        self.pool2 = maxPool23DUinit(kernel_size=3,stride=2,padding=1, dilation=1, nd=nd)
        self.pool3 = maxPool23DUinit(kernel_size=3,stride=2,padding=1, dilation=1, nd=nd)
        
        #self.pool4 = nn.MaxPool2d(2)

        self.conv_block1_64 = unetConvUnit(in_channels, 64, nd=nd)
        self.conv_block64_128 = unetConvUnit(64, 128, nd=nd)
        self.conv_block128_256 = unetConvUnit(128, 256, nd=nd)
        self.conv_block256_512 = unetConvUnit(256, 512, nd=nd)
        #self.conv_block512_1024 = unetConvUnit(512, 1024)

        #self.up_block1024_512 = unetUpUnit(1024, 512)
        self.up_block512_256 = unetUpUnit(512, 256, 256, isFModule=FModule, nd=nd)
        self.up_block256_128 = unetUpUnit(256, 128, 128, isFModule=FModule, nd=nd)
        self.up_block128_64 = unetUpUnit(128, 64, 64, isTModule= TModule, isFModule=FModule, nd=nd)

        self.last = nn.Conv2d(64, out_channels, 1)


    def forward(self, x):
        block1 = self.conv_block1_64(x)
        pool1 = self.pool1(block1)

        block2 = self.conv_block64_128(pool1)
        pool2 = self.pool2(block2)

        block3 = self.conv_block128_256(pool2)
        pool3 = self.pool3(block3)

        block4 = self.conv_block256_512(pool3)

        up2 = self.up_block512_256(block4, block3)
        up3 = self.up_block256_128(up2, block2)
        up4 = self.up_block128_64(up3, block1)
        return self.last(up4)


'''
Residual Learning for FCN: short-range residual connection and long-range U-Net concatenation
We use spatial dropout after each layer group
We use stochastic long-range connection
We use small dilation in the middle layers
In this version, I use deep (inter-mediate) supervision mechanism to help training
We also include the dilated residual module in the highest resolution's (HR) lateral connection
'''
class HRResUNet_DS(nn.Module):
    def __init__(self, in_channels, out_channels, isRandomConnection=None, isSmallDilation=None, isSpatialDropOut=None,
                 dropoutRate=0.25, TModule=None, FModule=False, nd=2):
        super(HRResSegNet_DS, self).__init__()
        # self.imsize = imsize

        assert nd == 1 or nd == 2 or nd == 3, 'nd is not correctly specified!!!!, it should be {1,2,3}'

        self.isSpatialDropOut = isSpatialDropOut
        self.isRandomConnection = isRandomConnection
        self.isSmallDilation = isSmallDilation

        self.activation = F.relu

        self.pool1 = maxPool23DUinit(kernel_size=3, stride=2, padding=1, dilation=1, nd=nd)
        self.pool2 = maxPool23DUinit(kernel_size=3, stride=2, padding=1, dilation=1, nd=nd)
        self.pool3 = maxPool23DUinit(kernel_size=3, stride=2, padding=1, dilation=1, nd=nd)
        # self.pool4 = nn.MaxPool2d(2)

        self.conv_block1_32 = conv23D_bn_relu_Unit(in_channels, 32, 3, padding=1, nd=nd)
        self.conv_block32_32 = residualUnit3(32, 32, isDilation=False, isEmptyBranch1=False, nd=nd)
        self.conv_block32_64 = residualUnit3(32, 64, isDilation=False, isEmptyBranch1=False, nd=nd)
        self.conv_block64_128 = residualUnit3(64, 128, isDilation=False, isEmptyBranch1=False, nd=nd)

        # the residual layers on the smallest resolution
        self.conv_block128_256 = residualUnit3(128, 256, isDilation=isSmallDilation, isEmptyBranch1=False, nd=nd)

        self.dropout23d = dropout23DUnit(prob=dropoutRate, nd=nd)


        self.up_block256_128 = resUnetUpUnit(256, 128, 128, spatial_dropout_rate=0, isFModule=FModule, nd=nd)
        self.up_block128_64 = resUnetUpUnit(128, 64, 64, spatial_dropout_rate=0, isFModule=FModule, nd=nd)
        self.up_block64_32 = resUnetUpUnit(64, 32, 32, spatial_dropout_rate=0, isTModule=TModule, isFModule=FModule, nd=nd)

        # main_output: the original main path
        self.last_main = conv23D_bn_relu_Unit(32, out_channels, 1, nd=nd)

        # intermediate_output1: upsampling 2 times: we firstly use plain conv and then use bilinearly upsmapling
        self.conv_path1 = conv23D_bn_relu_Unit(64, out_channels, 1, nd=nd)
        if nd == 2:
            self.last_path1 = nn.Upsample(scale_factor=2, mode='bilinear')
        elif nd == 3:
            self.last_path1 = nn.Upsample(scale_factor=2, mode='trilinear')

        # intermediate_output2: upsampling 4 times
        self.conv_path2 = conv23D_bn_relu_Unit(128, out_channels, 1, nd=nd)
        if nd == 2:
            self.last_path2 = nn.Upsample(scale_factor=4, mode='bilinear')
        elif nd == 3:
            self.last_path2 = nn.Upsample(scale_factor=4, mode='trilinear')

    def forward(self, x):
        block0 = self.conv_block1_32(x)
        block1 = self.conv_block32_32(block0)
        pool1 = self.pool1(block1)
        if self.isSpatialDropOut:
            pool1 = self.dropout23d(pool1)

        block2 = self.conv_block32_64(pool1)
        pool2 = self.pool2(block2)
        if self.isSpatialDropOut:
            pool2 = self.dropout23d(pool2)

        block3 = self.conv_block64_128(pool2)
        pool3 = self.pool3(block3)
        if self.isSpatialDropOut:
            pool3 = self.dropout23d(pool3)

        block4 = self.conv_block128_256(pool3)  # dilation in the smallest resolution

        up2 = self.up_block256_128(block4, block3)
        up3 = self.up_block128_64(up2, block2)
        up4 = self.up_block64_32(up3, block4)

        # for path1:2 times upsampling
        conv_path1 = self.conv_path1(up3)

        # for path2: 4 times upsampling
        conv_path2 = self.conv_path2(up2)

        #         return F.log_softmax(self.last(up4))
        return self.last_main(up4), self.last_path1(conv_path1), self.last_path2(conv_path2)


'''
Residual Learning for FCN: short-range residual connection and long-range residual connection
We use more residual layers, especially one more layer after each residual layer which have a non-empty branch1
We use spatial dropout after each layer group
We use stochastic long-range connection
We use small dilation in the middle layers
'''
class DeeperResSegNet(nn.Module):
    def __init__(self, in_channels, out_channels, isRandomConnection = False, isSmallDilation = True, isSpatialDropOut = True, dropoutRate=0.25, nd=2):
        super(DeeperResSegNet, self).__init__()

        self.isSpatialDropOut = isSpatialDropOut
        
        assert nd==1 or nd==2 or nd==3, 'nd is not correctly specified!!!!, it should be {1,2,3}'
        self.activation = F.relu

        self.pool1 = maxPool23DUinit(kernel_size=3,stride=2, padding=1, dilation=1, nd=nd)
        self.pool2 = maxPool23DUinit(kernel_size=3,stride=2, padding=1, dilation=1, nd=nd)
        self.pool3 = maxPool23DUinit(kernel_size=3,stride=2, padding=1, dilation=1, nd=nd)


        self.conv1_block1_64 = conv23D_bn_relu_Unit(in_channels, 32, 3, nd=nd)
        self.conv1_block64_64 = residualUnit3(32, 64, isEmptyBranch1=False, nd=nd)

        self.conv2_block64_128 = residualUnit3(64, 128, nd=nd)
        self.conv2_block128_128 = residualUnit3(128, 128, isEmptyBranch1=False, nd=nd)
        
        self.conv3_block128_256 = residualUnit3(128, 256, nd=nd)
        self.conv3_block256_256 = residualUnit3(256, 256, isEmptyBranch1=False, nd=nd)
        
        #dilated on the smallest resolution
        self.conv4_block256_512 = residualUnit3(256, 512, isDilation=isSmallDilation, nd=nd)
        
        self.dropout23d = dropout23DUnit(prob=dropoutRate, nd=nd)

        if isRandomConnection:
            self.up_block512_256 = ResUpUnit(512, 256, spatial_dropout_rate = 0.1, nd=nd)
            self.up_block256_128 = ResUpUnit(256, 128, spatial_dropout_rate = 0.05, nd=nd)
            
            self.up_block128_64 = ResUpUnit(128, 64, spatial_dropout_rate = 0.01, nd=nd)
        else:
            self.up_block512_256 = ResUpUnit(512, 256, spatial_dropout_rate = 0, nd=nd)
            self.up_block256_128 = ResUpUnit(256, 128, spatial_dropout_rate = 0, nd=nd)
            self.up_block128_64 = ResUpUnit(128, 64, spatial_dropout_rate = 0, nd=nd)

        self.last = nn.Conv2d(64, out_channels, 1)
        
    def forward(self, x):   
        block0 = self.conv1_block1_64(x)
        block1 = self.conv1_block64_64(block0)
        pool1 = self.pool1(block1)
        if self.isSpatialDropOut:
            pool1 = self.dropout23d(pool1)

        block2 = self.conv2_block64_128(pool1)
        block2a = self.conv2_block128_128(block2)
        pool2 = self.pool2(block2a)
        if self.isSpatialDropOut:
            pool2 = self.dropout23d(pool2)

        block3 = self.conv3_block128_256(pool2)
        block3a = self.conv3_block256_256(block3)
        pool3 = self.pool3(block3a)
        if self.isSpatialDropOut:
            pool3 = self.dropout23d(pool3)

        block4 = self.conv4_block256_512(pool3)

        up2 = self.up_block512_256(block4, block3)
        up3 = self.up_block256_128(up2, block2)
        up4 = self.up_block128_64(up3, block1)
#         return F.log_softmax(self.last(up4))
        return self.last(up4)


'''
Residual Learning for FCN: short-range residual connection and long-range residual connection
We use spatial dropout after each layer group
We use stochastic long-range connection
We use small dilation in the middle layers
'''
class ResSegNet(nn.Module):
    def __init__(self, in_channels, out_channels, isRandomConnection=None, isSmallDilation=None, isSpatialDropOut=None, dropoutRate=0.25, nd=2):
        super(ResSegNet, self).__init__()
        #self.imsize = imsize
        
        assert nd==1 or nd==2 or nd==3, 'nd is not correctly specified!!!!, it should be {1,2,3}'
    
        self.isSpatialDropOut = isSpatialDropOut
        self.isRandomConnection = isRandomConnection
        self.isSmallDilation = isSmallDilation
        
        self.activation = F.relu
        
        self.pool1 = maxPool23DUinit(kernel_size=3, stride=2, padding=1, dilation=1, nd=nd)
        self.pool2 = maxPool23DUinit(kernel_size=3, stride=2, padding=1, dilation=1, nd=nd)
        self.pool3 = maxPool23DUinit(kernel_size=3, stride=2, padding=1, dilation=1, nd=nd)

        self.conv_block1_64 = conv23D_bn_relu_Unit(in_channels, 32, 3, padding=1, nd=nd)
        self.conv_block64_64 = residualUnit3(32, 32, isDilation=False,isEmptyBranch1=False, nd=nd)

        self.conv_block64_128 = residualUnit3(32, 64, isDilation=False,isEmptyBranch1=False, nd=nd)
        
        self.conv_block128_256 = residualUnit3(64, 128, isDilation=False,isEmptyBranch1=False, nd=nd)
        
        #the residual layers on the smallest resolution
        self.conv_block256_512 = residualUnit3(128, 256, isDilation=isSmallDilation, isEmptyBranch1=False, nd=nd)

        self.dropout23d = dropout23DUnit(prob=dropoutRate, nd=nd)

        if isRandomConnection:
            self.up_block512_256 = ResUpUnit(256, 128, spatial_dropout_rate = 0.1, nd=nd)
            self.up_block256_128 = ResUpUnit(128, 64, spatial_dropout_rate = 0.05, nd=nd)
        
            self.up_block128_64 = ResUpUnit(64, 32, spatial_dropout_rate = 0.01, nd=nd)
        else:
            self.up_block512_256 = ResUpUnit(256, 128, spatial_dropout_rate = 0, nd=nd)
            self.up_block256_128 = ResUpUnit(128, 64, spatial_dropout_rate = 0, nd=nd)
            self.up_block128_64 = ResUpUnit(64, 32, spatial_dropout_rate = 0, nd=nd)

        self.last = conv23DUnit(32, out_channels, 1, nd=nd)
        
        
    def forward(self, x):
        block0 = self.conv_block1_64(x)
        block1 = self.conv_block64_64(block0)
        pool1 = self.pool1(block1)
        if self.isSpatialDropOut:
            pool1 = self.dropout23d(pool1)

        block2 = self.conv_block64_128(pool1)
        pool2 = self.pool2(block2)
        if self.isSpatialDropOut:
            pool2 = self.dropout23d(pool2)

        block3 = self.conv_block128_256(pool2)
        pool3 = self.pool3(block3)
        if self.isSpatialDropOut:
            pool3 = self.dropout23d(pool3)

        block4 = self.conv_block256_512(pool3)


        up2 = self.up_block512_256(block4, block3)
        up3 = self.up_block256_128(up2, block2)
        up4 = self.up_block128_64(up3, block1)
#         return F.log_softmax(self.last(up4))
        return self.last(up4)



'''
Residual Learning for FCN: short-range residual connection and long-range residual connection
We use spatial dropout after each layer group
We use stochastic long-range connection
We use small dilation in the middle layers
In this version, I use deep (inter-mediate) supervision mechanism to help training
'''
class ResSegNet_DS(nn.Module):
    def __init__(self, in_channels, out_channels, isRandomConnection=None, isSmallDilation=None, isSpatialDropOut=None, dropoutRate=0.25, nd=2):
        super(ResSegNet_DS, self).__init__()
        #self.imsize = imsize
        
        assert nd==1 or nd==2 or nd==3, 'nd is not correctly specified!!!!, it should be {1,2,3}'
    
        self.isSpatialDropOut = isSpatialDropOut
        self.isRandomConnection = isRandomConnection
        self.isSmallDilation = isSmallDilation
        
        self.activation = F.relu
        
        self.pool1 = maxPool23DUinit(kernel_size=3, stride=2, padding=1, dilation=1, nd=nd)
        self.pool2 = maxPool23DUinit(kernel_size=3, stride=2, padding=1, dilation=1, nd=nd)
        self.pool3 = maxPool23DUinit(kernel_size=3, stride=2, padding=1, dilation=1, nd=nd)

        self.conv_block1_64 = conv23D_bn_relu_Unit(in_channels, 32, 3, padding=1, nd=nd)
        self.conv_block64_64 = residualUnit3(32, 32, isDilation=False,isEmptyBranch1=False, nd=nd)

        self.conv_block64_128 = residualUnit3(32, 64, isDilation=False,isEmptyBranch1=False, nd=nd)
        
        self.conv_block128_256 = residualUnit3(64, 128, isDilation=False,isEmptyBranch1=False, nd=nd)
        
        #the residual layers on the smallest resolution
        self.conv_block256_512 = residualUnit3(128, 256, isDilation=isSmallDilation, isEmptyBranch1=False, nd=nd)

        self.dropout23d = dropout23DUnit(prob=dropoutRate, nd=nd)

        
        if isRandomConnection:
            self.up_block512_256 = ResUpUnit(256, 128, spatial_dropout_rate = 0.1, nd=nd)
            self.up_block256_128 = ResUpUnit(128, 64, spatial_dropout_rate = 0.05, nd=nd)
            self.up_block128_64 = ResUpUnit(64, 32, spatial_dropout_rate = 0.01, nd=nd)
        else:
            self.up_block512_256 = ResUpUnit(256, 128, spatial_dropout_rate = 0, nd=nd)
            self.up_block256_128 = ResUpUnit(128, 64, spatial_dropout_rate = 0, nd=nd)
            self.up_block128_64 = ResUpUnit(64, 32, spatial_dropout_rate = 0, nd=nd)
        
        self.last_main = conv23DUnit(32, out_channels, 1, nd=nd)
        
        # intermediate_output1: upsampling 2 times: we firstly use plain conv and then use bilinearly upsmapling
        self.conv_path1 = conv23D_bn_relu_Unit(64, out_channels, 1, nd=nd)
        if nd==2:
            self.last_path1 = nn.Upsample(scale_factor=2, mode='bilinear')
        elif nd==3:
            self.last_path1 = nn.Upsample(scale_factor=2, mode='trilinear')
        # intermediate_output2: upsampling 4 times
        self.conv_path2 = conv23D_bn_relu_Unit(128, out_channels, 1, nd=nd)
        if nd==2:
            self.last_path2 = nn.Upsample(scale_factor=4, mode='bilinear')
        elif nd==3:
            self.last_path2 = nn.Upsample(scale_factor=4, mode='trilinear')

    def forward(self, x):   
        block0 = self.conv_block1_64(x)
        block1 = self.conv_block64_64(block0)
        pool1 = self.pool1(block1)
        if self.isSpatialDropOut:
            pool1 = self.dropout23d(pool1)

        block2 = self.conv_block64_128(pool1)
        pool2 = self.pool2(block2)
        if self.isSpatialDropOut:
            pool2 = self.dropout23d(pool2)

        block3 = self.conv_block128_256(pool2)
        pool3 = self.pool3(block3)
        if self.isSpatialDropOut:
            pool3 = self.dropout23d(pool3)

        block4 = self.conv_block256_512(pool3)

        up2 = self.up_block512_256(block4, block3)
        up3 = self.up_block256_128(up2, block2)
        up4 = self.up_block128_64(up3, block1)

        # for path1:2 times upsampling
        conv_path1 = self.conv_path1(up3)
        
        # for path2: 4 times upsampling
        conv_path2 = self.conv_path2(up2)
        
#         return F.log_softmax(self.last(up4))
        return self.last_main(up4),self.last_path1(conv_path1),self.last_path2(conv_path2)

'''
Residual Learning for FCN: short-range residual connection and long-range residual connection
We use spatial dropout after each layer group
We use stochastic long-range connection
We use small dilation in the middle layers
In this version, I use deep (inter-mediate) supervision mechanism to help training
We also include the dilated residual module in the highest resolution's (HR) lateral connection
'''
class HRResSegNet_DS(nn.Module):
    def __init__(self, in_channels, out_channels, isRandomConnection=None, isSmallDilation=None, isSpatialDropOut=None, dropoutRate=0.25, nd=2):
        super(HRResSegNet_DS, self).__init__()
        #self.imsize = imsize
        
        assert nd==1 or nd==2 or nd==3, 'nd is not correctly specified!!!!, it should be {1,2,3}'
    
        self.isSpatialDropOut = isSpatialDropOut
        self.isRandomConnection = isRandomConnection
        self.isSmallDilation = isSmallDilation
        
        self.activation = F.relu
        
        self.pool1 = maxPool23DUinit(kernel_size=3, stride=2, padding=1, dilation=1, nd=nd)
        self.pool2 = maxPool23DUinit(kernel_size=3, stride=2, padding=1, dilation=1, nd=nd)
        self.pool3 = maxPool23DUinit(kernel_size=3, stride=2, padding=1, dilation=1, nd=nd)
        #self.pool4 = nn.MaxPool2d(2)

        self.conv_block1_32 = conv23D_bn_relu_Unit(in_channels, 32, 3, padding=1, nd=nd)
        self.conv_block32_32 = residualUnit3(32, 32, isDilation=False,isEmptyBranch1=False, nd=nd)
        self.conv_block32_64 = residualUnit3(32, 64, isDilation=False,isEmptyBranch1=False, nd=nd)
        self.conv_block64_128 = residualUnit3(64, 128, isDilation=False,isEmptyBranch1=False, nd=nd)

        #the residual layers on the smallest resolution
        self.conv_block128_256 = residualUnit3(128, 256, isDilation=isSmallDilation, isEmptyBranch1=False, nd=nd)

        self.dropout23d = dropout23DUnit(prob=dropoutRate, nd=nd)
        
        #high-resoluton module for the lateral connection between the largest feature map
        self.hr_block1 = DilatedResUnit(32, 32, kernel_size=3, stride=1, dilation=2, nd=3)
        self.hr_block2 = DilatedResUnit(32, 32, kernel_size=3, stride=1, dilation=3, nd=3)
        self.hr_block3 = DilatedResUnit(32, 32, kernel_size=3, stride=1, dilation=4, nd=3)
        self.hr_block4 = DilatedResUnit(32, 32, kernel_size=3, stride=1, dilation=5, nd=3)
        
        if isRandomConnection:
            self.up_block256_128 = ResUpUnit(256, 128, spatial_dropout_rate = 0.1, nd=nd)
            self.up_block128_64 = ResUpUnit(128, 64, spatial_dropout_rate = 0.05, nd=nd)
            self.up_block64_32 = ResUpUnit(64, 32, spatial_dropout_rate = 0.01, nd=nd)
        else:
            self.up_block256_128 = ResUpUnit(256, 128, spatial_dropout_rate = 0, nd=nd)
            self.up_block128_64 = ResUpUnit(128, 64, spatial_dropout_rate = 0, nd=nd)
            self.up_block64_32 = ResUpUnit(64, 32, spatial_dropout_rate = 0, nd=nd)
        
        # main_output: the original main path
        self.last_main = conv23DUnit(32, out_channels, 1, nd=nd)
        
        # intermediate_output1: upsampling 2 times: we firstly use plain conv and then use bilinearly upsmapling
        self.conv_path1 = conv23D_bn_relu_Unit(64, out_channels, 1, nd=nd)
        if nd==2:
            self.last_path1 = nn.Upsample(scale_factor=2, mode='bilinear')
        elif nd==3:
            self.last_path1 = nn.Upsample(scale_factor=2, mode='trilinear')

        # intermediate_output2: upsampling 4 times
        self.conv_path2 = conv23D_bn_relu_Unit(128, out_channels, 1, nd=nd)
        if nd==2:
            self.last_path2 = nn.Upsample(scale_factor=4, mode='bilinear')
        elif nd==3:
            self.last_path2 = nn.Upsample(scale_factor=4, mode='trilinear')

    def forward(self, x):
        block0 = self.conv_block1_32(x)
        block1 = self.conv_block32_32(block0)
        pool1 = self.pool1(block1)
        if self.isSpatialDropOut:
            pool1 = self.dropout23d(pool1)

        block2 = self.conv_block32_64(pool1)
        pool2 = self.pool2(block2)
        if self.isSpatialDropOut:
            pool2 = self.dropout23d(pool2)

        block3 = self.conv_block64_128(pool2)
        pool3 = self.pool3(block3)
        if self.isSpatialDropOut:
            pool3 = self.dropout23d(pool3)

        block4 = self.conv_block128_256(pool3) #dilation in the smallest resolution

        ### here, we inject the high-resolution module with 3 or 4 dilated residual module
        hr_block1 = self.hr_block1(block1)
        hr_block2 = self.hr_block2(hr_block1)
        hr_block3 = self.hr_block3(hr_block2)
        hr_block4 = self.hr_block4(hr_block3)

        up2 = self.up_block256_128(block4, block3)
        up3 = self.up_block128_64(up2, block2)
        up4 = self.up_block64_32(up3, hr_block4)
        
        # for path1:2 times upsampling
        conv_path1 = self.conv_path1(up3)
        
        # for path2: 4 times upsampling
        conv_path2 = self.conv_path2(up2)
        
#         return F.log_softmax(self.last(up4))
        return self.last_main(up4),self.last_path1(conv_path1),self.last_path2(conv_path2)


'''
Residual Learning for FCN: short-range residual connection and long-range residual connection
We use spatial dropout after each layer group
We use stochastic long-range connection
We use small dilation in the middle layers
We also include the dilated residual module in the highest resolution's (HR) lateral connection
'''
class HRResSegNet(nn.Module):
    def __init__(self, in_channels, out_channels, isRandomConnection=None, isSmallDilation=None, isSpatialDropOut=None, dropoutRate=0.25, nd=2):
        super(HRResSegNet, self).__init__()
        #self.imsize = imsize
        
        assert nd==1 or nd==2 or nd==3, 'nd is not correctly specified!!!!, it should be {1,2,3}'
    
        self.isSpatialDropOut = isSpatialDropOut
        self.isRandomConnection = isRandomConnection
        self.isSmallDilation = isSmallDilation
        
        self.activation = F.relu
        
        self.pool1 = maxPool23DUinit(kernel_size=3, stride=2, padding=1, dilation=1, nd=nd)
        self.pool2 = maxPool23DUinit(kernel_size=3, stride=2, padding=1, dilation=1, nd=nd)
        self.pool3 = maxPool23DUinit(kernel_size=3, stride=2, padding=1, dilation=1, nd=nd)

        self.conv_block1_32 = conv23D_bn_relu_Unit(in_channels, 32, 3, padding=1, nd=nd)
        self.conv_block32_32 = residualUnit3(32, 32, isDilation=False,isEmptyBranch1=False, nd=nd)

        self.conv_block32_64 = residualUnit3(32, 64, isDilation=False,isEmptyBranch1=False, nd=nd)
        
        self.conv_block64_128 = residualUnit3(64, 128, isDilation=False,isEmptyBranch1=False, nd=nd)

        #the residual layers on the smallest resolution
        self.conv_block128_256 = residualUnit3(128, 256, isDilation=isSmallDilation, isEmptyBranch1=False, nd=nd)

        self.dropout23d = dropout23DUnit(prob=dropoutRate, nd=nd)
        
        #high-resoluton module for the lateral connection between the largest feature map
        self.hr_block1 = DilatedResUnit(32, 32, kernel_size=3, stride=1, dilation=2, nd=3)
        self.hr_block2 = DilatedResUnit(32, 32, kernel_size=3, stride=1, dilation=3, nd=3)
        self.hr_block3 = DilatedResUnit(32, 32, kernel_size=3, stride=1, dilation=4, nd=3)
        self.hr_block4 = DilatedResUnit(32, 32, kernel_size=3, stride=1, dilation=5, nd=3)
        
        if isRandomConnection:
            self.up_block256_128 = ResUpUnit(256, 128, spatial_dropout_rate = 0.1, nd=nd)
            self.up_block128_64 = ResUpUnit(128, 64, spatial_dropout_rate = 0.05, nd=nd)
            self.up_block64_32 = ResUpUnit(64, 32, spatial_dropout_rate = 0.01, nd=nd)
        else:
            self.up_block256_128 = ResUpUnit(256, 128, spatial_dropout_rate = 0, nd=nd)
            self.up_block128_64 = ResUpUnit(128, 64, spatial_dropout_rate = 0, nd=nd)
            self.up_block64_32 = ResUpUnit(64, 32, spatial_dropout_rate = 0, nd=nd)

        self.last_main = conv23DUnit(32, out_channels, 1, nd=nd)

    def forward(self, x):
        block0 = self.conv_block1_32(x)
        block1 = self.conv_block32_32(block0)
        pool1 = self.pool1(block1)
        if self.isSpatialDropOut:
            pool1 = self.dropout23d(pool1)

        block2 = self.conv_block32_64(pool1)
        pool2 = self.pool2(block2)
        if self.isSpatialDropOut:
            pool2 = self.dropout23d(pool2)

        block3 = self.conv_block64_128(pool2)
        pool3 = self.pool3(block3)
        if self.isSpatialDropOut:
            pool3 = self.dropout23d(pool3)

        block4 = self.conv_block128_256(pool3) #dilation in the smallest resolution

        ### here, we inject the high-resolution module with 3 or 4 dilated residual module
        hr_block1 = self.hr_block1(block1)
        hr_block2 = self.hr_block2(hr_block1)
        hr_block3 = self.hr_block3(hr_block2)
        hr_block4 = self.hr_block4(hr_block3)

        up2 = self.up_block256_128(block4, block3)
        up3 = self.up_block128_64(up2, block2)
        up4 = self.up_block64_32(up3, hr_block4)

        #         return F.log_softmax(self.last(up4))
        return self.last_main(up4)

    
'''
Residual Learning for FCN: short-range residual connection and long-range residual connection
Besides the segmentation maps, we also return the contours
We use spatial dropout after each layer group
We use stochastic long-range connection
We use small dilation in the middle layers
'''
class ResSegContourNet(nn.Module):
    def __init__(self, in_channels, out_channels, isRandomConnection=None, isSmallDilation=None, isSpatialDropOut=None, dropoutRate=0.25, nd=2):
        super(ResSegContourNet, self).__init__()
        
        assert nd==1 or nd==2 or nd==3, 'nd is not correctly specified!!!!, it should be {1,2,3}'
        
        self.isSpatialDropOut = isSpatialDropOut
        self.isRandomConnection = isRandomConnection
        self.isSmallDilation = isSmallDilation
        
        self.activation = F.relu
        
        self.pool1 = maxPool23DUinit(kernel_size=3,stride=2,padding=1, dilation=1, nd=nd)
        self.pool2 = maxPool23DUinit(kernel_size=3,stride=2,padding=1, dilation=1, nd=nd)
        self.pool3 = maxPool23DUinit(kernel_size=3,stride=2,padding=1, dilation=1, nd=nd)

        self.conv_block1_64 = conv23D_bn_relu_Unit(in_channels, 64, 3, nd=nd)
        self.conv_block64_64 = residualUnit3(64, 64, isDilation=False,isEmptyBranch1=False, nd=nd)

        self.conv_block64_128 = residualUnit3(64, 128, isDilation=False,isEmptyBranch1=False, nd=nd)
        
        self.conv_block128_256 = residualUnit3(128, 256, isDilation=False,isEmptyBranch1=False, nd=nd)
        
        #the residual layers on the smallest resolution
        self.conv_block256_512 = residualUnit3(256, 512, isDilation=isSmallDilation, isEmptyBranch1=False, nd=nd)

        self.dropout23d = dropout23DUnit(prob=dropoutRate, nd=nd)
        
        if isRandomConnection:
            self.up_block512_256 = ResUpUnit(512, 256, spatial_dropout_rate = 0.01, nd=nd)
            self.up_block256_128 = ResUpUnit(256, 128, spatial_dropout_rate = 0.01, nd=nd)
        else:
            self.up_block512_256 = ResUpUnit(512, 256, spatial_dropout_rate = 0, nd=nd)
            self.up_block256_128 = ResUpUnit(256, 128, spatial_dropout_rate = 0, nd=nd)

        self.up_block128_64 = BaseResUpUnit(128, 64, nd=nd)
        
        #### for segmentation
        self.seg_conv1 = residualUnit3(64, 64, nd=nd)
        self.seg_conv2 = residualUnit3(64, 32, nd=nd)
        self.seg_last = nn.Conv2d(32, out_channels, 1)
        
        ##### for contour classification
        self.contour_conv1 = residualUnit3(64, 64, nd=nd)
        self.contour_conv2 = residualUnit3(64, 32, nd=nd)
        self.contour_last = nn.Conv2d(32, 2, 1, nd=nd)
        
    def forward(self, x):
        block0 = self.conv_block1_64(x)
        block1 = self.conv_block64_64(block0)
        pool1 = self.pool1(block1)
        if self.isSpatialDropOut:
            pool1 = self.dropout2d(pool1)

        block2 = self.conv_block64_128(pool1)
        pool2 = self.pool2(block2)
        if self.isSpatialDropOut:
            pool2 = self.dropout2d(pool2)

        block3 = self.conv_block128_256(pool2)
        pool3 = self.pool3(block3)
        if self.isSpatialDropOut:
            pool3 = self.dropout2d(pool3)

        block4 = self.conv_block256_512(pool3)

        up2 = self.up_block512_256(block4, block3)
        up3 = self.up_block256_128(up2, block2)
        up4 = self.up_block128_64(up3, block1)

        # seg conv1
        seg_conv1 = self.seg_conv1(up4)
        seg_conv2 = self.seg_conv2(seg_conv1)
        
        # regression task1
        contour_conv1 = self.contour_conv1(up4)
        contour_conv2 = self.contour_conv2(contour_conv1)

        return self.seg_last(seg_conv2), self.contour_last(contour_conv2)
    
    
'''
Residual Learning for FCN: short-range residual connection and long-range residual connection
Multi-task Learning: Using segmentation and regression for the same network
'''
class ResSegRegNet(nn.Module):
    def __init__(self, in_channels, out_channels, nd=2):
        super(ResSegNet, self).__init__()
        #self.imsize = imsize
        assert nd==1 or nd==2 or nd==3, 'nd is not correctly specified!!!!, it should be {1,2,3}'
        self.activation = F.relu
        
        self.pool1 = maxPool23DUinit(kernel_size=3,stride=2,padding=1, dilation=1, nd=nd)
        self.pool2 = maxPool23DUinit(kernel_size=3,stride=2,padding=1, dilation=1, nd=nd)
        self.pool3 = maxPool23DUinit(kernel_size=3,stride=2,padding=1, dilation=1, nd=nd)

        self.conv_block1_64 = conv23D_bn_relu_Unit(in_channels, 64, 3, nd=nd)
        self.conv_block64_64 = residualUnit3(64, 64, nd=nd)

        self.conv_block64_128 = residualUnit3(64, 128, nd=nd)
        self.conv_block128_256 = residualUnit3(128, 256, nd=nd)
        self.conv_block256_512 = residualUnit3(256, 512, nd=nd)

        self.up_block512_256 = ResUpUnit(512, 256, nd=nd)
        self.up_block256_128 = ResUpUnit(256, 128, nd=nd)
        self.up_block128_64 = BaseResUpUnit(128, 64, nd=nd)
        
        #### for segmentation
        self.seg_conv1 = residualUnit3(64,64, nd=nd)
        self.seg_conv2 = residualUnit3(64,32, nd=nd)
        self.seg_last = nn.Conv2d(32, out_channels, 1)
        
        ##### for regression 1
        self.reg1_conv1 = residualUnit3(64, 64, nd=nd)
        self.reg1_conv2 = residualUnit3(64, 32, nd=nd)
        self.reg1_last = nn.Conv2d(32, 1, 1)
        
        ##### for regression 2
        self.reg2_conv1 = residualUnit3(64, 64, nd=nd)
        self.reg2_conv2 = residualUnit3(64, 32, nd=nd)
        self.reg2_last = nn.Conv2d(32, 1, 1)        
 
        ##### for regression 3
        self.reg3_conv1 = residualUnit3(64, 64, nd=nd)
        self.reg3_conv2 = residualUnit3(64, 32, nd=nd)
        self.reg3_last = nn.Conv2d(32, 1, 1)   

    def forward(self, x):
        block0 = self.conv_block1_64(x)
        block1 = self.conv_block64_64(block0)
        pool1 = self.pool1(block1)

        block2 = self.conv_block64_128(pool1)
        pool2 = self.pool2(block2)

        block3 = self.conv_block128_256(pool2)
        pool3 = self.pool3(block3)

        block4 = self.conv_block256_512(pool3)

        up2 = self.up_block512_256(block4, block3)

        up3 = self.up_block256_128(up2, block2)

        up4 = self.up_block128_64(up3, block1)
        
        # seg conv1
        seg_conv1 = self.seg_conv1(up4)
        seg_conv2 = self.seg_conv2(seg_conv1)
        
        # regression task1
        reg1_conv1 = self.reg1_conv1(up4)
        reg1_conv2 = self.reg1_conv2(reg1_conv1)
        
        # regression task2
        reg2_conv1 = self.reg2_conv1(up4)
        reg2_conv2 = self.reg2_conv2(reg2_conv1)
        
        # regression task3
        reg3_conv1 = self.reg3_conv1(up4)
        reg3_conv2 = self.reg3_conv2(reg3_conv1)

#         return F.log_softmax(self.last(up4))
        return self.seg_last(seg_conv2), self.reg1_last(reg1_conv2), self.reg2_last(reg2_conv2), self.reg3_last(reg3_conv2)
   
'''
useless, so far
'''    
class Generator(nn.Module):
    def __init__(self, n_residual_blocks, upsample):
        super(Generator, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample = upsample

        self.conv1 = nn.Conv2d(3, 64, 9, stride=1, padding=1)

        for i in range(self.n_residual_blocks):
            self.add_module('res' + str(i+1), residualUnit(64))

        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        in_channels = 64
        out_channels = 256
        for i in range(self.upsample):
            self.add_module('upscale' + str(i+1), upsampleUnit(in_channels, out_channels))
            in_channels = out_channels
            out_channels = out_channels/2

        self.conv3 = nn.Conv2d(in_channels, 3, 9, stride=1, padding=1)

    def forward(self, x):
        x = F.elu(self.conv1(x))

        y = self.__getattr__('res1')(x)
        for i in range(1, self.n_residual_blocks):
            y = self.__getattr__('res' + str(i+1))(y)

        x = self.conv2(y) + x

        for i in range(self.upsample):
            x = self.__getattr__('upscale' + str(i+1))(x)

        return F.sigmoid(self.conv3(x))

'''
    useless, so far
'''
class regGenerator(nn.Module):
    def __init__(self):
        super(regGenerator, self).__init__()
        
        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
        self.residual = self.make_layer(residualUnit3, 15)

        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
                
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        out = torch.add(out,residual)
        out = self.upscale4x(out)
        out = self.conv_output(out)
        return out

'''
    Discriminator: a easy structure of CNN (pooling is replaced with stride=2)
'''
class Discriminator(nn.Module):
    def __init__(self, in_channels, nd=2):
        super(Discriminator, self).__init__()
        self.conv1 = conv23DUnit(in_channels, 32, 3, stride=1, padding=1, nd=nd)
        
        self.conv2 = conv23DUnit(32, 32, 3, stride=2, padding=1, nd=nd)
        
        self.conv3 = conv23DUnit(32, 64, 3, stride=1, padding=1, nd=nd)
        
        self.conv4 = conv23DUnit(64, 64, 3, stride=2, padding=1, nd=nd)
        
        self.conv5 = conv23DUnit(64, 128, 3, stride=1, padding=1, nd=nd)
        
        self.conv6 = conv23DUnit(128, 128, 3, stride=2, padding=1, nd=nd)
        self.conv7 = conv23DUnit(128, 128, 3, stride=1, padding=1, nd=nd)
        
        self.conv8 = conv23DUnit(128, 128, 3, stride=2, padding=1, nd=nd)
        
        if nd==2:
            self.fc1 = nn.Linear(9856, 1024)
        elif nd==3:
            self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = F.elu(self.conv1(x))
#         print 'line 260 ',x.size()
        x = F.elu(self.conv2(x))
#         print 'line 262 ',x.size()
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
#         print 'line 265 ',x.size()
        x = F.elu(self.conv5(x))
        x = F.elu(self.conv6(x))
#         print 'line 267 ',x.size()
        x = F.elu(self.conv7(x))
        x = F.elu(self.conv8(x))
#         print 'line 271 ',x.size()

        # Flatten
        x = x.view(x.size(0), -1)

#         print 'line 266 is ', x.size()

        x = F.elu(self.fc1(x))
        return F.sigmoid(self.fc2(x))


'''
    Discriminator: fully convolutional networks
'''
class Discriminator_2DFCN(nn.Module):
    def __init__(self,in_channels,negative_slope = 0.2):
        super(Discriminator_2DFCN, self).__init__()
        self._in_channels = in_channels
        self._negative_slope = negative_slope

        self.conv1 = nn.Conv2d(in_channels=self._in_channels,out_channels=64,kernel_size=4,stride=2,padding=2)
        self.relu1 = nn.LeakyReLU(self._negative_slope,inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4,stride=2,padding=2)
        self.relu2 = nn.LeakyReLU(self._negative_slope,inplace=True)
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=4,stride=2,padding=2)
        self.relu3 = nn.LeakyReLU(self._negative_slope,inplace=True)
        self.conv4 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=4,stride=2,padding=2)
        self.relu4 = nn.LeakyReLU(self._negative_slope,inplace=True)
        self.conv5 = nn.Conv2d(in_channels=512,out_channels=2,kernel_size=4,stride=2,padding=2)

    def forward(self,x):
        x= self.conv1(x) # -,-,161,161
        x = self.relu1(x)
        x= self.conv2(x) # -,-,81,81
        x = self.relu2(x)
        x= self.conv3(x) # -,-,41,41
        x = self.relu3(x)
        x= self.conv4(x) # -,-,21,21
        x = self.relu4(x)
        x = self.conv5(x) # -,-,11,11
        # upsample
        x = F.upsample_bilinear(x,scale_factor=2)
        x = x[:,:,:-1,:-1] # -,-, 21,21

        x = F.upsample_bilinear(x,scale_factor=2)
        x = x[:,:,:-1,:-1] # -,-,41,41

        x = F.upsample_bilinear(x,scale_factor=2)
        x = x[:,:,:-1,:-1] #-,-,81,81

        x = F.upsample_bilinear(x,scale_factor=2)
        x = x[:,:,:-1,:-1] #-,-,161,161

        x = F.upsample_bilinear(x,scale_factor=2)
        x = x[:,:,:-1,:-1] # -,-,321,321

        return x
    
    
'''
    Discriminator: fully convolutional networks for 23D flexible network
    This one will downsample 5 times and then upsample 5 times, so it is sutiable for the real big images
'''
class Discriminator_23DFCN(nn.Module):
    def __init__(self,in_channels, out_channels, nd=2, negative_slope = 0.2):
        super(Discriminator_23DFCN, self).__init__()
        self._in_channels = in_channels
        self._negative_slope = negative_slope
        self._nd = nd
        
        self.conv1 = conv23DUnit(in_channels, 32, kernel_size=4, stride=2, padding=2, nd=nd)
        #self.conv1 = nn.Conv2d(in_channels=self._in_channels,out_channels=64,kernel_size=4,stride=2,padding=2)
        self.relu1 = nn.LeakyReLU(self._negative_slope,inplace=True)
#         self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4,stride=2,padding=2)
        self.conv2 = conv23DUnit(32, 128,kernel_size=4,stride=2,padding=2, nd=nd)
        self.relu2 = nn.LeakyReLU(self._negative_slope,inplace=True)
#         self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=4,stride=2,padding=2)
        self.conv3 = conv23DUnit(128,256,kernel_size=4,stride=2,padding=2, nd=nd)
        self.relu3 = nn.LeakyReLU(self._negative_slope,inplace=True)
#         self.conv4 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=4,stride=2,padding=2)
        self.conv4 = conv23DUnit(256,512,kernel_size=4,stride=2,padding=2,nd=nd)
        self.relu4 = nn.LeakyReLU(self._negative_slope,inplace=True)
        #self.conv5 = nn.Conv2d(in_channels=512,out_channels=2,kernel_size=4,stride=2,padding=2)
        self.conv5 = conv23DUnit(in_channels=512,out_channels=out_channels,kernel_size=4,stride=2,padding=2)

    def forward(self,x):
        x= self.conv1(x) # -,-,161,161
        x = self.relu1(x)
        x= self.conv2(x) # -,-,81,81
        x = self.relu2(x)
        x= self.conv3(x) # -,-,41,41
        x = self.relu3(x)
        x= self.conv4(x) # -,-,21,21
        x = self.relu4(x)
        x = self.conv5(x) # -,-,11,11
        # upsample
        
        if self._nd==2:
            mode = 'bilinear'
        elif self._nd==3:
            mode = 'trilinear'
        elif self._nd==1:
            mode = 'linear'
        else:
            mode = 'nearest'
        
        x = F.upsample(x,scale_factor=2, mode=mode)
        if self._nd==2:
            x = x[:,:,:-1,:-1] # -,-, 21,21
        if self._nd==3:
            x = x[:,:,:-1,:-1,:-1]

        x = F.upsample(x,scale_factor=2, mode=mode)
        if self._nd==2:
            x = x[:,:,:-1,:-1] # -,-, 41,41
        if self._nd==3:
            x = x[:,:,:-1,:-1,:-1]

        x = F.upsample(x,scale_factor=2, mode=mode)
        if self._nd==2:
            x = x[:,:,:-1,:-1] # -,-, 81,81
        if self._nd==3:
            x = x[:,:,:-1,:-1,:-1]

        x = F.upsample(x,scale_factor=2, mode=mode)
        if self._nd==2:
            x = x[:,:,:-1,:-1] # -,-, 161,161
        if self._nd==3:
            x = x[:,:,:-1,:-1,:-1]

        x = F.upsample(x,scale_factor=2, mode=mode)
        if self._nd==2:
            x = x[:,:,:-1,:-1] # -,-, 321,321
        if self._nd==3:
            x = x[:,:,:-1,:-1,:-1]

        return x
    
    
'''
    Discriminator: fully convolutional networks for 23D flexible network
    This one will downsample 5 times and then upsample 5 times, so it is sutiable for the real big images
'''
class Discriminator_my23DFCN(nn.Module):
    def __init__(self,in_channels, out_channels, nd=2, negative_slope = 0.2):
        super(Discriminator_23DFCN, self).__init__()
        self._in_channels = in_channels
        self._negative_slope = negative_slope
        self._nd = nd
        
        self.conv1 = conv23DUnit(in_channels, 32, kernel_size=4, stride=2, padding=2, nd=nd)
        #self.conv1 = nn.Conv2d(in_channels=self._in_channels,out_channels=64,kernel_size=4,stride=2,padding=2)
        self.relu1 = nn.LeakyReLU(self._negative_slope,inplace=True)
#         self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4,stride=2,padding=2)
        self.conv2 = conv23DUnit(32, 128,kernel_size=4,stride=2,padding=2, nd=nd)
        self.relu2 = nn.LeakyReLU(self._negative_slope,inplace=True)
#         self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=4,stride=2,padding=2)
        self.conv3 = conv23DUnit(128,out_channels,kernel_size=4,stride=2,padding=2, nd=nd)

    def forward(self,x):
        x= self.conv1(x) # -,-,161,161
        x = self.relu1(x)
        x= self.conv2(x) # -,-,81,81
        x = self.relu2(x)
        x= self.conv3(x) # -,-,41,41
        # upsample
        
        if self._nd==2:
            mode = 'bilinear'
        elif self._nd==3:
            mode = 'trilinear'
        elif self._nd==1:
            mode = 'linear'
        else:
            mode = 'nearest'
        
        x = F.upsample(x,scale_factor=2, mode=mode)
        if self._nd==2:
            x = x[:,:,:-1,:-1] # -,-, 21,21
        if self._nd==3:
            x = x[:,:,:-1,:-1,:-1]

        x = F.upsample(x,scale_factor=2, mode=mode)
        if self._nd==2:
            x = x[:,:,:-1,:-1] # -,-, 41,41
        if self._nd==3:
            x = x[:,:,:-1,:-1,:-1]

        x = F.upsample(x,scale_factor=2, mode=mode)
        if self._nd==2:
            x = x[:,:,:-1,:-1] # -,-, 81,81
        if self._nd==3:
            x = x[:,:,:-1,:-1,:-1]
        return x
    
'''
    Discriminator: fully convolutional networks (Unet-like, long-range residual connection FCN) for 23D flexible network
    This one will downsample 3 times and then upsample 3 times, so it is sutiable for the small images
'''
class Discriminator_my23DLRResFCN(nn.Module):
    def __init__(self,in_channels, out_channels, nd=2, negative_slope = 0.2):
        super(Discriminator_my23DLRResFCN, self).__init__()
        self._in_channels = in_channels
        self._negative_slope = negative_slope
        self._nd = nd
        self.conv1 = conv23DUnit(in_channels, 32, kernel_size=3, stride=1, padding=1, nd=nd)  #16
       
        self.conv2 = conv23DUnit(32, 64, kernel_size=4, stride=2, padding=2, nd=nd) #8

        self.conv3 = conv23DUnit(64, 128,kernel_size=4,stride=2,padding=2, nd=nd) #4
        
        self.conv4 = conv23DUnit(128,256,kernel_size=4,stride=2,padding=2, nd=nd) #2
  
        
        self.upconv3 = BaseResUpUnit(256, 128, nd=nd) #4
        self.upconv2 = BaseResUpUnit(128, 64, nd=nd) #8
        self.upconv1 = BaseResUpUnit(64, 32, nd=nd) #16
        
        self.convX = conv23DUnit(32, out_channels, kernel_size=1, stride=1, padding=0, nd=nd)  #16

        self.relu = nn.LeakyReLU(self._negative_slope,inplace=True)
      
    def forward(self,x):
        conv1 = self.conv1(x) # 16->16
        conv1 = self.relu(conv1)
        #down sample learning
        conv2 = self.conv2(conv1) # 16->8
        if self._nd==2:
            conv2 = conv2[:,:,:-1,:-1] # -,-, 21,21
        if self._nd==3:
            conv2 = conv2[:,:,:-1,:-1,:-1]
        conv2 = self.relu(conv2)
        conv3 = self.conv3(conv2) # 8->4
        if self._nd==2:
            conv3 = conv3[:,:,:-1,:-1] # -,-, 21,21
        if self._nd==3:
            conv3 = conv3[:,:,:-1,:-1,:-1]
        conv3 = self.relu(conv3)
        conv4 = self.conv4(conv3) # 4->2
        if self._nd==2:
            conv4 = conv4[:,:,:-1,:-1] # -,-, 21,21
        if self._nd==3:
            conv4 = conv4[:,:,:-1,:-1,:-1]
        conv4 = self.relu(conv4)
        
        # upsample reconstruction
        upconv3 = self.upconv3(conv4, conv3)
        upconv3 = self.relu(upconv3)
        upconv2 = self.upconv2(conv3, conv2)
        upconv2 = self.relu(upconv2)        
        upconv1 = self.upconv1(conv2, conv1)
        upconv1 = self.relu(upconv1)  
        
        
        convX = self.convX(upconv1)          
        return convX