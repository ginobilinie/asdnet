# -*- coding: utf-8 -*-
"""Implements SRGAN models: https://arxiv.org/abs/1609.04802

TODO:
    * Add Dropout to Generator
    * Try to make this work with SELU
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# custom weights initialization for SELU activations
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        size = m.weight.size()
        fan_in = size[1]
        nn.init.normal(m.weight.data, mean=0, std=np.sqrt(1.0/fan_in))
        nn.init.constant(m.bias.data, 0.0)

class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=8):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)

'''
 two-layer residual unit: two conv without BN and identity mapping
'''
class residualBlock(nn.Module):
    def __init__(self, in_channels):
        super(residualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv2(F.elu(self.conv1(x))) + x
'''
    two-layer residual unit: two conv with BN and identity mapping
'''    
class residualBlock1(nn.Module): 
    def __init__(self):
        super(residualBlock1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        
    def forward(self, x): 
        identity_data = x
        output = self.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        output = torch.add(output,identity_data)
        return output 

'''
three-layer residual unit: three conv with BN and identity mapping
'''
class residualBlock3(nn.Module): 
    def __init__(self):
        super(residualBlock1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        
    def forward(self, x): 
        identity_data = x
        output = self.relu(self.bn1(self.conv1(x)))
        output = self.relu(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output))
        output = torch.add(output,identity_data)
        return output 
    
class upsampleBlock(nn.Module):
    # Implements resize-convolution
    def __init__(self, in_channels, out_channels):
        super(upsampleBlock, self).__init__()
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)

    def forward(self, x):
        return F.elu(self.conv1(self.upsample1(x)))

class unetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(unetConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size)
        self.activation = activation

    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.activation(self.conv2(out))

        return out


class unetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False):
        super(unetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, 2, stride=2)
        self.conv = nn.Conv2d(in_size, out_size, kernel_size)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size)
        self.activation = activation

    def center_crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x, bridge):#bridge is the corresponding lower layer
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.size()[2])
        out = torch.cat([up, crop1], 1)
        out = self.activation(self.conv(out))
        out = self.activation(self.conv2(out))

        return out

class Generator(nn.Module):
    def __init__(self, n_residual_blocks, upsample):
        super(Generator, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample = upsample

        self.conv1 = nn.Conv2d(3, 64, 9, stride=1, padding=1)

        for i in range(self.n_residual_blocks):
            self.add_module('res' + str(i+1), residualBlock(64))

        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        in_channels = 64
        out_channels = 256
        for i in range(self.upsample):
            self.add_module('upscale' + str(i+1), upsampleBlock(in_channels, out_channels))
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

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=2, padding=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = F.elu(self.conv1(x))

        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        x = F.elu(self.conv6(x))
        x = F.elu(self.conv7(x))
        x = F.elu(self.conv8(x))

        # Flatten
        x = x.view(x.size(0), -1)

        x = F.elu(self.fc1(x))
        return F.sigmoid(self.fc2(x))