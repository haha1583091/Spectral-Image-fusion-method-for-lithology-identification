#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score, precision_recall_fscore_support, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import spectral
import optuna
import time
import os
import copy
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
from optuna.integration import SkoptSampler
import datetime
import skopt
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 使用GPU训练


# ---
# # **1. 预训练**

# ## (1) 数据及网络结构
# 

# ## 1) 光谱分支——(光谱维度注意力机制 + ResNet网络结构)

# In[2]:


# ----------------------------------------光谱维度——注意力机制-------------------------------------------------
class SpectralAttention(nn.Module):
    def __init__(self, channels, spectral_dim):
        super(SpectralAttention, self).__init__()
        self.spectral_dim = spectral_dim
        self.global_avg_pool = nn.AdaptiveAvgPool3d((spectral_dim, 1, 1))
        self.fc = nn.Linear(spectral_dim, spectral_dim)
    
    def forward(self, x):
        batch_size, channels, spectral_dim, _, _ = x.shape
        out = self.global_avg_pool(x)
        out = out.view(batch_size, channels, spectral_dim)
        out = out.mean(dim=1)
        attention_weights = self.fc(out)
        attention_weights = F.softmax(attention_weights, dim=-1).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        attention_weights = attention_weights.expand_as(x)
        out = x * attention_weights
        return out
        
# ----------------------------------------光谱维度——Resnet_spectral网络-------------------------------------------------
def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def get_inplanes():
    return [64, 128, 256, 512]

class BasicBlock_spectral(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock_spectral, self).__init__()
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Bottleneck_spectral(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck_spectral, self).__init__()
        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet_spectral(nn.Module):
    def __init__(self, block, layers, block_inplanes, n_input_channels=1, conv1_t_size=1, conv1_t_stride=1, no_max_pool=False, shortcut_type='B', widen_factor=1.0, n_classes=50):
        super().__init__()
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.conv1 = nn.Conv3d(n_input_channels, self.in_planes, kernel_size=(conv1_t_size, 7, 7), stride=(conv1_t_stride, 2, 2), padding=(conv1_t_size // 2, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], shortcut_type, stride=2)
        self.spectral_attention_layer1 = SpectralAttention(channels=256 * block.expansion, spectral_dim=5)
        self.spectral_attention_layer2 = SpectralAttention(channels=512 * block.expansion, spectral_dim=3)
        self.spectral_attention_layer3 = SpectralAttention(channels=1024 * block.expansion, spectral_dim=2)
        self.spectral_attention_layer4 = SpectralAttention(channels=2048 * block.expansion, spectral_dim=1) 
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()
        out = torch.cat([out.data, zero_pads], dim=1)
        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block, planes=planes * block.expansion, stride=stride)
            else:
                downsample = nn.Sequential(conv1x1x1(self.in_planes, planes * block.expansion, stride), nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(block(in_planes=self.in_planes, planes=planes, stride=stride, downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, ))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.spectral_attention_layer1(x)
        
        x = self.layer2(x)
        x = self.spectral_attention_layer2(x)
        
        x = self.layer3(x)
        x = self.spectral_attention_layer3(x)
        
        x = self.layer4(x)
        x = self.spectral_attention_layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

def generate_model_spectral(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]
    if model_depth == 10:
        model = ResNet_spectral(BasicBlock_spectral, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet_spectral(BasicBlock_spectral, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet_spectral(BasicBlock_spectral, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet_spectral(Bottleneck_spectral, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet_spectral(Bottleneck_spectral, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet_spectral(Bottleneck_spectral, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet_spectral(Bottleneck_spectral, [3, 24, 36, 3], get_inplanes(), **kwargs)
    return model


# ### 2) 纹理分支——(空间维度注意力机制 + ResNet网络结构)

# In[3]:


# ----------------------------------------空间维度——STN注意力机制-------------------------------------------------
class STN(nn.Module):
    def __init__(self, in_channels):
        super(STN, self).__init__()
        # 局部网络（定位网络）
        self.localization = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=7, padding=3),
            nn.MaxPool3d(2, stride=2, padding=(0,1,1)),
            nn.ReLU(True),
            nn.Conv3d(8, 16, kernel_size=5, padding=2),
            nn.MaxPool3d(2, stride=2, padding=(1,0,0)),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(16 * 1 * 2 * 2, 64),
            nn.ReLU(True),
            nn.Linear(64, 3 * 4)  # 需要输出12个参数（3x4变换矩阵）对应3D仿射变换
        )

        # 初始化权重/偏置为3D仿射变换的初始值
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 16 * 1 * 2 * 2)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 3, 4)

        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x

# ----------------------------------------纹理维度——Resnet网络-------------------------------------------------
class BasicBlock_texture(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock_texture, self).__init__()
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Bottleneck_texture(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        
        return out


class ResNet_texture(nn.Module):
    def __init__(self, block, layers, block_inplanes, n_input_channels=1, conv1_t_size=1, conv1_t_stride=1, no_max_pool=False, shortcut_type='B', widen_factor=1.0, n_classes=50):
        super().__init__()
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.conv1 = nn.Conv3d(n_input_channels, self.in_planes, kernel_size=(conv1_t_size, 7, 7), stride=(conv1_t_stride, 2, 2), padding=(conv1_t_size // 2, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.stn = STN(self.in_planes) # 初始化STN模块
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], shortcut_type, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()
        out = torch.cat([out.data, zero_pads], dim=1)
        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block, planes=planes * block.expansion, stride=stride)
            else:
                downsample = nn.Sequential(conv1x1x1(self.in_planes, planes * block.expansion, stride), nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(block(in_planes=self.in_planes, planes=planes, stride=stride, downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
        x = self.stn(x) # 首先通过STN模块处理输入
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

def generate_model_texture(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]
    if model_depth == 10:
        model = ResNet_texture(BasicBlock_texture, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet_texture(BasicBlock_texture, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet_texture(BasicBlock_texture, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet_texture(Bottleneck_texture, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet_texture(Bottleneck_texture, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet_texture(Bottleneck_texture, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet_texture(Bottleneck_texture, [3, 24, 36, 3], get_inplanes(), **kwargs)
    return model


# ### 3) Resnet网络融合

# In[4]:


class HybridSNLateFusion(nn.Module):
    def __init__(self, class_num, dropout_prob=0.2):
        super(HybridSNLateFusion, self).__init__()

        # 正确初始化光谱分支和纹理分支，指定输入通道数
        self.spectral_branch = generate_model_spectral(50, n_input_channels=1, n_classes=class_num)
        self.texture_branch = generate_model_texture(50, n_input_channels=1, n_classes=class_num)
        
        # 适应两个特征向量合并后的维度
        self.fc1 = nn.Linear(2048*2, class_num)  
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x_spectral, x_texture):
        # 通过各自的分支处理输入
        x_spectral = self.spectral_branch(x_spectral)
        x_texture = self.texture_branch(x_texture)
        
        # 合并两个分支的特征
        x_combined = torch.cat((x_spectral, x_texture), dim=1)
        x = self.fc1(x_combined)
        return x


# ### 4) 数据

# In[5]:


patch_size = 25  # 每个像素周围提取 patch 的尺寸
# class_num = 45
class_num = 50
# 2. ----------------------------------------定义相关函数-------------------------------------------------
''' 对高光谱数据 X 应用 PCA 变换 '''
def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX

''' 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作 '''
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

''' 在每个像素周围提取 patch ，然后创建成符合 keras 处理的格式 '''
def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):
    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin].item()
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

''' 划分训练集和测试机 '''
def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState, stratify=y)
    return X_train, X_test, y_train, y_test
        
# 3. ----------------------------------------导入数据-------------------------------------------------
test_ratio = 0.4 # 用于测试样本的比例
pca_components = 10 # 主成分的数量
spectral_channels = 10 #光谱数据通道
texture_channels = 4 # 纹理数据通道

# 路径设置
# X_spectral_path = '/root/autodl-tmp/Jupyter_project/00_Data/迁移学习光谱+纹理+标签_mat/45-back.mat'
# X_texture_path = '/root/autodl-tmp/Jupyter_project/00_Data/迁移学习光谱+纹理+标签_mat/45-back_texture.mat'
# y_path = '/root/autodl-tmp/Jupyter_project/00_Data/迁移学习光谱+纹理+标签_mat/45-back_label.mat'

X_spectral_path = '/root/autodl-tmp/Jupyter_project/00_Data/预训练200/1-50-B.mat'  
X_texture_path = '/root/autodl-tmp/Jupyter_project/00_Data/预训练200/1-50-B_texture.mat'
y_path = '/root/autodl-tmp/Jupyter_project/00_Data/预训练200/1-50-B_label.mat'

# 加载数据
X_spectral = sio.loadmat(X_spectral_path)['data']
X_texture = sio.loadmat(X_texture_path)['texture']
y = sio.loadmat(y_path)['data']
zero_count = y.size - np.count_nonzero(y) # 计算标签中零元素的个数

print("\033[1;91m\n... ... ... ... ... ... 原始数据维度 ... ... ... ... ... ...\033[0m")
print('Spectral data shape: ', X_spectral.shape)
print('Texture data shape: ', X_texture.shape)
print('Label shape: ', y.shape)
print("零元素个数为：", zero_count)

print("\033[1;91m\n\n... ... ... ... ... ... 光谱数据处理 ... ... ... ... ... ...\033[0m")
print('... ... PCA tranformation ... ...')
X_spectral_pca = applyPCA(X_spectral, numComponents=pca_components)
print('Spectral data shape after PCA: ', X_spectral_pca.shape)

print('\n... ... create data cubes ... ...')
X_spectral_pca, y_spectral = createImageCubes(X_spectral_pca, y, windowSize=patch_size)
print('Spectral data cube X_spectral shape: ', X_spectral_pca.shape)
print('Spectral data cube y_spectral shape: ', y_spectral.shape)

print('\n... ... 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求 ... ...')
X_spectral = X_spectral_pca.reshape(-1, patch_size, patch_size, spectral_channels, 1)
print('before transpose: X_spectral  shape: ', X_spectral.shape) 

print('\n... ... 为了适应 pytorch 结构，数据要做 transpose ... ...')
X_spectral = X_spectral.transpose(0, 4, 3, 1, 2)
print('after transpose: X_spectral  shape: ', X_spectral.shape) 

y = sio.loadmat(y_path)['data']
print("\033[1;91m\n\n... ... ... ... ... ... 纹理数据处理 ... ... ... ... ... ...\033[0m")

print('... ... create data cubes ... ...')
X_texture, y_texture = createImageCubes(X_texture, y, windowSize=patch_size)
print('Texture data cube X_texture shape: {}\n'
      'Texture data cube y_texture shape: {}\n'
      .format(X_texture.shape, y_texture.shape))

print('... ... 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求 ... ...')
X_texture = X_texture.reshape(-1, patch_size, patch_size, texture_channels, 1)
print('before transpose: X_texture  shape: ', X_texture.shape) 

print('\n... ... 为了适应 pytorch 结构，数据要做 transpose ... ...')
X_texture = X_texture.transpose(0, 4, 3, 1, 2)
print('after transpose: X_texture  shape: ', X_texture.shape)

print("\033[1;91m\n\n... ... ... ... ... ... 合并数据，划分训练测试集 ... ... ... ... ... ...\033[0m")
# 确保标签顺序相同
if y_spectral[0] == y_texture[0]:
    print("Label and data array lengths match successfully.\n")
else:
    print("Labels do not match between spectral and texture data.\n")
y = y_texture

# 合并光谱和纹理数据
X_combined = np.concatenate((X_spectral, X_texture), axis=2)
print('X_combined shape: ', X_combined.shape, '\n')

X_train, X_test, y_train, y_test = splitTrainTestSet(X_combined, y, test_ratio)
print('X_train shape: {}\n'
      'y_train shape: {}\n'
      'X_test shape: {}\n'
      'y_test shape: {}'
      .format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

""" Training dataset """
class TrainDS(torch.utils.data.Dataset):
    def __init__(self):
        self.len = X_train.shape[0]
        self.x_data = torch.FloatTensor(X_train)
        self.y_data = torch.LongTensor(y_train)
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self): 
        # 返回文件数据的数目
        return self.len

""" Testing dataset"""
class TestDS(torch.utils.data.Dataset):
    def __init__(self):
        self.len = X_test.shape[0]
        self.x_data = torch.FloatTensor(X_test)
        self.y_data = torch.LongTensor(y_test)
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self): 
        # 返回文件数据的数目
        return self.len

# 创建 trainloader 和 testloader
trainset = TrainDS()
testset = TestDS()
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=256, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=testset,  batch_size=256, shuffle=False, num_workers=0)


# ## (2) 预训练

# ### 1) 训练测试一起

# In[6]:


# 网络放到GPU上
net = HybridSNLateFusion(class_num, dropout_prob=0.2).to(device)
# Epoch_pretrain = 40
Epoch_pretrain = 40
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

# 初始化参数列表
best_acc = 0
best_model_wts = copy.deepcopy(net.state_dict())  # 保存最佳模型权重

batch_train_losses = []
batch_train_accuracies = []

epoch_train_losses = []
epoch_train_accuracies = []
epoch_test_losses = []
epoch_test_accuracies = []

# 开始训练循环
for epoch in range(Epoch_pretrain):
    net.train()  # 设置模型为训练模式
    total_loss_train = 0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs_spectral, inputs_texture = inputs[:, :, :10, :, :], inputs[:, :, 10:, :, :]  # 分割inputs
        inputs_spectral, inputs_texture, labels = inputs_spectral.to(device), inputs_texture.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs_spectral, inputs_texture)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss_train += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        # 记录每个batch的loss和准确率
        batch_train_losses.append(loss.item())
        batch_accuracy = 100 * correct_train / total_train
        batch_train_accuracies.append(batch_accuracy)

    # 每个epoch的平均训练loss和准确率
    epoch_train_loss = total_loss_train / len(train_loader)
    epoch_train_accuracy = 100 * correct_train / total_train
    epoch_train_losses.append(epoch_train_loss)
    epoch_train_accuracies.append(epoch_train_accuracy)

    # 测试部分
    net.eval()  # 设置模型为评估模式
    total_loss_test = 0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs_spectral, inputs_texture = inputs[:, :, :10, :, :], inputs[:, :, 10:, :, :]  # 分割inputs
            inputs_spectral, inputs_texture, labels = inputs_spectral.to(device), inputs_texture.to(device), labels.to(device)

            outputs = net(inputs_spectral, inputs_texture)
            loss = criterion(outputs, labels)

            total_loss_test += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    # 每个epoch的平均测试loss和准确率
    epoch_test_loss = total_loss_test / len(test_loader)
    epoch_test_accuracy = 100 * correct_test / total_test
    epoch_test_losses.append(epoch_test_loss)
    epoch_test_accuracies.append(epoch_test_accuracy)

    # 更新最佳模型（如果适用）
    if epoch_test_accuracy > best_acc:
        best_acc = epoch_test_accuracy
        best_model_wts = copy.deepcopy(net.state_dict())

    print(f"[Epoch {epoch+1}] Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.2f}%, Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_accuracy:.2f}%")

print('Finished Training')
print(f'Best Testing Accuracy: {best_acc:.2f}%')

# 加载最佳模型权重
net.load_state_dict(best_model_wts)

# 保存预训练最佳模型
output_filename = 'best_pretrain_model_' + os.path.basename(X_spectral_path).split('.')[0] + '.pth'
torch.save(net.state_dict(), output_filename)
print(f"预训练最佳模型已保存为: {output_filename}")


# In[7]:


# 创建一个图和六个子图
fig, axs = plt.subplots(3, 2, figsize=(15, 15))

# 分别绘制每个变量的趋势
axs[0, 0].plot(batch_train_losses, label='Batch Train Loss')
axs[0, 0].set_title('Batch Train Loss')
axs[0, 0].set_xlabel('Batch')
axs[0, 0].set_ylabel('Loss')
axs[0, 0].legend()

axs[0, 1].plot(batch_train_accuracies, label='Batch Train Accuracy', color='orange')
axs[0, 1].set_title('Batch Train Accuracy')
axs[0, 1].set_xlabel('Batch')
axs[0, 1].set_ylabel('Accuracy (%)')
axs[0, 1].legend()

axs[1, 0].plot(epoch_train_losses, label='Epoch Train Loss', color='green')
axs[1, 0].set_title('Epoch Train Loss')
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].set_ylabel('Loss')
axs[1, 0].legend()

axs[1, 1].plot(epoch_train_accuracies, label='Epoch Train Accuracy', color='red')
axs[1, 1].set_title('Epoch Train Accuracy')
axs[1, 1].set_xlabel('Epoch')
axs[1, 1].set_ylabel('Accuracy (%)')
axs[1, 1].legend()

axs[2, 0].plot(epoch_test_losses, label='Epoch Test Loss', color='purple')
axs[2, 0].set_title('Epoch Test Loss')
axs[2, 0].set_xlabel('Epoch')
axs[2, 0].set_ylabel('Loss')
axs[2, 0].legend()

axs[2, 1].plot(epoch_test_accuracies, label='Epoch Test Accuracy', color='brown')
axs[2, 1].set_title('Epoch Test Accuracy')
axs[2, 1].set_xlabel('Epoch')
axs[2, 1].set_ylabel('Accuracy (%)')
axs[2, 1].legend()

# 调整子图之间的间距
plt.tight_layout()

# 显示图表
# plt.show()


# In[8]:


df = pd.DataFrame({
    "Batch Train Losses": pd.Series(batch_train_losses),
    "Batch Train Accuracies": pd.Series(batch_train_accuracies),
    "Epoch Train Losses": pd.Series(epoch_train_losses),
    "Epoch Train Accuracies": pd.Series(epoch_train_accuracies),
    "Epoch Test Losses": pd.Series(epoch_test_losses),
    "Epoch Test Accuracies": pd.Series(epoch_test_accuracies)
})

# 指定保存路径，注意更改为你想要保存的路径
save_path = "training_ResNetX2-50.csv"

# 保存为CSV
df.to_csv(save_path, index=False)


# ### 2) 训练测试分开

# In[9]:


# 网络放到GPU上
net = HybridSNLateFusion(class_num, dropout_prob=0.2).to(device)
# Epoch_pretrain = 3
Epoch_pretrain = 3
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0025)

# 用于记录准确率的数组
epoch_accuracies_pretrain = []  # 记录每个epoch的准确率
batch_accuracies_pretrain = []  # 记录每个batch的准确率

best_acc = 0.0  # 记录最佳准确率
best_model_wts = copy.deepcopy(net.state_dict())  # 保存最佳模型权重

# 开始训练
for epoch in range(Epoch_pretrain):
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        # 保证将inputs分割为光谱数据和纹理数据时，通道维度正确
        inputs_spectral = inputs[:, :, :10, :, :]  # 取前10个通道为光谱数据
        inputs_texture = inputs[:, :, 10:, :, :]  # 取后4个通道为纹理数据

        inputs_spectral, inputs_texture, labels = inputs_spectral.to(device), inputs_texture.to(device), labels.to(device)

        # 优化器梯度归零
        optimizer.zero_grad()

        # 正向传播 + 反向传播 + 优化
        outputs = net(inputs_spectral, inputs_texture)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()  # 这里累加当前批次的损失

        # 计算准确率和损失
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # 使用累积的总损失来计算平均损失
    epoch_loss = total_loss / len(train_loader)

    # 计算并记录每个epoch的平均准确率
    epoch_accuracy = 100 * correct / total
    epoch_accuracies_pretrain.append(epoch_accuracy)

    # 更新最佳模型权重
    if epoch_accuracy > best_acc:
        best_acc = epoch_accuracy
        best_model_wts = copy.deepcopy(net.state_dict())

    print('[Epoch: %d]   [Loss Avg: %.4f]   [Current Loss: %.4f]   [Accuracy: %.2f%%]' %
          (epoch + 1, epoch_loss, loss.item(), epoch_accuracy))

print('Finished Training')
print('Best Training Accuracy: {:.2f}%'.format(best_acc))

# 加载最佳模型权重
net.load_state_dict(best_model_wts)

# 保存预训练最佳模型
output_filename = 'best_pretrain_model_' + os.path.basename(X_spectral_path).split('.')[0] + '.pth' # 提取原始 X 数据文件名用于输出文件命名
torch.save(net.state_dict(), output_filename)
print(f"预训练最佳模型已保存为: {output_filename}")


# ## (3) 预训练测试

# In[10]:


# 调用模型
net = HybridSNLateFusion(class_num)
net.load_state_dict(torch.load('best_pretrain_model_1-50-B.pth'))
# pretrained_dict = torch.load('best_pretrain_model_45-back.pth') 
# new_state_dict = {}
# for k, v in pretrained_dict.items():
#     if k in net.state_dict() and net.state_dict()[k].size() == v.size():
#         new_state_dict[k] = v
# net.load_state_dict(new_state_dict, strict=False)
net = net.to(device)

# 将模型设置为评估模式
net.eval()

y_pred_test = []
y_true_test = []

# 不计算梯度，以节省内存和计算资源
with torch.no_grad():
    for inputs, labels in test_loader:
        # 分割合并后的数据为光谱数据和纹理数据
        inputs_spectral = inputs[:, :, :10, :, :]
        inputs_texture = inputs[:, :, 10:, :, :]

        # 将数据移动到指定设备
        inputs_spectral, inputs_texture, labels = inputs_spectral.to(device), inputs_texture.to(device), labels.to(device)

        # 正向传播，获取输出
        outputs = net(inputs_spectral, inputs_texture)

        # 获取预测结果
        _, predicted = torch.max(outputs.data, 1)

        # 保存预测和实际标签
        y_pred_test.extend(predicted.cpu().numpy())
        y_true_test.extend(labels.cpu().numpy())

# 将预测和实际标签转换为 NumPy 数组
y_pred_test = np.array(y_pred_test)
y_true_test = np.array(y_true_test)

# 生成分类报告
classification = classification_report(y_true_test, y_pred_test, digits=6)
print(classification)


# ---
# # **2. 迁移学习**

# ## (1) 导入数据

# In[ ]:


test_ratio = 0.4 # 用于测试样本的比例

# 路径设置
X_spectral_path = '/root/autodl-tmp/Jupyter_project/00_Data/迁移学习光谱+纹理+标签_mat/45-back.mat'
X_texture_path = '/root/autodl-tmp/Jupyter_project/00_Data/迁移学习光谱+纹理+标签_mat/45-back_texture.mat'
y_path = '/root/autodl-tmp/Jupyter_project/00_Data/迁移学习光谱+纹理+标签_mat/45-back_label.mat'
# 加载数据
X_spectral = sio.loadmat(X_spectral_path)['data']
X_texture = sio.loadmat(X_texture_path)['texture']
y = sio.loadmat(y_path)['data']
zero_count = y.size - np.count_nonzero(y) # 计算标签中零元素的个数

print("\033[1;91m\n... ... ... ... ... ... 原始数据维度 ... ... ... ... ... ...\033[0m")
print('Spectral data shape: ', X_spectral.shape)
print('Texture data shape: ', X_texture.shape)
print('Label shape: ', y.shape)
print("零元素个数为：", zero_count)

print("\033[1;91m\n\n... ... ... ... ... ... 光谱数据处理 ... ... ... ... ... ...\033[0m")
print('... ... PCA tranformation ... ...')
X_spectral_pca = applyPCA(X_spectral, numComponents=pca_components)
print('Spectral data shape after PCA: ', X_spectral_pca.shape)

print('\n... ... create data cubes ... ...')
X_spectral_pca, y_spectral = createImageCubes(X_spectral_pca, y, windowSize=patch_size)
print('Spectral data cube X_spectral shape: ', X_spectral_pca.shape)
print('Spectral data cube y_spectral shape: ', y_spectral.shape)

print('\n... ... 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求 ... ...')
X_spectral = X_spectral_pca.reshape(-1, patch_size, patch_size, spectral_channels, 1)
print('before transpose: X_spectral  shape: ', X_spectral.shape) 

print('\n... ... 为了适应 pytorch 结构，数据要做 transpose ... ...')
X_spectral = X_spectral.transpose(0, 4, 3, 1, 2)
print('after transpose: X_spectral  shape: ', X_spectral.shape) 

y = sio.loadmat(y_path)['data']
print("\033[1;91m\n\n... ... ... ... ... ... 纹理数据处理 ... ... ... ... ... ...\033[0m")

print('... ... create data cubes ... ...')
X_texture, y_texture = createImageCubes(X_texture, y, windowSize=patch_size)
print('Texture data cube X_texture shape: {}\n'
      'Texture data cube y_texture shape: {}\n'
      .format(X_texture.shape, y_texture.shape))

print('... ... 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求 ... ...')
X_texture = X_texture.reshape(-1, patch_size, patch_size, texture_channels, 1)
print('before transpose: X_texture  shape: ', X_texture.shape) 

print('\n... ... 为了适应 pytorch 结构，数据要做 transpose ... ...')
X_texture = X_texture.transpose(0, 4, 3, 1, 2)
print('after transpose: X_texture  shape: ', X_texture.shape)

print("\033[1;91m\n\n... ... ... ... ... ... 合并数据，划分训练测试集 ... ... ... ... ... ...\033[0m")
# 确保标签顺序相同
if y_spectral[0] == y_texture[0]:
    print("Label and data array lengths match successfully.\n")
else:
    print("Labels do not match between spectral and texture data.\n")
y = y_texture

# 合并光谱和纹理数据
X_combined = np.concatenate((X_spectral, X_texture), axis=2)
print('X_combined shape: ', X_combined.shape, '\n')

X_train, X_test, y_train, y_test = splitTrainTestSet(X_combined, y, test_ratio)
print('X_train shape: {}\n'
      'y_train shape: {}\n'
      'X_test shape: {}\n'
      'y_test shape: {}'
      .format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

""" Training dataset """
class TrainDS(torch.utils.data.Dataset):
    def __init__(self):
        self.len = X_train.shape[0]
        self.x_data = torch.FloatTensor(X_train)
        self.y_data = torch.LongTensor(y_train)        
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self): 
        # 返回文件数据的数目
        return self.len

""" Testing dataset"""
class TestDS(torch.utils.data.Dataset):
    def __init__(self):
        self.len = X_test.shape[0]
        self.x_data = torch.FloatTensor(X_test)
        self.y_data = torch.LongTensor(y_test)
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self): 
        # 返回文件数据的数目
        return self.len

# 创建 trainloader 和 testloader
trainset = TrainDS()
testset = TestDS()
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=64, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=testset,  batch_size=64, shuffle=False, num_workers=0)


# ## (2) 迁移训练

# In[ ]:


# 如果需要，修改适应新任务
class_num = 45  # 新任务的类别数量

# 调用模型
net = HybridSNLateFusion(class_num)
# net.load_state_dict(torch.load('pretrain_45-back.pth'))
pretrained_dict = torch.load('best_pretrain_model_1-50-B.pth') 
new_state_dict = {}
for k, v in pretrained_dict.items():
    if k in net.state_dict() and net.state_dict()[k].size() == v.size():
        new_state_dict[k] = v
net.load_state_dict(new_state_dict, strict=False)

# 冻结前面的层
for param in net.parameters():
    param.requires_grad = False

# 如果需要，修改模型参数适应新任务
net.fc1 = nn.Linear(2048*2, 45)  # 替换第一个全连接层
net = net.to(device)

# Epoch = 30
Epoch = 30
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0025)

# 初始化参数列表
best_acc = 0
best_model_wts = copy.deepcopy(net.state_dict())  # 保存最佳模型权重

batch_train_losses = []
batch_train_accuracies = []

epoch_train_losses = []
epoch_train_accuracies = []
epoch_test_losses = []
epoch_test_accuracies = []

# 开始训练循环
for epoch in range(Epoch):
    net.train()  # 设置模型为训练模式
    total_loss_train = 0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs_spectral, inputs_texture = inputs[:, :, :10, :, :], inputs[:, :, 10:, :, :]  # 分割inputs
        inputs_spectral, inputs_texture, labels = inputs_spectral.to(device), inputs_texture.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs_spectral, inputs_texture)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss_train += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        # 记录每个batch的loss和准确率
        batch_train_losses.append(loss.item())
        batch_accuracy = 100 * correct_train / total_train
        batch_train_accuracies.append(batch_accuracy)

    # 每个epoch的平均训练loss和准确率
    epoch_train_loss = total_loss_train / len(train_loader)
    epoch_train_accuracy = 100 * correct_train / total_train
    epoch_train_losses.append(epoch_train_loss)
    epoch_train_accuracies.append(epoch_train_accuracy)

    # 测试部分
    net.eval()  # 设置模型为评估模式
    total_loss_test = 0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs_spectral, inputs_texture = inputs[:, :, :10, :, :], inputs[:, :, 10:, :, :]  # 分割inputs
            inputs_spectral, inputs_texture, labels = inputs_spectral.to(device), inputs_texture.to(device), labels.to(device)

            outputs = net(inputs_spectral, inputs_texture)
            loss = criterion(outputs, labels)

            total_loss_test += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    # 每个epoch的平均测试loss和准确率
    epoch_test_loss = total_loss_test / len(test_loader)
    epoch_test_accuracy = 100 * correct_test / total_test
    epoch_test_losses.append(epoch_test_loss)
    epoch_test_accuracies.append(epoch_test_accuracy)

    # 更新最佳模型（如果适用）
    if epoch_test_accuracy > best_acc:
        best_acc = epoch_test_accuracy
        best_model_wts = copy.deepcopy(net.state_dict())

    print(f"[Epoch {epoch+1}] Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.2f}%, Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_accuracy:.2f}%")

print('Finished Training')
print(f'Best Testing Accuracy: {best_acc:.2f}%')

# 加载最佳模型权重
net.load_state_dict(best_model_wts)

# 保存预训练最佳模型
output_filename = 'best_transfer_model_' + os.path.basename(X_spectral_path).split('.')[0] + '.pth'
torch.save(net.state_dict(), output_filename)
print(f"预训练最佳模型已保存为: {output_filename}")


# In[ ]:


df = pd.DataFrame({
    "Batch Train Losses": pd.Series(batch_train_losses),
    "Batch Train Accuracies": pd.Series(batch_train_accuracies),
    "Epoch Train Losses": pd.Series(epoch_train_losses),
    "Epoch Train Accuracies": pd.Series(epoch_train_accuracies),
    "Epoch Test Losses": pd.Series(epoch_test_losses),
    "Epoch Test Accuracies": pd.Series(epoch_test_accuracies)
})

# 指定保存路径，注意更改为你想要保存的路径
save_path = "testing_ResNetX2-50.csv"

# 保存为CSV
df.to_csv(save_path, index=False)


# ## (3) 迁移测试

# ### 1) 测试结果

# In[ ]:


# 调用模型
net = HybridSNLateFusion(class_num)
net.load_state_dict(torch.load('transfer_45-front.pth'))
net = net.to(device)

net.eval()  # 设置模型为评估模式

y_pred = []  # 用于存储预测的类别
y_score = []  # 用于存储预测的概率
y_true = []  # 真实标签

with torch.no_grad():
    for inputs, labels in test_loader:
        # 分割合并后的数据为光谱数据和纹理数据
        inputs_spectral = inputs[:, :, :10, :, :]
        inputs_texture = inputs[:, :, 10:, :, :]

        # 将数据移动到指定设备
        inputs_spectral, inputs_texture, labels = inputs_spectral.to(device), inputs_texture.to(device), labels.to(device)

        # 正向传播，获取输出
        outputs = net(inputs_spectral, inputs_texture)

        # 获取预测结果及其概率
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

        # 保存预测、概率和实际标签
        y_pred.extend(predicted.cpu().numpy())
        y_score.extend(probabilities.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

# 将列表转换为 NumPy 数组
y_pred = np.array(y_pred)
y_true = np.array(y_true)
y_score = np.array(y_score)

# 生成分类报告
classification = classification_report(y_true, y_pred, digits=4)
print(classification)


# ### 2) 评价指标

# In[ ]:


# 获取所有唯一的标签
unique_labels = np.unique(np.concatenate([y_true, y_pred]))

# 计算指标
precision, recall, f1_score, support = precision_recall_fscore_support(y_true, y_pred, average=None, labels=unique_labels)
overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', labels=unique_labels)

# 指标DataFrame
metrics_df = pd.DataFrame({
    'Label': [f'Class {label}' for label in unique_labels] + ['Overall'],
    'Precision': np.append(precision, overall_precision),
    'Recall': np.append(recall, overall_recall),
    'F1-Score': np.append(f1_score, overall_f1),
    'Support': np.append(support, np.sum(support))
})

# 未归一化混淆矩阵
conf_mat_1 = confusion_matrix(y_true, y_pred, labels=unique_labels)
conf_mat = conf_mat_1.T
conf_mat_df = pd.DataFrame(conf_mat, index=[f' {label}' for label in unique_labels], columns=[f'Predicted Class {label}' for label in unique_labels])

# 对混淆矩阵进行归一化处理
conf_mat_normalized = (conf_mat.astype('float') / conf_mat.sum(axis=0)[np.newaxis, :]) * 100

# 使用向下取整并确保类型为整数
conf_mat_normalized_floor = np.floor(conf_mat_normalized).astype(int)

# 调整每列以保证总和为100
for i in range(conf_mat_normalized_floor.shape[1]):  # 对每一列
    column_sum = conf_mat_normalized_floor[:, i].sum()
    # 找到差额并加到该列最小值上
    if column_sum != 100:
        min_indices = np.where(conf_mat_normalized_floor[:, i] == conf_mat_normalized_floor[:, i].min())[0]
        # 如果有多个最小值，选择一个用于调整
        min_index = np.random.choice(min_indices)
        conf_mat_normalized_floor[min_index, i] += (100 - column_sum)

# 将调整后的归一化混淆矩阵转换为DataFrame
conf_mat_norm_df = pd.DataFrame(conf_mat_normalized_floor, index=[f'{label}' for label in unique_labels], columns=[f'{label}' for label in unique_labels])

# 保存到Excel文件的过程保持不变
save_path = 'classification_metrics_ResNetX2-50.xlsx'
with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
    metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
    conf_mat_df.to_excel(writer, sheet_name='Confusion Matrix', index=True)
    conf_mat_norm_df.to_excel(writer, sheet_name='Normalized Confusion Matrix', index=True)

print(f"文件已保存为 '{save_path}'")



