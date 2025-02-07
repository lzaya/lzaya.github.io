---
layout: post
title: "CAM与Grad_CAM的原理和代码"
date: 2025-01-29
---

## 前言

深度学习模型，通常被认为是“黑箱”，即输入和输出之间的关系不易理解。通过模型的可视化，可以让我们更好得理解和改进模型。可视化中最著名的方法就是CAM家族，英文名是Class Activation Mapping，中文名是类激活映射。

CAM家族看起来成员众多，随着时间发展，名字也越来越复杂唬人。但他们的思路都是：
$$
XX-CAM = weight * FeatureMaps.
$$
其中weight是CAM成员间的区别，feature maps是模型最后一层的特征图。

## CAM

### 1. 分类网络ResNet18

我们以resnet18举例来说明计算流程。用于分类任务的resent18的结构中包含17个卷积层和1个全连接层，具体的细节可以参照下面的表格，输入图像的大小是224 $\times$ 224：


| layer name  | output size         | details                        |
|-------------|---------------------|--------------------------------|
| stem1       | 112 $\times$ 112    | 7 $\times$ 7, 64, stride 2     |
| stem2       | 56 $\times$ 56      | 3 $\times$ 3 maxpool, stride 2 |
| conv1       | 56 $\times$ 56      | $\left[ \begin{matrix} 3 \times 3, & 64 \\ 3 \times 3, & 64 \end{matrix} \right] \times 2$ |
| conv2       | 28 $\times$ 28      | $\left[ \begin{matrix} 3 \times 3, & 128 \\ 3 \times 3, & 128 \end{matrix} \right] \times 2$ |
| conv3       | 14 $\times$ 14      | $\left[ \begin{matrix} 3 \times 3, & 256 \\ 3 \times 3, & 256 \end{matrix} \right] \times 2$ |
| conv4       | 7 $\times$ 7        | $\left[ \begin{matrix} 3 \times 3, & 512 \\ 3 \times 3, & 512 \end{matrix} \right] \times 2$ |
| cls         | 1 $\times$ 1        | GAP -> 1000-d fc, softmax      |

> PS：为了叙述方便，layer name和原论文不同

```python
# python 实现
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
        
def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2,2,2,2], num_classes) 
```
### 2. CAM原理
CAM原文“Learning Deep Features for Discriminative Localization”，发表在cvpr2016上。CAM的原理非常简单：

<p align="center">
    <img src="_posts/cam&gradcam_image/cam.png" alt="CAM论文的示意图" width=500/>
</p>

对类别$K$来说，其类激活映射的计算方式是：
$$
CAM_{K} = W_{K} * FeatureMaps.
$$ 
其中$W_{K}$是全连接层中对应类别$K$的权重，对应图中的$\{w_{1}, w_{2}, ..., w_{n} \}$。FeatureMaps对应图中的$\{f_{1}, f_{2}, ..., f_{n} \}$

### 3. CAM实现

## Grad-CAM
