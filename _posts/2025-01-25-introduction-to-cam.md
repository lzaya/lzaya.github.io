---
layout: post
title: "CAM 介绍"
date: 2025-01-25
categories: [深度学习]
---

# CAM（Class Activation Mapping）介绍

CAM（Class Activation Mapping）是一种用来解释卷积神经网络（CNN）决策的技术。它通过生成热力图来显示图像中的哪些区域对分类结果贡献最大。

主要步骤：
1. 获取预测类别的概率值。
2. 计算该类别相对于最后卷积层输出的梯度。
3. 使用梯度加权的特征图生成热力图。

这篇文章将介绍 CAM 的基本原理，并通过代码实现展示其应用。
