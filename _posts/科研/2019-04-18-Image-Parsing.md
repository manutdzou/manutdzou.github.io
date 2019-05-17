---
layout: post
title: Image Parsing
category: 科研
tags: 
keywords: 
description:
---

# Image Parsing

[github](https://github.com/manutdzou/Image-Parsing)

本文的目的是为了复现和分析主流的基于PSPnet和DeepLab系列的图像分割框架。

直接贴结果：实验主要在CityScapes上single scale的测试validation data

首先需要预训练backbone在Imagenet的分类，实验中发现如果train from scartch,将完全无法复现论文的结果而且结果差距会比较大。本文训练了基于不同版本的resnet101作为backbone。

PSPNet backbone     在Imagenet的validation上single test: top1 72.64%, top5 91.16%

Deeplab             在Imagenet的validation上single test: top1 69.38%, top5 88.92%(由于时间问题未充分训练)

PSPNet-GN backbone  在Imagenet的validation上single test: top1 72.09%, top5 90.83%

下面分别展示Imagenet上训练曲线

PSPNet backbone

![1](/public/img/posts/Image Parsing/psp_backbone.PNG)

PSPNet-GN backbone

![2](/public/img/posts/Image Parsing/psp_gn_backbone.PNG)

Deeplab

![3](/public/img/posts/Image Parsing/deeplab_backbone.PNG)

Deeplab v2 mIoU为 71.16 可见使用element-wise add方式聚合不同感受野尺度的特征混乱了多尺度的特征

Deeplab v3 mIoU为 76.03

PSPNet mIoU为 77.08

PSPNet GN mIoU为 76.11 实验表明利用Group normalization在GPU显存受限时候能得到多卡Synchronous batch normalization接近的结果

看我写的辛苦求打赏啊！！！有学术讨论和指点请加微信manutdzou,注明

![20](/public/img/pay.jpg)