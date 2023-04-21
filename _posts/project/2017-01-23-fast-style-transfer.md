---
layout: post
title: Fast-style-transfer
category: project
tags: 深度学习
keywords: 应用
description: 
---

#  Perceptual Losses for Real-Time Style Transfer and Super-Resolution

这篇文章是关于快速艺术风格图片迁移的文章，思路非常简单。

项目代码： https://github.com/OlavHN/fast-neural-style

算法的原理非常简单，请看下图：

![1](/public/img/posts/fast style/net.png)

风格图像$y_{s}$和待迁移的图像$y_{c}$输入固定参数的VGG-16 loss network，这个网络是由Imagenet训练出来的，保证了隐藏层具有的高级语义特征。这个过程只需要一次前向传播获得特征层的feature map.另外输入图像$x$也就是$y_{c}$将输入风格迁移网络$f_{W}$获得$y\hat{}$,再把$y\hat{}$输入固定的VGG-16分别和$y_{s}$和$y_{c}$的指定特征层算loss.算法的目标函数是最小化这个loss和来训练$f_{W}$这个网络。

对于图像超分辨率重构算法，只需要$y_{c}$和$x$两个输入，其中$x$是$y_{c}$的低分辨率图像，$y_{c}$是高分辨率图像，算法目标是最小化$x$经过$f_{W}$后获得的$y\hat{}$在VGG-16中和$y_{c}$在VGG-16中指定特征层的误差，使得训练$f_{W}$网络，获得$y\hat{}$为$x$的高分辨率图像。

下面给出我训练的快速迁移网络的结果

风格图像

![2](/public/img/posts/fast style/style.jpg)

内容图像

![3](/public/img/posts/fast style/004545.jpg)

迁移图像

![4](/public/img/posts/fast style/res.jpg)
