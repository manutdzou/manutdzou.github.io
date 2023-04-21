---
layout: post
title: From conv to atrous
category: 科研
tags: 
keywords: 
description:
---

在finetune模型的时候经常需要将一个分类的预训练模型迁移到其他任务比如分割和检测，但是由于分类模型经常通过stride很快将feature map缩减到1/32，过小的尺寸将不利于分割或者检测细粒度的目标。所以需要在迁移模型参数时候保持feature map不能缩小到太小。

在模型迁移的时候，为了保证迁移的参数可靠性，必须要保证迁移前和迁移后对应层的kernel有相同感受野的feature map.否则将导致模型迁移后特征失效。

下面以resnet为例子，来分析一下如何将Imagenet上训练的分类模型迁移到相同backbone的分割网络：

分类模型

```
(self.feed('conv3_4/relu')
    .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_1_1x1_reduce')
    .batch_normalization(relu=True, name='conv4_1_1x1_reduce_bn')
    .zero_padding(paddings=1, name='padding8')
    .conv(3, 3, 256, 2, 2, biased=False, relu=False, name='conv4_1_3x3') #stride =16
    .batch_normalization(relu=True, name='conv4_1_3x3_bn')
    .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_1_1x1_increase')
    .batch_normalization(relu=False, name='conv4_1_1x1_increase_bn'))

(self.feed('conv4_1_1x1_proj_bn',
           'conv4_1_1x1_increase_bn')
    .add(name='conv4_1')
    .relu(name='conv4_1/relu')
    .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_2_1x1_reduce')
    .batch_normalization(relu=True, name='conv4_2_1x1_reduce_bn')
    .zero_padding(paddings=1, name='padding9')
    .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='conv4_2_3x3') #stride =16
    .batch_normalization(relu=True, name='conv4_2_3x3_bn')
    .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_2_1x1_increase')
    .batch_normalization(relu=False, name='conv4_2_1x1_increase_bn'))
```

对应的分割模型部分，使用了atrous算法

```
(self.feed('conv3_4/relu')
    .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_1_1x1_reduce')
    .batch_normalization(relu=True, name='conv4_1_1x1_reduce_bn')
    .zero_padding(paddings=2, name='padding8')
    .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_1_3x3') #stride =8
    .batch_normalization(relu=True, name='conv4_1_3x3_bn')
    .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_1_1x1_increase')
    .batch_normalization(relu=False, name='conv4_1_1x1_increase_bn'))

(self.feed('conv4_1_1x1_proj_bn',
           'conv4_1_1x1_increase_bn')
    .add(name='conv4_1')
    .relu(name='conv4_1/relu')
    .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_2_1x1_reduce')
    .batch_normalization(relu=True, name='conv4_2_1x1_reduce_bn')
    .zero_padding(paddings=2, name='padding9')
    .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_2_3x3') #stride =8
    .batch_normalization(relu=True, name='conv4_2_3x3_bn')
    .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_2_1x1_increase')
    .batch_normalization(relu=False, name='conv4_2_1x1_increase_bn'))
```

原则就是将对应stride=2的conv地方改成atrous=2的conv,然后后续所有的大于1的卷积核都需要添加空洞来保持前后感受野一致，当多次修改stride后对应大于1的卷积核要根据stride的次数添加更多的空洞。


看我写的辛苦求打赏啊！！！有学术讨论和指点请加微信manutdzou,注明

![20](/public/img/pay.jpg)