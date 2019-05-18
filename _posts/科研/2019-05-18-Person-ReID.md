---
layout: post
title: Person ReID
category: 科研
tags: 
keywords: 
description:
---

# Person ReID

[github](https://github.com/manutdzou/ReID/tree/master/Person_ReID_Baseline)

Person_ReID_Baseline使用resnet50 finetune提特征，加入了triplet loss

[github](https://github.com/manutdzou/ReID/tree/master/Strong_Person_ReID_Baseline)

Strong_Person_ReID_Baseline使用resnet50。在上面基础上修改Last stride为1,增加特征的细粒度；添加Warm up learning rate,防止模型训练初期的抖动；Label smoothing，降低模型的过拟合，平滑分类能力；BNNeck，将抽取特征和ID分类在不同的维度上，增加特征的泛化能力；Center loss，减少类内特征的方差。

## Batch Size 64 : Rank1(mAP)，Re_ranking Rank1(mAP)

|           |Softmax             |Softmax+S1          |Softmax+Triplet     |Softmax+Triplet+S1  |Strong baseline     |
|     ---   |     --             |     --             |     --             |     --             | --                 |
| Market1501|91.5(77.8)91.0(86.0)|91.7(78.7)93.2(91.6)|92.8(81.7)93.0(90.0)|93.3(84.9)94.7(93.6)|93.8(85.4)94.8(93.5)|
| DukeMTMC  |83.3(66.1)84.4(79.0)|82.8(66.9)86.3(88.3)|86.0(72.3)88.8(83.8)|86.4(74.0)90.1(88.3)|86.0(74.4)90.2(88.2)|


看我写的辛苦求打赏啊！！！有学术讨论和指点请加微信manutdzou,注明

![20](/public/img/pay.jpg)