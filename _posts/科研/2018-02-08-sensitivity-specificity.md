---
layout: post
title: 灵敏度和特异性指标
category: 科研
tags: 医疗
keywords: 医疗指标
description:
---

# 简介

医学领域的常用指标，灵敏性与假阴性率（漏诊率），特异性与假阳性率（误诊率）

灵敏度（也称真阳性率，sensitivity）=真阳性人数/（真阳性人数+假阴性人数）*100%

特异性（也称真阴性率，specificity）=真阴性人数/（真阴性人数+假阳性人数）*100%

True Positive （真正, TP）被模型预测为正的正样本；可以称作判断为真的正确率

True Negative（真负 , TN）被模型预测为负的负样本 ；可以称作判断为假的正确率

False Positive （假正, FP）被模型预测为正的负样本；可以称作误报率

False Negative（假负 , FN）被模型预测为负的正样本；可以称作漏报率

True Positive Rate（真正率 , TPR）或灵敏度（sensitivity）： 

TPR = TP /（TP + FN）

正样本预测结果数 / 正样本实际数

True Negative Rate（真负率 , TNR）或特异性（specificity）：
 
TNR = TN /（TN + FP）
 
负样本预测结果数 / 负样本实际数

False Positive Rate （假正率, FPR）：
 
FPR = FP /（FP + TN）

被预测为正的负样本结果数 /负样本实际数

False Negative Rate（假负率 , FNR）：

FNR = FN /（TP + FN） 

被预测为负的正样本结果数 / 正样本实际数

精确度（Precision）： 

P = TP/(TP+FP) ; 反映了被分类器判定的正例中真正的正例样本的比重

准确率（Accuracy）：

A = (TP + TN)/(P+N) = (TP + TN)/(TP + FN + FP + TN);

反映了分类器统对整个样本的判定能力——能将正的判定为正，负的判定为负

召回率(Recall)，也称为 True Positive Rate: 

R = TP/(TP+FN) = 1 - FN/T; 反映了被正确判定的正例占总的正例的比重

看我写的辛苦求打赏啊！！！有学术讨论和指点请加微信manutdzou,注明

![20](/public/img/pay.jpg)