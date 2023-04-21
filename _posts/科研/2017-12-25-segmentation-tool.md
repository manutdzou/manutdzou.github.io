---
layout: post
title: segmentation标注工具
category: 科研
tags: 深度学习
keywords: 标注工具
description:
---

# Installation

```
git clone https://github.com/wkentaro/labelme.git
python setup.py install
```

# Usage

在终端labelme打开软件

![1](/public/img/posts/segmentation tool/1.PNG)

Open打开图片

Create Polygons 勾勒目标轮廓，勾勒完以后命名label，最后保存，如下

![2](/public/img/posts/segmentation tool/2.PNG)

将labelme/scripts下的labelme_draw_json和labelme_json_to_dataset拷贝到存放json文件的路径下

## 展示标注结果

```
python labelme_draw_json COCO_val2014_000000000711.json
```

## 存储标注结果

```
python labelme_json_to_dataset COCO_val2014_000000000711.json
```

将会新建一个同名文件夹存放label.png和可视化label_viz/png

注意label.png是黑色的单通道图片所以看不出东西，0是背景其他是每个类的前景，每个类别对应信息存放在info.yaml里，如下

```
label_names:
- background
- desk
- disk
- picture
- sofa
```
label_viz
![3](/public/img/posts/segmentation tool/label_viz.png)

看我写的辛苦求打赏啊！！！有学术讨论和指点请加微信manutdzou,注明

![20](/public/img/pay.jpg)