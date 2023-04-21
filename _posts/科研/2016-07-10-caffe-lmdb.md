---
layout: post
title: 利用caffe与lmdb读写图像数据
category: 科研
tags: 深度学习
keywords: source code
description: 
---

# 简介

lmdb是一种轻量级的数据库，caffe中主要就是使用lmdb模块来进行图像数据集的保存。据说是因为lmdb有读取速度快，支持多线程、多进程并发，等这样那样的优点，注意到这个数据库其实并没有任何压缩处理的作用，它的目的只是为了快速的索引和存取。它的数据都会带着一定的数据结构从而使体积略微增大。

事实上如果仅仅看lmdb的用法是无法直接应用于图像文件的处理的。由于caffe是将图像以它自带的数据类型的形式传入lmdb中的，因此我们必须结合caffe的数据类型才能完成读取和使用。

# 生成数据文件

```Python
#coding:utf-8
import lmdb
import numpy as np
import cv2
import caffe
from caffe.proto import caffe_pb2 
 
#basic setting
lmdb_file = 'lmdb_data'#期望生成的数据文件
batch_size = 200       #lmdb对于数据进行的是先缓存后一次性写入从而提高效率，因此定义一个batch_size控制每次写入的量。
 
# create the leveldb file
lmdb_env = lmdb.open(lmdb_file, map_size=int(1e12))#生成一个数据文件，定义最大空间
lmdb_txn = lmdb_env.begin(write=True)              #打开数据库的句柄
datum = caffe_pb2.Datum()                          #这是caffe中定义数据的重要类型
 
for x in range(1000):
    x+=1
    img=cv2.imread('A/'+str(x)+'.png').convert('RGB')     #从A/文件夹中依次读取图像
 
    # save in datum
    data = img.astype('int').transpose(2,0,1)      #图像矩阵，注意需要调节维度
    #data = np.array([img.convert('L').astype('int')]) #或者这样增加维度
    label = x                                      #图像的标签，为了方便存储，这个必须是整数。
    datum = caffe.io.array_to_datum(data, label)   #将数据以及标签整合为一个数据项
 
    keystr = '{:0>8d}'.format(x-1)                 #lmdb的每一个数据都是由键值对构成的，因此生成一个用递增顺序排列的定长唯一的key
    lmdb_txn.put( keystr, datum.SerializeToString())#调用句柄，写入内存。
 
    # write batch
    if x % batch_size == 0:                        #每当累计到一定的数据量，便用commit方法写入硬盘。
        lmdb_txn.commit()
        lmdb_txn = lmdb_env.begin(write=True)      #commit之后，之前的txn就不能用了，必须重新开一个。
        print 'batch {} writen'.format(x)
 
lmdb_env.close()                                   #结束后记住释放资源，否则下次用的时候打不开。
```

# 读取数据文件

```Python
#coding:utf-8
 
import caffe
import lmdb
import numpy as np
import cv2
from caffe.proto import caffe_pb2
 
lmdb_env = lmdb.open('lmdb_data')#打开数据文件
lmdb_txn = lmdb_env.begin()      #生成句柄
lmdb_cursor = lmdb_txn.cursor()  #生成迭代器指针
datum = caffe_pb2.Datum()        #caffe定义的数据类型
 
for key, value in lmdb_cursor:   #循环获取数据
    datum.ParseFromString(value) #从value中读取datum数据
 
    label = datum.label          #获取标签以及图像数据
    data = caffe.io.datum_to_array(datum)
    print data.shape
    print datum.channels
    image =data.transpose(1,2,0)
    cv2.imshow('cv2.png', image) #显示
    cv2.waitKey(0)
 
cv2.destroyAllWindows()
lmdb_env.close()
```

看我写的辛苦求打赏啊！！！有学术讨论和指点请加微信manutdzou,注明

![20](/public/img/pay.jpg)