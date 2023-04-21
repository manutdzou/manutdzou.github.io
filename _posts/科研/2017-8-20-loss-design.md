---
layout: post
title: 深度学习中的loss函数汇总
category: 科研
tags: 深度学习
keywords: loss function
description:
---

# Softmax loss

```
def get_softmax_loss(features,one_hot_labels):
    prob = tf.nn.softmax(features + 1e-5)
    cross_entropy = tf.multiply(one_hot_labels,tf.log(tf.clip_by_value(prob,1e-5,1.0)))
    loss = -tf.reduce_mean(cross_entropy)
    return loss
```

# Center loss

[参考](http://www.jianshu.com/p/773fbd0b2472)

```
def get_center_loss(features, labels, alpha, num_classes):
    # alpha:中心的更新比例
    # 获取特征长度
    len_features = features.get_shape()[1]
    # 建立一个变量，存储每一类的中心，不训练
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    # 将label reshape成一维
    labels = tf.reshape(labels, [-1])
 
    # 获取当前batch每个样本对应的中心
    centers_batch = tf.gather(centers, labels)
    # 计算center loss的数值
    loss = tf.nn.l2_loss(features - centers_batch)
 
    # 以下为更新中心的步骤
    diff = centers_batch - features
 
    # 获取一个batch中同一样本出现的次数，这里需要理解论文中的更新公式
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])
 
    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff
    # 更新中心
    centers = tf.scatter_sub(centers, labels, diff)
 
    return loss, centers
```

# Focal loss

[参考](http://blog.csdn.net/yaoqi_isee/article/details/77051205)

```
def get_focal_loss(features,one_hot_labels,n):
    prob = tf.nn.softmax(features + 1e-5)
    cross_entropy = tf.multiply(one_hot_labels,tf.log(tf.clip_by_value(prob,1e-5,1.0)))
    weight = tf.pow(tf.subtract(1.0,prob),n)
    loss = -tf.reduce_mean(tf.multiply(weight,cross_entropy))
    return loss
```

# Triplet loss

[参考](http://blog.csdn.net/tangwei2014/article/details/46788025)

```
def compute_triplet_loss(anchor_feature, positive_feature, negative_feature, margin):

    """
    Compute the contrastive loss as in
    L = || f_a - f_p ||^2 - || f_a - f_n ||^2 + m
    **Parameters**
     anchor_feature:
     positive_feature:
     negative_feature:
     margin: Triplet margin
    **Returns**
     Return the loss operation
    """
	
    def compute_euclidean_distance(x, y):
        """
        Computes the euclidean distance between two tensorflow variables
        """
        d = tf.square(tf.sub(x, y))
        d = tf.sqrt(tf.reduce_sum(d)) # What about the axis ???
        return d

	
    with tf.name_scope("triplet_loss"):
        d_p_squared = tf.square(compute_euclidean_distance(anchor_feature, positive_feature))
        d_n_squared = tf.square(compute_euclidean_distance(anchor_feature, negative_feature))

        loss = tf.maximum(0., d_p_squared - d_n_squared + margin)
        #loss = d_p_squared - d_n_squared + margin

        return tf.reduce_mean(loss), tf.reduce_mean(d_p_squared), tf.reduce_mean(d_n_squared)
```

# Huber_loss

```
def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)
```

看我写的辛苦求打赏啊！！！有学术讨论和指点请加微信manutdzou,注明

![20](/public/img/pay.jpg)