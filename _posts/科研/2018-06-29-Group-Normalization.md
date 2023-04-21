---
layout: post
title: Group Normalization
category: 科研
tags: 
keywords: 
description:
---

# Group Normalization

![1](/public/img/posts/papers/gn.png)

BN在batch的维度上norm，归一化维度为[N，H，W]，对batch中对应的channel归一化;

LN避开了batch维度，归一化的维度为[C，H，W];

IN归一化的维度为[H，W];

而GN介于LN和IN之间，其首先将channel分为许多组，对每一组做归一化，及先将feature的维度由[N, C, H, W]reshape为[N, G，C//G , H, W]，归一化的维度为[C//G , H, W];

```
def norm(x, norm_type, is_train, G=32, esp=1e-5):
    with tf.variable_scope('{}_norm'.format(norm_type)):
        if norm_type == 'none':
            output = x
        elif norm_type == 'batch':
            output = tf.contrib.layers.batch_norm(
                x, center=True, scale=True, decay=0.999,
                is_training=is_train, updates_collections=None
            )
        elif norm_type == 'group':
            # normalize
            # tranpose: [bs, h, w, c] to [bs, c, h, w] following the paper
            x = tf.transpose(x, [0, 3, 1, 2])
            N, C, H, W = x.get_shape().as_list()
            G = min(G, C)
            x = tf.reshape(x, [N, G, C // G, H, W])
            mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
            x = (x - mean) / tf.sqrt(var + esp)
            # per channel gamma and beta
            gamma = tf.get_variable('gamma', [C],
                                    initializer=tf.constant_initializer(1.0))
            beta = tf.get_variable('beta', [C],
                                   initializer=tf.constant_initializer(0.0))
            gamma = tf.reshape(gamma, [1, C, 1, 1])
            beta = tf.reshape(beta, [1, C, 1, 1])

            output = tf.reshape(x, [N, C, H, W]) * gamma + beta
            # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
            output = tf.transpose(output, [0, 2, 3, 1])
        else:
            raise NotImplementedError
    return output
```

看我写的辛苦求打赏啊！！！有学术讨论和指点请加微信manutdzou,注明

![20](/public/img/pay.jpg)