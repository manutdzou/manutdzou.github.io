---
layout: post
title: tensorflow模型恢复与inference的模型简化
category: tensorflow
tags: 深度学习
keywords: tf学习
description: tf学习
---

# Finetune or restore model In Tensorflow

# 打印模型参数名

```
import tensorflow as tf
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

output_path = './checkpoint'
ckpt = tf.train.get_checkpoint_state(output_path)
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta')

var = [v for v in tf.trainable_variables()]
#var = [v for v in tf.get_default_graph().as_graph_def().node]

for i in range(len(var)):
    logging.info(var[i].name)
```

# 恢复模型时指定某些layer不恢复

```
not_restore = ['Variable_44:0',"Variable_45:0"]
restore_var = [v for v in tf.trainable_variables() if v.name not in not_restore]

saver = tf.train.Saver(restore_var)
ckpt = tf.train.get_checkpoint_state('./checkpoint')
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, ckpt.model_checkpoint_path)
```

# restore without graph

```
output_path = './checkpoint'
ckpt = tf.train.get_checkpoint_state(output_path)
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta')

g = tf.get_default_graph()
input_tensor = g.get_tensor_by_name("input:0")
output_tensor = g.get_tensor_by_name("output:0")

with tf.Session() as sess:
    saver.restore(sess, ckpt.model_checkpoint_path)
    output = sess.run(output_tensor, feed_dict={input_tensor: x_test})
```

