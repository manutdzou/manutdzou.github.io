---
layout: post
title: tensorflow使用记录
category: tensorflow
tags: 深度学习
keywords: tf学习
description: tf学习
---

# 减均值后图像复原

```
image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
channel_mean = tf.constant(np.array([123.68,116.779,103.938], dtype=np.float32))
image_before_process = tf.add(image,channel_mean)
```

# 在构造图时候查看tensor的shape

```
tensor.shape
#returns tensor's static shape, while the graph is being built.

tensor.shape.as_list() 
#returns the static shape as a integer list.

tensor.shape[i].value 
#returns the static shape's i-th dimension size as an integer.

tf.shape(t) 
#returns t's run-time shape as a tensor.

#An example:
x = tf.placeholder(tf.float32, shape=[None, 8]) # x shape is non-deterministic while building the graph.
print(x.shape) # Outputs static shape (?, 8).
shape_t = tf.shape(x)
with tf.Session() as sess:
    print(sess.run(shape_t, feed_dict={x: np.random.random(size=[4, 8])})) # Outputs run-time shape (4, 8).
```

# tf.app.run()pudb调试时候遇到参数错误，运行时候正常

使用tf.app.run(main=main)

# tensorflow查看可用设备

```
from tensorflow.python.client import device_lib as _device_lib
print _device_lib.list_local_devices()
```

# 从checkpoint中读取tensor

```
import os
from tensorflow.python import pywrap_tensorflow

checkpoint_path = os.path.join(model_dir, "model.ckpt")
# Read data from checkpoint file
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
# Print tensor name and values
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key))
```

或者

```
import tensorflow as tf
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
ckpt = tf.train.get_checkpoint_state('./model_dir/')                          # 通过检查点文件锁定最新的模型
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')   # 载入图结构，保存在.meta文件中
var = [v for v in tf.trainable_variables()]

with tf.Session() as sess:
    saver.restore(sess,ckpt.model_checkpoint_path)
    logging.info("load parameter done")
    parameter = []
    for i in range(len(var)):
        parameter.append(sess.run(var[i]))
        logging.info(var[i].name)
```

# 关于opencv读图片和tf.image.decode_image区别

需要在numpy和cv数据的IO以及tf解码时候注意，不然训练模型时和测试模型数据通道顺序不一样会导致模型预测出错

```
import tensorflow as tf
import cv2

image = cv2.imread('test.jpg')   #BGR order
image_string = tf.read_file('test.jpg')                                                                                                                   
image_decoded = tf.image.decode_image(image_string)
with tf.Session() as sess:                                                                                                                                                        
    image_d = sess.run(image_decoded)      #RGB order                                                                                                                                       
```