---
layout: post
title: tensorflow ckpt to pb
category: 科研
tags: 
keywords: 
description:
---

假设我们通过tensorflow的dataset类训练好了一个densenet的模型，我们想要将checkpoint固化成pb文件部署使用，但是dataset类训练的模型不包含数据输入接口，所以我们需要转换模型接口

下面是model_zoo.py

```
import numpy as np
import tensorflow as tf
import six
slim = tf.contrib.slim

class Densenet(object):
    """Densenet model."""
    def __init__(self, num_class, images, is_training):
        self.num_classes = num_class
        self._images = images
        self.training  = is_training
    
    def build_graph(self):
        """Build a whole graph for the model."""
        self._build_model()
    
    def _build_model(self):
        with slim.arg_scope(self.densenet_arg_scope()):
            self.logits, _ = self.densenet169(self._images, num_classes=self.num_classes, is_training=self.training)

    @slim.add_arg_scope
    def _global_avg_pool2d(self, inputs, data_format='NHWC', scope=None, outputs_collections=None):
        with tf.variable_scope(scope, 'xx', [inputs]) as sc:
            axis = [1, 2] if data_format == 'NHWC' else [2, 3]
            net = tf.reduce_mean(inputs, axis=axis, keep_dims=True)
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)
            return net


    @slim.add_arg_scope
    def _conv(self, inputs, num_filters, kernel_size, stride=1, dropout_rate=None, scope=None, outputs_collections=None):
        with tf.variable_scope(scope, 'xx', [inputs]) as sc:
            net = slim.batch_norm(inputs)
            net = tf.nn.relu(net)
            net = slim.conv2d(net, num_filters, kernel_size)

            if dropout_rate:
                net = tf.nn.dropout(net)

            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

        return net


    @slim.add_arg_scope
    def _conv_block(self, inputs, num_filters, data_format='NHWC', scope=None, outputs_collections=None):
        with tf.variable_scope(scope, 'conv_blockx', [inputs]) as sc:
            net = inputs
            net = self._conv(net, num_filters*4, 1, scope='x1')
            net = self._conv(net, num_filters, 3, scope='x2')
            if data_format == 'NHWC':
                net = tf.concat([inputs, net], axis=3)
            else: # "NCHW"
                net = tf.concat([inputs, net], axis=1)

            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

        return net


    @slim.add_arg_scope
    def _dense_block(self, inputs, num_layers, num_filters, growth_rate,
                 grow_num_filters=True, scope=None, outputs_collections=None):

        with tf.variable_scope(scope, 'dense_blockx', [inputs]) as sc:
            net = inputs
            for i in range(num_layers):
                branch = i + 1
                net = self._conv_block(net, growth_rate, scope='conv_block'+str(branch))

                if grow_num_filters:
                    num_filters += growth_rate

            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

        return net, num_filters


    @slim.add_arg_scope
    def _transition_block(self, inputs, num_filters, compression=1.0,
                      scope=None, outputs_collections=None):

        num_filters = int(num_filters * compression)
        with tf.variable_scope(scope, 'transition_blockx', [inputs]) as sc:
            net = inputs
            net = self._conv(net, num_filters, 1, scope='blk')

            net = slim.avg_pool2d(net, 2)

            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

        return net, num_filters


    def densenet(self, inputs,
             num_classes=1000,
             reduction=None,
             growth_rate=None,
             num_filters=None,
             num_layers=None,
             dropout_rate=None,
             data_format='NHWC',
             is_training=True,
             reuse=None,
             scope=None):
        assert reduction is not None
        assert growth_rate is not None
        assert num_filters is not None
        assert num_layers is not None

        compression = 1.0 - reduction
        num_dense_blocks = len(num_layers)

        if data_format == 'NCHW':
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

        with tf.variable_scope(scope, 'densenetxxx', [inputs, num_classes],
                         reuse=reuse) as sc:
            end_points_collection = sc.name + '_end_points'
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                             is_training=is_training), \
                slim.arg_scope([slim.conv2d, self._conv, self._conv_block,
                             self._dense_block, self._transition_block], 
                             outputs_collections=end_points_collection), \
                slim.arg_scope([self._conv], dropout_rate=dropout_rate):
                net = inputs

                # initial convolution
                net = slim.conv2d(net, num_filters, 7, stride=2, scope='conv1')
                net = slim.batch_norm(net)
                net = tf.nn.relu(net)
                net = slim.max_pool2d(net, 3, stride=2, padding='SAME')

                # blocks
                for i in range(num_dense_blocks - 1):
                    # dense blocks
                    net, num_filters = self._dense_block(net, num_layers[i], num_filters,
                                        growth_rate,
                                        scope='dense_block' + str(i+1))

                    # Add transition_block
                    net, num_filters = self._transition_block(net, num_filters,
                                             compression=compression,
                                             scope='transition_block' + str(i+1))

                net, num_filters = self._dense_block(
                    net, num_layers[-1], num_filters,
                    growth_rate,
                    scope='dense_block' + str(num_dense_blocks))

                # final blocks
                with tf.variable_scope('final_block', [inputs]):
                    net = slim.batch_norm(net)
                    net = tf.nn.relu(net)
                    net = self._global_avg_pool2d(net, scope='global_avg_pool')

                net = slim.conv2d(net, num_classes, 1,
                        biases_initializer=tf.zeros_initializer(),
                        scope='logits')

                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)

                if num_classes is not None:
                    end_points['predictions'] = slim.softmax(net, scope='predictions')

                return tf.reduce_mean(net,[1,2]), end_points


    def densenet121(self, inputs, num_classes=1000, data_format='NHWC', is_training=True, reuse=None):
        return self.densenet(inputs,
                  num_classes=num_classes, 
                  reduction=0.5,
                  growth_rate=32,
                  num_filters=64,
                  num_layers=[6,12,24,16],
                  data_format=data_format,
                  is_training=is_training,
                  reuse=reuse,
                  scope='densenet121')
    densenet121.default_image_size = 224


    def densenet161(self, inputs, num_classes=1000, data_format='NHWC', is_training=True, reuse=None):
        return self.densenet(inputs,
                  num_classes=num_classes, 
                  reduction=0.5,
                  growth_rate=48,
                  num_filters=96,
                  num_layers=[6,12,36,24],
                  data_format=data_format,
                  is_training=is_training,
                  reuse=reuse,
                  scope='densenet161')
    densenet161.default_image_size = 224


    def densenet169(self, inputs, num_classes=1000, data_format='NHWC', is_training=True, reuse=None):
        return self.densenet(inputs,
                  num_classes=num_classes, 
                  reduction=0.5,
                  growth_rate=32,
                  num_filters=64,
                  num_layers=[6,12,32,32],
                  data_format=data_format,
                  is_training=is_training,
                  reuse=reuse,
                  scope='densenet169')
    densenet169.default_image_size = 224


    def densenet_arg_scope(self, weight_decay=1e-4,
                       batch_norm_decay=0.99,
                       batch_norm_epsilon=1.1e-5,
                       data_format='NHWC'):
        with slim.arg_scope([slim.conv2d, slim.batch_norm, slim.avg_pool2d, slim.max_pool2d,
                       self._conv_block, self._global_avg_pool2d],
                      data_format=data_format):
            with slim.arg_scope([slim.conv2d],
                         weights_regularizer=slim.l2_regularizer(weight_decay),
                         activation_fn=None,
                         biases_initializer=None):
                with slim.arg_scope([slim.batch_norm],
                          scale=True,
                          decay=batch_norm_decay,
                          epsilon=batch_norm_epsilon) as scope:
                    return scope

```

inference.py是一个封装上面的模型的类

```
import model_zoo
import numpy as np
import tensorflow as tf

class Model_Graph(object):
    def __init__(self, num_class = 2, is_training = True):
        self.num_class = num_class
        self.is_training = is_training

    def _build_defaut_graph(self, images):
        """
        Densenet
        """
        model = model_zoo.Densenet(num_class = self.num_class,
                                     images = images, is_training = self.is_training)
        model.build_graph()

        return model
```

如下代码是生成一个带placeholder输入的模型，将dataset接口转化为placeholder

```
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path
import time
from inference import Model_Graph

import numpy as np
import tensorflow as tf

def save(saver, sess, logdir):
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)

   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path)
   print('The checkpoint has been created.')

def process_image(image_decoded):
    #data argumentation
    image_decoded = tf.image.random_flip_left_right(image_decoded)
    image_decoded = tf.image.random_flip_up_down(image_decoded)
    image_decoded = tf.image.random_brightness(image_decoded, max_delta=0.1)
    image_decoded = tf.image.random_saturation(image_decoded, lower=0.7, upper=1.3)
    image_decoded = tf.image.random_contrast(image_decoded, lower=0.7, upper=1.3)

    channel_mean = tf.constant(np.array([177.26,109.55,168.66], dtype=np.float32))
    image = tf.subtract(tf.cast(image_decoded, dtype=tf.float32),channel_mean)
    return tf.reshape(image,[256,256,3])

def generate_placeholder_ckpt():
    with tf.Graph().as_default():
        images = tf.placeholder(tf.float32, (None, 256,256,3),name='input')

        image_batch = tf.map_fn(lambda img: process_image(img), images)

        graph = Model_Graph(num_class = 2, is_training = False)

        model = graph._build_defaut_graph(images = image_batch)
        prob = tf.nn.softmax(model.logits, name ='probs')
        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(0.99)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        #saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state('densenet_model')
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print('Successfully loaded model from %s at step=%s.' %(ckpt.model_checkpoint_path, global_step))
            save(saver, sess, 'model')

if __name__ == '__main__':
    generate_placeholder_ckpt()
```

假如我们想部署一个多卡的版本，使得每个卡负责一组数据的预测处理，所有卡上模型共享一组参数，如下

```
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path
import time
from inference import Model_Graph

import numpy as np
import tensorflow as tf

def save(saver, sess, logdir):
   model_name = 'multi_model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)

   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path)
   print('The checkpoint has been created.')

def process_image(image_decoded):
    #data argumentation
    image_decoded = tf.image.random_flip_left_right(image_decoded)
    image_decoded = tf.image.random_flip_up_down(image_decoded)
    image_decoded = tf.image.random_brightness(image_decoded, max_delta=0.1)
    image_decoded = tf.image.random_saturation(image_decoded, lower=0.7, upper=1.3)
    image_decoded = tf.image.random_contrast(image_decoded, lower=0.7, upper=1.3)

    channel_mean = tf.constant(np.array([177.26,109.55,168.66], dtype=np.float32))
    image = tf.subtract(tf.cast(image_decoded, dtype=tf.float32),channel_mean)
    return tf.reshape(image,[256,256,3])

def generate_placeholder_ckpt():
    with tf.Graph().as_default():
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(2):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ('Card', i)) as scope:
                        images = tf.placeholder(tf.float32, (None, 256,256,3),name='input')

                        image_batch = tf.map_fn(lambda img: process_image(img), images)

                        graph = Model_Graph(num_class = 2, is_training = False)

                        model = graph._build_defaut_graph(images = image_batch)
                        prob = tf.nn.softmax(model.logits, name ='probs')

                        tf.get_variable_scope().reuse_variables()

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(0.99)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        #saver = tf.train.Saver(tf.global_variables())

        config = tf.ConfigProto(allow_soft_placement = True)
        with tf.Session(config = config) as sess:
            ckpt = tf.train.get_checkpoint_state('densenet_model')
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print('Successfully loaded model from %s at step=%s.' %(ckpt.model_checkpoint_path, global_step))
            save(saver, sess, 'model')

if __name__ == '__main__':
    generate_placeholder_ckpt()
```

如下代码可以实现checkpoint向pb的转化

```
# -*-coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np
 
def freeze_graph_test(pb_path):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
 
            input_image_tensor = sess.graph.get_tensor_by_name("input:0")
 
            output_tensor_name = sess.graph.get_tensor_by_name("probs:0")

            #输入模型需要的输入
            im=np.ones((1,256,256,3))
            out=sess.run(output_tensor_name, feed_dict={input_image_tensor: im})
            print("out:{}".format(out))
 
def freeze_graph(input_checkpoint,output_graph):
    output_node_names = "probs"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
 
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=sess.graph_def,
            output_node_names=output_node_names.split(","))
 
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))
        #for op in sess.graph.get_operations():
        #    print(op.name, op.values()) 

def freeze_graph_v2(input_checkpoint,output_graph):
    output_node_names = "probs"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
 
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=output_node_names.split(","))
 
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))
        #for op in sess.graph.get_operations():
        #    print(op.name, op.values())

if __name__ == '__main__':
    #model中存储着checkpoint,data,index,meta
    input_checkpoint='model'
    ckpt = tf.train.get_checkpoint_state(input_checkpoint)
    out_pb_path="model/frozen_model.pb"
    freeze_graph(ckpt.model_checkpoint_path, out_pb_path)
 
    freeze_graph_test(pb_path=out_pb_path)
```

checkpoint转pb的多卡版本

```
# -*-coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np
 
def freeze_graph_test(pb_path):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        config = tf.ConfigProto(allow_soft_placement = True)
        with tf.Session(config = config) as sess:

            sess.run(tf.global_variables_initializer())
 
            input_image_tensor_a = sess.graph.get_tensor_by_name("Card_0/input:0")
            input_image_tensor_b = sess.graph.get_tensor_by_name("Card_1/input:0")
 
            output_tensor_name_a = sess.graph.get_tensor_by_name("Card_0/probs:0")
            output_tensor_name_b = sess.graph.get_tensor_by_name("Card_1/probs:0")
 
            im=np.ones((1,256,256,3))
            
            out1,out2=sess.run([output_tensor_name_a,output_tensor_name_b], feed_dict={input_image_tensor_a: im,input_image_tensor_b:im})
            print("out1:{},out2:{}".format(out1,out2))
 
def freeze_graph(input_checkpoint,output_graph):
    output_node_names = "Card_0/probs,Card_1/probs"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=False)

    config = tf.ConfigProto(allow_soft_placement = True)
    with tf.Session(config = config) as sess:
        saver.restore(sess, input_checkpoint)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=sess.graph_def,
            output_node_names=output_node_names.split(","))
 
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))
        #for op in sess.graph.get_operations():
        #    print(op.name, op.values()) 

def freeze_graph_v2(input_checkpoint,output_graph):
    output_node_names = "Card_0/probs,Card_1/probs"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=False)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    config = tf.ConfigProto(allow_soft_placement = True)
    with tf.Session(config = config) as sess: 
        saver.restore(sess, input_checkpoint)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=output_node_names.split(","))
 
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))
        #for op in sess.graph.get_operations():
        #    print(op.name, op.values())

if __name__ == '__main__':
    input_checkpoint='model'
    ckpt = tf.train.get_checkpoint_state(input_checkpoint)
    out_pb_path="model/multi_frozen_model.pb"
    freeze_graph(ckpt.model_checkpoint_path, out_pb_path)
 
    freeze_graph_test(pb_path=out_pb_path)
```


看我写的辛苦求打赏啊！！！有学术讨论和指点请加微信manutdzou,注明

![20](/public/img/pay.jpg)