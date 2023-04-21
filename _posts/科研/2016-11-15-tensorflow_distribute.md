---
layout: post
title: 基于tensorflow的分布式部署
category: 科研
tags: 深度学习
keywords: 应用
description: 
---

# 基于tensorflow的分布式部署

tensorflow本身支持分布式运行，可以单机多卡多机多卡或者单机cpu/gpu混合等等，本人实际测试了这些功能并总结了一下

tensorflow的分布式运行可以基于一个框架例子，python代码如下

```python
#coding=utf-8
import numpy as np
import tensorflow as tf

# Define parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 0.00001, 'Initial learning rate.')
tf.app.flags.DEFINE_integer('steps_to_validate', 1000,
                     'Steps to validate and print loss')

# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

# Hyperparameters
learning_rate = FLAGS.learning_rate
steps_to_validate = FLAGS.steps_to_validate

def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")
# 创建一个集群，包含两个job,每个job中运行若干个task,每个task由一个worker来执行
#Create a tf.train.ClusterSpec that describes all of the tasks in the cluster. This should be the same for each task.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
#Create a tf.train.Server, passing the tf.train.ClusterSpec to the constructor, and identifying the local task with a job name and task index.
#A server can communicate with any other server in the cluster.
  server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":
    with tf.device(tf.train.replica_device_setter(
                    worker_device="/job:worker/task:%d" % FLAGS.task_index,
                    cluster=cluster)):
      # Build model...
      global_step = tf.Variable(0, name='global_step', trainable=False)

      input = tf.placeholder("float")
      label = tf.placeholder("float")

      weight = tf.get_variable("weight", [1], tf.float32, initializer=tf.random_normal_initializer())
      biase  = tf.get_variable("biase", [1], tf.float32, initializer=tf.random_normal_initializer())
      pred = tf.mul(input, weight) + biase

      loss_value = loss(label, pred)

      train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_value, global_step=global_step)
      init_op = tf.initialize_all_variables()
      
      saver = tf.train.Saver()
      tf.scalar_summary('cost', loss_value)
      summary_op = tf.merge_all_summaries()
    # Create a "supervisor", which oversees the training process.
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                            logdir="./checkpoint/",
                            init_op=init_op,
                            summary_op=None,
                            saver=saver,
                            global_step=global_step,
                            save_model_secs=60)   
    # The supervisor takes care of session initialization, restoring from
    # a checkpoint, and closing when done or an error occurs.
    with sv.managed_session(server.target) as sess:
    # Loop until the supervisor shuts down or steps have completed.
      step = 0
      while  step < 100000000:
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.
        train_x = np.random.randn(1)
        train_y = 2 * train_x + np.random.randn(1) * 0.33  + 10
        _, loss_v, step = sess.run([train_op, loss_value,global_step], feed_dict={input:train_x, label:train_y})
        if step % steps_to_validate == 0:
          w,b = sess.run([weight,biase])
          print("step: %d, weight: %f, biase: %f, loss: %f" %(step, w, b, loss_v))
    # Ask for all the services to stop.
    sv.stop()

def loss(label, pred):
  return tf.square(label - pred)



if __name__ == "__main__":
  tf.app.run()
```

假设我们有两台机器，分别搭载了2块显卡，其中一台同时承担一个worker和一个ps,另一台承担一个worker.同时将代码拷贝到两台服务器上，两台服务器互相设置无密码登陆组成集群（cluster）.分别在1号机器上和2号机器上不同终端下分别执行如下命令，CUDA_VISIBLE_DEVICES=0 表示利用gpu0,CUDA_VISIBLE_DEVICES=''表示仅使用cpu.

```
#ps 节点执行： 
CUDA_VISIBLE_DEVICES='' python distribute.py --ps_hosts=172.25.85.12:2222 --worker_hosts=172.25.85.12:2223,172.25.85.114:2222 --job_name=ps     --task_index=0

#worker 节点执行:
CUDA_VISIBLE_DEVICES='' python distribute.py --ps_hosts=172.25.85.12:2222 --worker_hosts=172.25.85.12:2223,172.25.85.114:2222 --job_name=worker --task_index=0

CUDA_VISIBLE_DEVICES='' python distribute.py --ps_hosts=172.25.85.12:2222 --worker_hosts=172.25.85.12:2223,172.25.85.114:2222 --job_name=worker --task_index=1
```

至此完成了分布式的搭建和配置。

集群是由所有的ps和worker组成。在运行tensorflow脚本的时候，集群之间每台机器需要知道谁是谁，比如谁是参数服务器（ps）,谁是计算服务器（worker）。只有当所有的机器配置挂载完成以后程序才会开始运行。参数服务器只负责共享参数的计算更新，计算服务器负责每个单独流图的计算。

这是另一个例子，可见其实框架都一样

如下定义了pc-01为参数服务器，pc-02，pc-03,pc-04为计算服务器。

```python
# cluster specification
parameter_servers = ["pc-01:2222"]
workers = [ "pc-02:2222", 
            "pc-03:2222",
            "pc-04:2222"]
cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})
```

完整的代码如下

```python
'''
Distributed Tensorflow example of using data parallelism and share model parameters.
Trains a simple sigmoid neural network on mnist for 20 epochs on three machines using one parameter server. 

Change the hardcoded host urls below with your own hosts. 
Run like this: 

pc-01$ python example.py --job_name="ps" --task_index=0 
pc-02$ python example.py --job_name="worker" --task_index=0 
pc-03$ python example.py --job_name="worker" --task_index=1 
pc-04$ python example.py --job_name="worker" --task_index=2 
'''

from __future__ import print_function

import tensorflow as tf
import sys
import time

# cluster specification
parameter_servers = ["pc-01:2222"]
workers = [ "pc-02:2222", 
      "pc-03:2222",
      "pc-04:2222"]
cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

# start a server for a specific task
server = tf.train.Server(cluster, 
                          job_name=FLAGS.job_name,
                          task_index=FLAGS.task_index)

# config
batch_size = 100
learning_rate = 0.001
training_epochs = 20
logs_path = "/tmp/mnist/1"

# load mnist data set
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

if FLAGS.job_name == "ps":
  server.join()
elif FLAGS.job_name == "worker":

  # Between-graph replication
  with tf.device(tf.train.replica_device_setter(
    worker_device="/job:worker/task:%d" % FLAGS.task_index,
    cluster=cluster)):

    # count the number of updates
    global_step = tf.get_variable('global_step', [], 
                                initializer = tf.constant_initializer(0), 
                                trainable = False)

    # input images
    with tf.name_scope('input'):
      # None -> batch size can be any size, 784 -> flattened mnist image
      x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
      # target 10 output classes
      y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")

    # model parameters will change during training so we use tf.Variable
    tf.set_random_seed(1)
    with tf.name_scope("weights"):
      W1 = tf.Variable(tf.random_normal([784, 100]))
      W2 = tf.Variable(tf.random_normal([100, 10]))

    # bias
    with tf.name_scope("biases"):
      b1 = tf.Variable(tf.zeros([100]))
      b2 = tf.Variable(tf.zeros([10]))

    # implement model
    with tf.name_scope("softmax"):
      # y is our prediction
      z2 = tf.add(tf.matmul(x,W1),b1)
      a2 = tf.nn.sigmoid(z2)
      z3 = tf.add(tf.matmul(a2,W2),b2)
      y  = tf.nn.softmax(z3)

    # specify cost function
    with tf.name_scope('cross_entropy'):
      # this is our cost
      cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    # specify optimizer
    with tf.name_scope('train'):
      # optimizer is an "operation" which we can execute in a session
      grad_op = tf.train.GradientDescentOptimizer(learning_rate)
      '''
      rep_op = tf.train.SyncReplicasOptimizer(grad_op, 
                                          replicas_to_aggregate=len(workers),
                                          replica_id=FLAGS.task_index, 
                                          total_num_replicas=len(workers),
                                          use_locking=True
                                          )
      train_op = rep_op.minimize(cross_entropy, global_step=global_step)
      '''
      train_op = grad_op.minimize(cross_entropy, global_step=global_step)
      
    '''
    init_token_op = rep_op.get_init_tokens_op()
    chief_queue_runner = rep_op.get_chief_queue_runner()
    '''

    with tf.name_scope('Accuracy'):
      # accuracy
      correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # create a summary for our cost and accuracy
    tf.scalar_summary("cost", cross_entropy)
    tf.scalar_summary("accuracy", accuracy)

    # merge all summaries into a single "operation" which we can execute in a session 
    summary_op = tf.merge_all_summaries()
    init_op = tf.initialize_all_variables()
    print("Variables initialized ...")

  sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                            global_step=global_step,
                            init_op=init_op)

  begin_time = time.time()
  frequency = 100
  with sv.prepare_or_wait_for_session(server.target) as sess:
    '''
    # is chief
    if FLAGS.task_index == 0:
      sv.start_queue_runners(sess, [chief_queue_runner])
      sess.run(init_token_op)
    '''
    # create log writer object (this will log on every machine)
    writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
        
    # perform training cycles
    start_time = time.time()
    for epoch in range(training_epochs):

      # number of batches in one epoch
      batch_count = int(mnist.train.num_examples/batch_size)

      count = 0
      for i in range(batch_count):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        
        # perform the operations we defined earlier on batch
        _, cost, summary, step = sess.run(
                        [train_op, cross_entropy, summary_op, global_step], 
                        feed_dict={x: batch_x, y_: batch_y})
        writer.add_summary(summary, step)

        count += 1
        if count % frequency == 0 or i+1 == batch_count:
          elapsed_time = time.time() - start_time
          start_time = time.time()
          print("Step: %d," % (step+1), 
                " Epoch: %2d," % (epoch+1), 
                " Batch: %3d of %3d," % (i+1, batch_count), 
                " Cost: %.4f," % cost, 
                " AvgTime: %3.2fms" % float(elapsed_time*1000/frequency))
          count = 0

    print("Test-Accuracy: %2.2f" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    print("Total Time: %3.2fs" % float(time.time() - begin_time))
    print("Final Cost: %.4f" % cost)

  sv.stop()
  print("done")
```

看我写的辛苦求打赏啊！！！有学术讨论和指点请加微信manutdzou,注明

![20](/public/img/pay.jpg)
