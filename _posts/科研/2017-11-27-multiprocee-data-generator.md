---
layout: post
title: 利用多线程读取数据加快网络训练
category: 科研
tags: 深度学习
keywords: multi-process
description:
---

# 利用多线程生成数据

## 当CPU读取数据跟不上GPU处理数据速度时候可以考虑这种方式，这种方法的好处是数据接口简单而且可以大幅加快网络训练时间。特别是针对服务器端超多核CPU配置。实验证实，该方法可以大大提升GPU的利用率。

### 定义一个队列的类

```
"""
this file is modified from keras implemention of data process multi-threading,
see https://github.com/fchollet/keras/blob/master/keras/utils/data_utils.py
"""
import time
import numpy as np
import threading
import multiprocessing
try:
    import queue
except ImportError:
    import Queue as queue


class GeneratorEnqueuer():
    """
    Builds a queue out of a data generator.
    Used in `fit_generator`, `evaluate_generator`, `predict_generator`.
    # Arguments
        generator: a generator function which endlessly yields data
        use_multiprocessing: use multiprocessing if True, otherwise threading
        wait_time: time to sleep in-between calls to `put()`
        random_seed: Initial seed for workers,
            will be incremented by one for each workers.
    """

    def __init__(self, generator,
                 use_multiprocessing=False,
                 wait_time=0.05,
                 random_seed=None):
        self.wait_time = wait_time
        self._generator = generator
        self._use_multiprocessing = use_multiprocessing
        self._threads = []
        self._stop_event = None
        self.queue = None
        self.random_seed = random_seed

    def start(self, workers=1, max_queue_size=10):
        """Kicks off threads which add data from the generator into the queue.
        # Arguments
            workers: number of worker threads
            max_queue_size: queue size
                (when full, threads could block on `put()`)
        """

        def data_generator_task():
            while not self._stop_event.is_set():
                try:
                    if self._use_multiprocessing or self.queue.qsize() < max_queue_size:
                        generator_output = next(self._generator)
                        self.queue.put(generator_output)
                    else:
                        time.sleep(self.wait_time)
                except Exception:
                    self._stop_event.set()
                    raise

        try:
            if self._use_multiprocessing:
                self.queue = multiprocessing.Queue(maxsize=max_queue_size)
                self._stop_event = multiprocessing.Event()
            else:
                self.queue = queue.Queue()
                self._stop_event = threading.Event()

            for _ in range(workers):
                if self._use_multiprocessing:
                    # Reset random seed else all children processes
                    # share the same seed
                    np.random.seed(self.random_seed)
                    thread = multiprocessing.Process(target=data_generator_task)
                    thread.daemon = True
                    if self.random_seed is not None:
                        self.random_seed += 1
                else:
                    thread = threading.Thread(target=data_generator_task)
                self._threads.append(thread)
                thread.start()
        except:
            self.stop()
            raise

    def is_running(self):
        return self._stop_event is not None and not self._stop_event.is_set()

    def stop(self, timeout=None):
        """Stops running threads and wait for them to exit, if necessary.
        Should be called by the same thread which called `start()`.
        # Arguments
            timeout: maximum time to wait on `thread.join()`.
        """
        if self.is_running():
            self._stop_event.set()

        for thread in self._threads:
            if thread.is_alive():
                if self._use_multiprocessing:
                    thread.terminate()
                else:
                    thread.join(timeout)

        if self._use_multiprocessing:
            if self.queue is not None:
                self.queue.close()

        self._threads = []
        self._stop_event = None
        self.queue = None

    def get(self):
        """Creates a generator to extract data from the queue.
        Skip the data if it is `None`.
        # Returns
            A generator
        """
        while self.is_running():
            if not self.queue.empty():
                inputs = self.queue.get()
                if inputs is not None:
                    yield inputs
            else:
                time.sleep(self.wait_time)
```

### 然后定义一个data_provide函数用于生成data和label

```
def data_provide():
    ...
    return data, label
```

### batch生成器

```
def generator(batch_size=32):
    images = []
    labels = []
    while True:
        try:
            im, label = data_provide()
            images.append(im)
            labels.append(label)

            if len(images) == batch_size:
                yield images, labels
                images = []
                labels = []
        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()
            continue

def get_batch(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        enqueuer.start(max_queue_size=64, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()

if __name__ == '__main__':
    gen = get_batch(num_workers=64,batch_size=128)
    while 1:
        start = time.time()
        images, labels =  next(gen)
        end = time.time()
        print end-start
        print(len(images)," ",images[0].shape)
        print(len(labels)," ",labels[0].shape)
```

## tensorflow1.3+自带的dataset API

```
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import random

import tensorflow as tf

class DataReader(object):

    def __init__(self, num_class = 2, batch_size = 128, epoch = 10, file_list = None):
        self.num_class = num_class
        self.train_data_list = file_list
        self.batch_size = batch_size
        self.epoch = epoch
        self.dataset_iterator = None
        self._init_dataset()

    def _get_filename_list(self):
        lines = open(self.train_data_list,'r').read().splitlines()
        random.shuffle(lines)
        filename_list = []
        label_list = []
        for i in range(len(lines)):
            filename_list.append(lines[i].split(' ')[0])
            label_list.append(int(lines[i].split(' ')[1]))
        return tf.constant(filename_list), tf.constant(label_list) #must be tensor type

    def _parse_function(self, filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_image(image_string)
        channel_mean = tf.constant(np.array([123.68,116.779,103.938], dtype=np.float32))
        image = tf.subtract(tf.cast(image_decoded, dtype=tf.float32),channel_mean)
        image_label = tf.one_hot(indices = label, depth = self.num_class)
        return tf.reshape(image,[256,256,3]), tf.reshape(image_label,[self.num_class])

    def _init_dataset(self):
        # Read sample files
        sample_files, sample_labels = self._get_filename_list()
        # make dataset
        dataset = tf.data.Dataset.from_tensor_slices((sample_files, sample_labels))
        dataset = dataset.map(self._parse_function)
        dataset = dataset.shuffle(1000).batch(self.batch_size).repeat(self.epoch)
        self.dataset_iterator = dataset.make_one_shot_iterator()

    def inputs(self):
        return self.dataset_iterator.get_next()

if __name__ == '__main__':
    data_generator = DataReader(file_list = "train_list.txt")
	"""
	train_list.txt
	
	path/image1.jpg label_1
	path/image2.jpg label_2
	path/image3.jpg label_3
	path/image4.jpg label_4	
	"""
    sess = tf.Session()
    x,y = sess.run(data_generator.inputs())
```

# 如果需要用到python处理数据或有特殊需求可以使用如下接口,完美兼容python numpy

```
    def _read_py_function(self, filename, label):
        image_decoded = py_function(filename)
		label = py_function(label)
        return image_decoded, label


    def _init_dataset(self):

        # Read sample files
        sample_files, sample_labels = self._get_filename_list()

        # make dataset
        dataset = tf.data.Dataset.from_tensor_slices((sample_files, sample_labels))

        dataset = dataset.map(
            lambda filename, label: tuple(tf.py_func(
                self._read_py_function, [filename, label], [tf.int32, label.dtype])))

        self.dataset_iterator = dataset.make_one_shot_iterator()

```

# 利用python的multiprocess包多线程处理生成数据

```
import numpy as np
import time
from multiprocessing import Pool, cpu_count

def batch_works(k):
    #将任务集合拆分到每个cpu核中
    if k == n_processes - 1:
        nums = jobs[k * int(len(jobs) / n_processes) : ]
    else:
        nums = jobs[k * int(len(jobs) / n_processes) : (k + 1) * int(len(jobs) / n_processes)]
    for j in nums:
        py_function()

if __name__ == '__main__':
    jobs = range(100) #所需执行的任务序列
    n_processes = cpu_count()

    pool = Pool(processes=n_processes)
    start = time.time()
    pool.map(batch_works, range(n_processes))
    pool.close()
    pool.join()
    end = time.time()
    print end -start

```

看我写的辛苦求打赏啊！！！有学术讨论和指点请加微信manutdzou,注明

![20](/public/img/pay.jpg)