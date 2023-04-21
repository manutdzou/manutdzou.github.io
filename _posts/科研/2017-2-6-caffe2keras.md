---
layout: post
title: caffe model convert to keras model
category: 科研
tags: 深度学习
keywords: 应用
description: 
---

# Caffe End: Dumping the Model

```
from __future__ import division, print_function
import caffe
import numpy as np
import os

DATA_DIR = "/path/to/my/data"
OUTPUT_DIR = os.path.join(DATA_DIR, "vgg-cnn-weights")

CAFFE_HOME="/path/to/my/caffe/installation"

MODEL_DIR = os.path.join(CAFFE_HOME, "models", "vgg_cnn_s")
MODEL_PROTO = os.path.join(MODEL_DIR, "deploy.prototxt")
MODEL_WEIGHTS = os.path.join(MODEL_DIR, "VGG_CNN_S.caffemodel")
MEAN_IMAGE = os.path.join(MODEL_DIR, "VGG_mean.binaryproto")

caffe.set_mode_cpu()
net = caffe.Net(MODEL_PROTO, MODEL_WEIGHTS, caffe.TEST)

# layer names and output shapes
for layer_name, blob in net.blobs.iteritems():
    print(layer_name, blob.data.shape)

# write out weight matrices and bias vectors
for k, v in net.params.items():
    print(k, v[0].data.shape, v[1].data.shape)
    np.save(os.path.join(OUTPUT_DIR, "W_{:s}.npy".format(k)), v[0].data)
    np.save(os.path.join(OUTPUT_DIR, "b_{:s}.npy".format(k)), v[1].data)

# write out mean image
blob = caffe.proto.caffe_pb2.BlobProto()
with open(MEAN_IMAGE, 'rb') as fmean:
    mean_data = fmean.read()
blob.ParseFromString(mean_data)
mu = np.array(caffe.io.blobproto_to_array(blob))
print("Mean image:", mu.shape)
np.save(os.path.join(OUTPUT_DIR, "mean_image.npy"), mu)
```

# Keras End: Rebuilding the Model and classify

```
from __future__ import division, print_function
from keras import backend as K
from keras.layers import Input
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.convolutional import Convolution2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from scipy.misc import imresize
import matplotlib.pyplot as plt
import numpy as np
import os
import re

K.set_image_dim_ordering("th") # set the dimension ordering to Theano in Keras

# Defining the Local Response Normalization Layer
from keras.engine.topology import Layer
class LRN(Layer):
    def __init__(self, n=5, alpha=0.0005, beta=0.75, k=2, **kwargs):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.k = k
        super(LRN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.shape = input_shape
        super(LRN, self).build(input_shape)

    def call(self, x, mask=None):
        if K.image_dim_ordering == "th":
            _, f, r, c = self.shape
        else:
            _, r, c, f = self.shape
        half_n = self.n // 2
        squared = K.square(x)

        pooled = K.pool2d(squared, (half_n, half_n), strides=(1, 1),
                         border_mode="same", pool_mode="avg")
        if K.image_dim_ordering == "th":
            summed = K.sum(pooled, axis=1, keepdims=True)
            averaged = (self.alpha / self.n) * K.repeat_elements(summed, f, axis=1)
        else:
            summed = K.sum(pooled, axis=3, keepdims=True)
            averaged = (self.alpha / self.n) * K.repeat_elements(summed, f, axis=3)
        denom = K.pow(self.k + averaged, self.beta)
        return x / denom
    
    def get_output_shape_for(self, input_shape):
        return input_shape

def transform_conv_weight(W):
    # for non FC layers, do this because Keras does convolution vs Caffe correlation

# I'm not sure if it needs rot, but when without rot it can output accurate result, while rot it gets wrong

    #for i in range(W.shape[0]):
    #    for j in range(W.shape[1]):
    #        W[i, j] = np.rot90(W[i, j],2)
    return W

def transform_fc_weight(W):
    return W.T

CAFFE_WEIGHTS_DIR = "/home/dlg/caffe2keras/data/vgg-cnn-weights"

W_conv1 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv1.npy")))
b_conv1 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv1.npy"))

W_conv2 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv2.npy")))
b_conv2 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv2.npy"))

W_conv3 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv3.npy")))
b_conv3 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv3.npy"))

W_conv4 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv4.npy")))
b_conv4 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv4.npy"))

W_conv5 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv5.npy")))
b_conv5 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv5.npy"))

W_fc6 = transform_fc_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_fc6.npy")))
b_fc6 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_fc6.npy"))

W_fc7 = transform_fc_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_fc7.npy")))
b_fc7 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_fc7.npy"))

W_fc8 = transform_fc_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_fc8.npy")))
b_fc8 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_fc8.npy"))

data = Input(shape=(3, 224, 224), name="DATA")

conv1 = Convolution2D(96, 7, 7, subsample=(2, 2),
                     weights=(W_conv1, b_conv1))(data)
conv1 = Activation("relu", name="CONV1")(conv1)

norm1 = LRN(name="NORM1")(conv1)

# caffe中在pooling时隐式调用了pad操作，需要在keras中补上
pool1 = ZeroPadding2D(padding=(2,2))(norm1)
pool1 = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name="POOL1")(pool1)

conv2 = Convolution2D(256, 5, 5, weights=(W_conv2, b_conv2))(pool1)
conv2 = Activation("relu", name="CONV2")(conv2)

pool2 = ZeroPadding2D(padding=(1, 1))(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="POOL2")(pool2)

conv3 = ZeroPadding2D(padding=(1, 1))(pool2)
conv3 = Convolution2D(512, 3, 3, weights=(W_conv3, b_conv3))(conv3)
conv3 = Activation("relu", name="CONV3")(conv3)

conv4 = ZeroPadding2D(padding=(1, 1))(conv3)
conv4 = Convolution2D(512, 3, 3, weights=(W_conv4, b_conv4))(conv4)
conv4 = Activation("relu", name="CONV4")(conv4)

conv5 = ZeroPadding2D(padding=(1, 1))(conv4)
conv5 = Convolution2D(512, 3, 3, weights=(W_conv5, b_conv5))(conv5)
conv5 = Activation("relu", name="CONV5")(conv5)

pool5 = ZeroPadding2D(padding=(1, 1))(conv5)
pool5 = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name="POOL5")(pool5)

fc6 = Flatten()(pool5)
fc6 = Dense(4096, weights=(W_fc6, b_fc6))(fc6)
fc6 = Activation("relu", name="FC6")(fc6)

fc7 = Dense(4096, weights=(W_fc7, b_fc7))(fc6)
fc7 = Activation("relu", name="FC7")(fc7)

fc8 = Dense(1000, weights=(W_fc8, b_fc8), name="FC8")(fc7)
prob = Activation("softmax", name="PROB")(fc8)

model = Model(input=[data], output=[prob])

model.compile(optimizer="adam", loss="categorical_crossentropy")

id2label = {}
flabel = open("/home/dlg/caffe2keras/synset_words.txt", "rb")
for (id,line) in enumerate(flabel):
    id2label[id] = line
flabel.close()

def preprocess_image(img, resize_wh, mean_image):
    # resize
    img4d = imresize(img, (resize_wh, resize_wh))
    img4d = img4d.astype("float32")
    # BGR -> RGB
    img4d = img4d[:, :, ::-1]
    # swap axes to theano mode
    img4d = np.transpose(img4d, (2, 0, 1))
    # add batch dimension
    img4d = np.expand_dims(img4d, axis=0)
    # subtract mean image
    img4d -= mean_image
    # clip to uint
    img4d = np.clip(img4d, 0, 255).astype("uint8")
    return img4d

CAT_IMAGE = "/home/dlg/caffe2keras/cat.jpg"
MEAN_IMAGE = "/home/dlg/caffe2keras/data/vgg-cnn-weights/mean_image.npy"
RESIZE_WH = 224

mean_image = np.load(MEAN_IMAGE)
image = plt.imread(CAT_IMAGE)
img4d = preprocess_image(image, RESIZE_WH, mean_image)

print(image.shape, mean_image.shape, img4d.shape)
plt.imshow(image)

preds = model.predict(img4d)[0]
print(np.argmax(preds))

top_preds = np.argsort(preds)[::-1][0:10]
print(top_preds)

pred_probas = [(x, id2label[x], preds[x]) for x in top_preds]
print(pred_probas)

```

Result

```
[285 282 281 723 326 277 287 835 626 322]
[(285, 'n02124075 Egyptian cat\n', 0.073803827), (282, 'n02123159 tiger cat\n', 0.047371995), (281, 'n02123045 tabby, tabby cat\n', 0.039496809), (723, 'n03944341 pinwheel\n', 0.037778851), (326, 'n02281787 lycaenid, lycaenid butterfly\n', 0.028855985), (277, 'n02119022 red fox, Vulpes vulpes\n', 0.01975202), (287, 'n02127052 lynx, catamount\n', 0.019132387), (835, 'n04355338 sundial\n', 0.017088085), (626, 'n03666591 lighter, light, igniter, ignitor\n', 0.013068036), (322, 'n02277742 ringlet, ringlet butterfly\n', 0.012498086)]
```

同时在caffe中调用C++的API测试caffe模型

```
./build/examples/cpp_classification/classification.bin /home/dlg/caffe2keras/caffe_model/models/vgg_cnn_s/deploy.prototxt /home/dlg/caffe2keras/caffe_model/models/vgg_cnn_s/VGG_CNN_S.caffemodel /home/dlg/caffe2keras/caffe_model/models/vgg_cnn_s/VGG_mean.binaryproto /home/dlg/caffe2keras/synset_words.txt /home/dlg/caffe2keras//cat.jpg

---------- Prediction for /home/dlg/caffe2keras//cat.jpg ----------
0.2311 - "n02119789 kit fox, Vulpes macrotis"
0.1623 - "n02123045 tabby, tabby cat"
0.1329 - "n02119022 red fox, Vulpes vulpes"
0.1257 - "n02124075 Egyptian cat"
0.1017 - "n02123159 tiger cat"
```



参考http://sujitpal.blogspot.jp/2017/01/migrating-vgg-cnn-from-caffe-to-keras.html，修改了部分错误

看我写的辛苦求打赏啊！！！有学术讨论和指点请加微信manutdzou,注明

![20](/public/img/pay.jpg)