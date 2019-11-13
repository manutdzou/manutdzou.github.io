---
layout: post
title: tensorflow python deploy
category: 科研
tags: 
keywords: 
description:
---

训练好了模型需要部署，代码打包加密等

首先将训练好的模型AES加密,记下加密Key

```
import base64
import hashlib
import os
import sys
import string
import random

try:
    from Crypto import Random
    from Crypto.Cipher import AES
except:
    raise Exception('Install Crypto! \n pip install pycrypto ')
try:
    import tensorflow as tf
except:
    raise Exception('Install Tensorflow!')


class AESCipher(object):
    
    def __init__(self, _key):
        self.bs = 32
        self.key = hashlib.sha256(_key.encode()).digest()
    
    def encrypt(self, raw):
        raw = self._pad(raw)
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
#        return base64.b64encode(iv + cipher.encrypt(raw))
        return (iv + cipher.encrypt(raw))

    
    def decrypt(self, enc):
#        enc = base64.b64decode(enc)
        iv = enc[:AES.block_size]
        print('Iv: %s' % iv)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return self._unpad(cipher.decrypt(enc[AES.block_size:]))#.decode('utf-8')
    
    def _pad(self, s):
        #python2.7
        #return s + (self.bs - len(s) % self.bs) * chr(self.bs - len(s) % self.bs)
        #python3.6
        return s + str.encode((self.bs - len(s) % self.bs) * chr(self.bs - len(s) % self.bs))
    
    @staticmethod
    def _unpad(s):
        return s[:-ord(s[len(s)-1:])]

###############  Util Methods ###############

def load_graph(path):
    with tf.gfile.GFile(path, 'rb') as f:
        if not tf.gfile.Exists(path):
            raise Exception('File doesn\'t exist at path: %s' % path)
        
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        f.close()
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name=None)
        return graph_def

def generate_output_path(input_path, suffix):
    filename, file_extension = os.path.splitext(input_path)
    return filename + suffix + file_extension

def random_string(size=30, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(size))

def read_arg(index, default=None, err_msg=None):
    def print_error():
        if err_msg is not None:
            raise Exception(err_msg)
        else:
            raise Exception('Not found arg with index %s' % index)
    if len(sys.argv) <= index:
        if default is not None:
            return default
        print_error()
    return sys.argv[index]

#############################################

def main():
    USAGE = 'python encrypt_model.py <INPUT_PB_MODEL> <OUTPUT_PB_MODEL> <KEY>'
    print('\nUSAGE: %s\n' % USAGE)

    # Args:

    INPUT_PATH      = read_arg(1, default='yolov3_coco.pb')
    default_out     = generate_output_path(INPUT_PATH, '-encrypted')
    OUTPUT_PATH     = read_arg(2, default=default_out)
    KEY             = read_arg(3, default='CCCL88QM8IPZDAJDBD6Y2816V0CQQQ')
    #KEY             = read_arg(3, default=random_string())




    graph_def = load_graph(INPUT_PATH)

    cipher = AESCipher(KEY)

    nodes_binary_str = graph_def.SerializeToString()


    nodes_binary_str = cipher.encrypt(nodes_binary_str)

    with tf.gfile.GFile(OUTPUT_PATH, 'wb') as f:
        f.write(nodes_binary_str);
        f.close()
    print('Saved with key="%s" to %s' % (KEY, OUTPUT_PATH))


if __name__ == "__main__":
    main()

```


解密模型，封装inference代码，暴露api接口

yolov3_detector.py如下

```
import cv2
import numpy as np
import tensorflow as tf

import base64
import hashlib
import os
import sys
import string

from Crypto import Random
from Crypto.Cipher import AES

class AESCipher(object):

    def __init__(self, _key):
        self.bs = 32
        self.key = hashlib.sha256(_key.encode()).digest()

    def decrypt(self, enc):
        iv = enc[:AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return self._unpad(cipher.decrypt(enc[AES.block_size:]))  # .decode('utf-8')

    def _pad(self, s):
        #python3.6
        return s + str.encode((self.bs - len(s) % self.bs) * chr(self.bs - len(s) % self.bs))

    @staticmethod
    def _unpad(s):
        return s[:-ord(s[len(s) - 1:])]

class Yolov3_Detector(object):
    def __init__(self, input_size, num_classes, pb):
        # config
        return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
        pb_file = pb
        self.num_classes = num_classes
        self.input_size = input_size
        self.key = 'CCCL88QM8IPZDAJDBD6Y2816V0CQQQ'
        self.graph = tf.Graph()
        self.return_tensors = self.decrypt_read_pb_return_tensors(self.graph, pb_file, return_elements)
        self.sess = tf.Session(graph=self.graph)

    def decrypt_read_pb_return_tensors(self, graph, pb_file, return_elements):
        with tf.gfile.FastGFile(pb_file, 'rb') as f:
            nodes_binary_str = f.read()
        cipher = AESCipher(self.key)
        nodes_str_decrypt = cipher.decrypt(nodes_binary_str)        
        
        frozen_graph_def = tf.GraphDef()
        frozen_graph_def.ParseFromString(nodes_str_decrypt)

        with graph.as_default():
            return_elements = tf.import_graph_def(frozen_graph_def, return_elements=return_elements)
        return return_elements

    def read_pb_return_tensors(self, graph, pb_file, return_elements):
        with tf.gfile.FastGFile(pb_file, 'rb') as f:
            frozen_graph_def = tf.GraphDef()
            frozen_graph_def.ParseFromString(f.read())

        with graph.as_default():
            return_elements = tf.import_graph_def(frozen_graph_def, return_elements=return_elements)

        return return_elements

    def image_preporcess(self, image, target_size, gt_boxes=None):
   
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        ih, iw    = target_size
        h,  w, _  = image.shape

        scale = min(iw/float(w), ih/float(h))
        nw, nh  = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
        dw, dh = (iw - nw) // 2, (ih-nh) // 2
        image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
        image_paded = image_paded / 255.

        if gt_boxes is None:
            return image_paded

        else:
            gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
            gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
            return image_paded, gt_boxes

    def nms(self, bboxes, iou_threshold, sigma=0.3, method='nms'):
        """
        :param bboxes: (xmin, ymin, xmax, ymax, score, class, feature)
        Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
              https://github.com/bharatsingh430/soft-nms
        """
        classes_in_img = list(set(bboxes[:, 5]))
        best_bboxes = []

        for cls in classes_in_img:
            cls_mask = (bboxes[:, 5] == cls)
            cls_bboxes = bboxes[cls_mask]

            while len(cls_bboxes) > 0:
                max_ind = np.argmax(cls_bboxes[:, 4])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
                iou = self.bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
                weight = np.ones((len(iou),), dtype=np.float32)

                assert method in ['nms', 'soft-nms']

                if method == 'nms':
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0

                if method == 'soft-nms':
                    weight = np.exp(-(1.0 * iou ** 2 / sigma))

                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
                score_mask = cls_bboxes[:, 4] > 0.
                cls_bboxes = cls_bboxes[score_mask]

        return np.array(best_bboxes)

    def postprocess_boxes(self, pred_bbox, org_img_shape, input_size, score_threshold):

        valid_scale=[0, np.inf]
        pred_bbox = np.array(pred_bbox)

        pred_xywh = pred_bbox[:, 0:4]
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:5+self.num_classes]

        # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
        pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                    pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
        # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
        org_h, org_w = org_img_shape
        resize_ratio = min(input_size[1] / float(org_w), input_size[0] / float(org_h))

        dw = (input_size[1] - resize_ratio * org_w) / 2.0
        dh = (input_size[0] - resize_ratio * org_h) / 2.0

        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        # # (3) clip some boxes those are out of range
        pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                    np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
        invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
        pred_coor[invalid_mask] = 0

        # # (4) discard some invalid boxes
        bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
        scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

        # # (5) discard some boxes with low scores
        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > score_threshold
        mask = np.logical_and(scale_mask, score_mask)
        coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

        return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

    def bboxes_iou(self, boxes1, boxes2):
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area    = inter_section[..., 0] * inter_section[..., 1]
        union_area    = boxes1_area + boxes2_area - inter_area
        ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

        return ious
    
    # inference
    def inference(self, input):
        original_image_size = input.shape[:2]
        image_data = self.image_preporcess(np.copy(input), self.input_size)
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run(
            [self.return_tensors[1], self.return_tensors[2], self.return_tensors[3]],
            feed_dict={self.return_tensors[0]: image_data})

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)

        bboxes = self.postprocess_boxes(pred_bbox, original_image_size, self.input_size, 0.3)
        bboxes = self.nms(bboxes, 0.45, method='nms')

        return bboxes
```

detection_api.py如下

```
import yolov3_detector
import cv2
import numpy as np

class Detector(object):
    def __init__(self,  input_size = [768,768], num_classes = 1, pb = "yolov3_coco_person-encrypted.pb"):
        self.model = yolov3_detector.Yolov3_Detector(input_size, num_classes,pb)
        
    def draw_bbox(self, image, bboxes, classes=['person'], show_label=True):
        """
        bboxes: [x_min, y_min, x_max, y_max, tracking_id] format coordinates.
        """

        image_h, image_w, _ = image.shape

        for i, bbox in enumerate(bboxes):
            coor = np.array(bbox[:4], dtype=np.int32)
            fontScale = 0.5
            ids = int(bbox[4])
            bbox_color = (0,0,255)
            bbox_thick = int(0.6 * (image_h + image_w) / 600.0)
            c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
            cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

            if show_label:
                bbox_mess = '%s' % (classes[ids])
                t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
                cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled
                cv2.putText(image, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)
        return image
```

使用script将上述两段源码打包成.so文件, python setup.py folder将把文件夹内所有源码编译成.so的文件

```
#-* -coding: UTF-8 -* -
__author__ = 'Arvin'

"""
执行前提：
    系统安装python-devel 和 gcc
    Python安装cython

编译整个当前目录：
    python py-setup.py
编译某个文件夹：
    python py-setup.py BigoModel

生成结果：
    目录 build 下

生成完成后：
    启动文件还需要py/pyc担当，须将启动的py/pyc拷贝到编译目录并删除so文件

"""

import sys, os, shutil, time
from distutils.core import setup
from Cython.Build import cythonize

starttime = time.time()
currdir = os.path.abspath('.')
parentpath = sys.argv[1] if len(sys.argv)>1 else ""
setupfile= os.path.join(os.path.abspath('.'), __file__)
build_dir = "build"
build_tmp_dir = build_dir + "/temp"

def getpy(basepath=os.path.abspath('.'), parentpath='', name='', excepts=(), copyOther=False,delC=False):
    """
    获取py文件的路径
    :param basepath: 根路径
    :param parentpath: 父路径
    :param name: 文件/夹
    :param excepts: 排除文件
    :param copy: 是否copy其他文件
    :return: py文件的迭代器
    """
    fullpath = os.path.join(basepath, parentpath, name)
    for fname in os.listdir(fullpath):
        ffile = os.path.join(fullpath, fname)
        #print basepath, parentpath, name,file
        if os.path.isdir(ffile) and fname != build_dir and not fname.startswith('.'):
            for f in getpy(basepath, os.path.join(parentpath, name), fname, excepts, copyOther, delC):
                yield f
        elif os.path.isfile(ffile):
            ext = os.path.splitext(fname)[1]
            if ext == ".c":
                if delC and os.stat(ffile).st_mtime > starttime:
                    os.remove(ffile)
            elif ffile not in excepts and os.path.splitext(fname)[1] not in('.pyc', '.pyx'):
                if os.path.splitext(fname)[1] in('.py', '.pyx') and not fname.startswith('__'):
                    yield os.path.join(parentpath, name, fname)
                elif copyOther:
                        dstdir = os.path.join(basepath, build_dir, parentpath, name)
                        if not os.path.isdir(dstdir): os.makedirs(dstdir)
                        shutil.copyfile(ffile, os.path.join(dstdir, fname))
        else:
            pass

#获取py列表
module_list = list(getpy(basepath=currdir,parentpath=parentpath, excepts=(setupfile)))
try:
    setup(ext_modules = cythonize(module_list),script_args=["build_ext", "-b", build_dir, "-t", build_tmp_dir])
except Exception as ex:
    print("error! ", ex.message)
else:
    module_list = list(getpy(basepath=currdir, parentpath=parentpath, excepts=(setupfile), copyOther=True))

module_list = list(getpy(basepath=currdir, parentpath=parentpath, excepts=(setupfile), delC=True))
if os.path.exists(build_tmp_dir): shutil.rmtree(build_tmp_dir)

print("complate! time:", time.time()-starttime, 's')
```

最后生成yolov3_detector.cpython-36m-x86_64-linux-gnu.so和detection_api.cpython-36m-x86_64-linux-gnu.so文件可以直接调用

```
from detection_api import Detector
import cv2

if __name__ == '__main__':
    det = Detector()
    img = cv2.imread('test.jpg')
    boxes = det.model.inference(img)
    img = det.draw_bbox(img, boxes)
    cv2.imwrite('result.jpg', img)
```

看我写的辛苦求打赏啊！！！有学术讨论和指点请加微信manutdzou,注明

![20](/public/img/pay.jpg)