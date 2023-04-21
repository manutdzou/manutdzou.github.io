---
layout: post
title: 分析voc2007检测数据
category: 科研
tags: 深度学习
keywords: source code
description: 
---

# 分析voc2007检测数据在faster-rcnn上的rpn anchor的recall和proposal的recall

科研过程是一个漫长而谨慎的环节，可谓得细节者得天下。所以细节分析成为科研领域一个很重要的突破口。下面针对faster-rcnn在voc2007检测数据上的表现分析一下算法的瓶颈。需要计算初始化anchor的recall,RPN网络proposal的recall

首先需要存储训练数据训练过程中的rpn的acnhors和训练过程中的生成的proposals，这些数据在开源程序中都有保存
其次需要保存测试过程中RPN网络生成的proposals，这在源码中并未提供需要自己部分修改保存

# 保存test的proposal,将以下修改代码替换源码

```Python

"""Test a Fast R-CNN network on an imdb (image database)."""

from fast_rcnn.config import cfg, get_output_dir
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from fast_rcnn.nms_wrapper import nms
import cPickle
import heapq
from utils.blob import im_list_to_blob
import os

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    if not cfg.TEST.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors

def im_detect(net, im, boxes=None):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    else:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    else:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)

    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scales[0]

    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['cls_score'].data
    else:
        # use softmax estimated probabilities
        scores = blobs_out['cls_prob']

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    return scores, pred_boxes,boxes #boxes is proposals，这里将test的proposal一起返回

def vis_detections(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()

def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            keep = nms(dets, thresh)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes

def test_net(net, imdb):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # heuristic: keep an average of 40 detections per class per images prior
    # to NMS
    max_per_set = 40 * num_images
    # heuristic: keep at most 100 detection per class per image prior to NMS
    max_per_image = 100
    # detection thresold for each class (this is adaptively set based on the
    # max_per_set constraint)
    thresh = -np.inf * np.ones(imdb.num_classes)
    # top_scores will hold one minheap of scores per class (used to enforce
    # the max_per_set constraint)
    top_scores = [[] for _ in xrange(imdb.num_classes)]
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, net)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    if not cfg.TEST.HAS_RPN:
        roidb = imdb.roidb

    proposals = [[] for _ in xrange(num_images)] #开辟一个存储空间
    for i in xrange(num_images):
        # filter out any ground truth boxes
        if cfg.TEST.HAS_RPN:
            box_proposals = None
        else:
            box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]
        im = cv2.imread(imdb.image_path_at(i))
        _t['im_detect'].tic()
        scores, boxes, proposals[i] = im_detect(net, im, box_proposals) #将proposal保存在proposals
        _t['im_detect'].toc()
        _t['misc'].tic()
        for j in xrange(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh[j])[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            top_inds = np.argsort(-cls_scores)[:max_per_image]
            cls_scores = cls_scores[top_inds]
            cls_boxes = cls_boxes[top_inds, :]
            # push new scores onto the minheap
            for val in cls_scores:
                heapq.heappush(top_scores[j], val)
            # if we've collected more than the max number of detection,
            # then pop items off the minheap and update the class threshold
            if len(top_scores[j]) > max_per_set:
                while len(top_scores[j]) > max_per_set:
                    heapq.heappop(top_scores[j])
                thresh[j] = top_scores[j][0]

            all_boxes[j][i] = \
                    np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                    .astype(np.float32, copy=False)

            if 0:
                keep = nms(all_boxes[j][i], 0.3)
                vis_detections(im, imdb.classes[j], all_boxes[j][i][keep, :])
        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)

    for j in xrange(1, imdb.num_classes):
        for i in xrange(num_images):
            inds = np.where(all_boxes[j][i][:, -1] > thresh[j])[0]
            all_boxes[j][i] = all_boxes[j][i][inds, :]

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    proposal_file = os.path.join(output_dir, 'proposals.pkl') #将proposals通过cPikle包写入指定的文件内
    with open(proposal_file, 'wb') as f:
        cPickle.dump(proposals, f, cPickle.HIGHEST_PROTOCOL)
    print 'Applying NMS to all detections'
    nms_dets = apply_nms(all_boxes, cfg.TEST.NMS)

    print 'Evaluating detections'
    imdb.evaluate_detections(nms_dets, output_dir)
```

# voc2007检测数据分析代码

```Python
# --------------------------------------------------------
# Copyright (c) 2016 RICOH
# Written by Zou Jinyi
# --------------------------------------------------------
import os.path as osp
import sys
import cv2

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)
# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, '..','..', 'lib')
add_path(lib_path)
import numpy as np
import os
from utils.cython_bbox import bbox_overlaps
import xml.dom.minidom as minidom
import cPickle

def load_image_set_index(data_path,image_set):
    """
    Load the indexes listed in this dataset's image set file.
    """
    # Example path to image set file:
    # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
    image_set_file = os.path.join(data_path, 'ImageSets', 'Main',
                                  image_set + '.txt')
    assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
    with open(image_set_file) as f:
        image_index = [x.strip() for x in f.readlines()]
    return image_index

def load_pascal_annotation(data_path,index):
    """
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    """
    classes = ('__background__', # always index 0
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')
    class_to_ind = dict(zip(classes, xrange(len(classes))))

    filename = os.path.join(data_path, 'Annotations', index + '.xml')
    # print 'Loading: {}'.format(filename)
    def get_data_from_tag(node, tag):
        return node.getElementsByTagName(tag)[0].childNodes[0].data

    with open(filename) as f:
        data = minidom.parseString(f.read())

    objs = data.getElementsByTagName('object')
    sizes = data.getElementsByTagName('size')
    if not 0:
        # Exclude the samples labeled as difficult
        non_diff_objs = [obj for obj in objs
                         if int(get_data_from_tag(obj, 'difficult')) == 0]
        if len(non_diff_objs) != len(objs):
            print 'Removed {} difficult objects' \
                .format(len(objs) - len(non_diff_objs))
        objs = non_diff_objs
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)

    for ind,size in enumerate(sizes):
        width=get_data_from_tag(size, 'width')
        height=get_data_from_tag(size, 'height')
        image_size=[int(width),int(height)]

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        # Make pixel indexes 0-based
        x1 = float(get_data_from_tag(obj, 'xmin')) - 1
        y1 = float(get_data_from_tag(obj, 'ymin')) - 1
        x2 = float(get_data_from_tag(obj, 'xmax')) - 1
        y2 = float(get_data_from_tag(obj, 'ymax')) - 1
        cls = class_to_ind[
                str(get_data_from_tag(obj, "name")).lower().strip()]
        boxes[ix, :] = [x1, y1, x2, y2]
        gt_classes[ix] = cls
    return {'boxes' : boxes,
            'gt_classes': gt_classes,
            'flipped' : False,
            'size':image_size}

def append_flipped_images(num_images, gt_roidb):
    widths = [gt_roidb[i]['size'][0]
              for i in xrange(num_images)]
    for i in xrange(num_images):
        boxes = gt_roidb[i]['boxes'].copy()
        image_size=gt_roidb[i]['size']
        oldx1 = boxes[:, 0].copy()
        oldx2 = boxes[:, 2].copy()
        boxes[:, 0] = widths[i] - oldx2 - 1
        boxes[:, 2] = widths[i] - oldx1 - 1
        assert (boxes[:, 2] >= boxes[:, 0]).all()
        entry = {'boxes' : boxes,
                 'gt_classes' : gt_roidb[i]['gt_classes'],
                 'flipped' : True,
                 'size':image_size}
        gt_roidb.append(entry)
    return gt_roidb

def image_path_at(i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return image_path_from_index(image_index[i])

def image_path_from_index(index):
    """
    Construct an image path from the image's "index" identifier.
    """
    image_path = os.path.join(data_path, 'JPEGImages',
                              index + image_ext)
    assert os.path.exists(image_path), \
           'Path does not exist: {}'.format(image_path)
    return image_path

def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2**np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in xrange(ratio_anchors.shape[0])])
    return anchors

def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def scale_and_ratio(min_size,max_size,image_size):
    image_size_min=min(image_size)
    image_size_max=max(image_size)
    im_scale = float(min_size) / float(image_size_min)
    if np.round(im_scale * image_size_max) > max_size:
        im_scale = float(max_size) / float(image_size_max)
    ratio=float(image_size[0])/float(image_size[1])
    return im_scale, ratio

def generate_all_anchors(anchors,feat_stride,num_anchors,conv_width,conv_height,resize_image_size):
    shift_x = np.arange(0, conv_width) * feat_stride
    shift_y = np.arange(0, conv_height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
    A = num_anchors
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))
    allowed_border = 0
    # only keep anchors inside the image
    inds_inside = np.where(
        (all_anchors[:, 0] >=allowed_border) &
        (all_anchors[:, 1] >=allowed_border) &
        (all_anchors[:, 2] < resize_image_size[0] + allowed_border) &  # width
        (all_anchors[:, 3] < resize_image_size[1] + allowed_border)    # height
    )[0]
    anchors = all_anchors[inds_inside, :]
    return anchors


if __name__ == '__main__':
    data_path='/home/bsl/py-faster-rcnn-master/data/VOCdevkit2007/VOC2007'
    proposal_path='/home/bsl/py-faster-rcnn-master/output/faster_rcnn_alt_opt/voc_2007_trainval/vgg16_rpn_stage2_iter_80000_proposals.pkl' #训练过程中存储的proposals
    image_set='trainval'
    image_set_test='test'
    image_ext = '.jpg'
    image_index=load_image_set_index(data_path,image_set)
    gt_roidb = [load_pascal_annotation(data_path,index) for index in image_index]
    num_images=len(gt_roidb)
    gt_roidb = append_flipped_images(num_images, gt_roidb)

    min_size=600
    max_size=1000
    feat_stride=16
    im_scale=np.zeros(len(gt_roidb),dtype=np.float32)
    ratio=np.zeros(len(gt_roidb),dtype=np.float32)
    recall=np.zeros(10,dtype=np.float32)
    anchors = generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2**np.arange(3, 6))
    index=0
    color_gt=(255,0,0)
    color_anchor=(0,255,0)
    ##==============================================================## anchor recall
    for j in np.arange(0.1,1.1,0.1):
        ind_nums=0
        recall_nums=0
        dictionary=os.path.join(data_path,'analysis',str(index))
        if not os.path.exists(dictionary):
            os.mkdir(dictionary)
        for i in range(len(gt_roidb)):
            if i>=len(gt_roidb)/2:
                image_ind=i-len(gt_roidb)/2
            else:
                image_ind=i
            path=os.path.join(data_path,'JPEGImages',image_index[image_ind]+image_ext)
    
            image_size=gt_roidb[i]['size']
            im_scale[i],ratio[i] = scale_and_ratio(min_size,max_size,image_size)
            resize_image_size = im_scale[i]*np.array(image_size)
            conv_width = resize_image_size[0]/feat_stride
            conv_height = resize_image_size[1]/feat_stride
            projection_gt = np.array(gt_roidb[i]['boxes'])*im_scale[i]
            num_anchors=len(anchors)
            all_anchors=generate_all_anchors(anchors,feat_stride,num_anchors,conv_width,conv_height,resize_image_size)
            overlaps = bbox_overlaps(
                np.ascontiguousarray(all_anchors, dtype=np.float),
                np.ascontiguousarray(projection_gt, dtype=np.float))
            max_overlaps = overlaps.max(axis=0)
            argmax_overlaps = overlaps.argmax(axis=0)
            im_file = os.path.join(path)
            im = cv2.imread(im_file)
            if gt_roidb[i]['flipped']:
                im = im[:, ::-1, :]
                write_path=os.path.join(dictionary,image_index[image_ind]+'_filp'+image_ext)
            else:
                write_path=os.path.join(dictionary,image_index[image_ind]+image_ext)
            img=cv2.resize(im,(int(resize_image_size[0]),int(resize_image_size[1])))
#========================================================================================#
            for ii in range(len(max_overlaps)):
                if max_overlaps[ii]<j:
                    rect_start=(int(projection_gt[ii][0]),int(projection_gt[ii][1]))
                    rect_end=(int(projection_gt[ii][2]),int(projection_gt[ii][3]))
                    cv2.rectangle(img, rect_start, rect_end, color_gt, 2)
                    ind=argmax_overlaps[ii]
                    anchor_start=(int(all_anchors[ind][0]),int(all_anchors[ind][1]))
                    anchor_end=(int(all_anchors[ind][2]),int(all_anchors[ind][3]))
                    cv2.rectangle(img, anchor_start, anchor_end, color_anchor, 2)
                    cv2.imwrite(write_path,img)
#========================================================================================#将不同阈值IOU下没有召回的真值框和对应最大anchor可视化    
            recall_num=len(np.where(max_overlaps>=j)[0])
            ind_nums+=len(projection_gt)
            recall_nums+=recall_num
        recall[index]=recall_nums/float(ind_nums)
        index+=1
    print 'Anchor_Recall: {}'.format(recall) #训练数据的anchors和训练gt的recall，可以通过不同scale和ratio的调节选择更合理的anchor尺寸比


    ##=======================================================## generation proposals recall
    if os.path.exists(proposal_path):
        with open(proposal_path, 'rb') as fid:
            proposal_roidb = cPickle.load(fid)
    index=0
    for j in np.arange(0.1,1.1,0.1):
        ind_nums=0
        recall_nums=0
        for i in range(len(proposal_roidb)):
            image_size=gt_roidb[i]['size']
            im_scale[i],ratio[i] = scale_and_ratio(min_size,max_size,image_size)
            projection_gt = np.array(gt_roidb[i]['boxes'])*im_scale[i]
            proposal = np.array(proposal_roidb[i])*im_scale[i]
            overlaps = bbox_overlaps(
                np.ascontiguousarray(proposal, dtype=np.float),
                np.ascontiguousarray(projection_gt, dtype=np.float))
            max_overlaps = overlaps.max(axis=0)
            recall_num=len(np.where(max_overlaps>=j)[0])
            ind_nums+=len(projection_gt)
            recall_nums+=recall_num
        recall[index]=recall_nums/float(ind_nums)
        index+=1
    print 'Proposal_Recall: {}'.format(recall) #训练数据在训练好的RPN上生成的propsoal的recall



    ##=======================================================## detection proposals recall
    image_test_index=load_image_set_index(data_path,image_set_test)
    test_gt_roidb = [load_pascal_annotation(data_path,index) for index in image_test_index]
    im_scale_test=np.zeros(len(test_gt_roidb),dtype=np.float32)
    test_ratio=np.zeros(len(test_gt_roidb),dtype=np.float32)
    test_recall=np.zeros(10,dtype=np.float32)
    detection_path='/home/bsl/py-faster-rcnn-master/output/faster_rcnn_alt_opt/voc_2007_test/VGG16_faster_rcnn_final/proposals.pkl' #测试数据在模型中的proposals由修改后的test.py生成
    if os.path.exists(detection_path):
        with open(detection_path, 'rb') as fid:
            detection_roidb = cPickle.load(fid)
    index=0
    for j in np.arange(0.1,1.1,0.1):
        ind_nums=0
        recall_nums=0
        for i in range(len(detection_roidb)):
            image_size=test_gt_roidb[i]['size']
            im_scale_test[i],test_ratio[i] = scale_and_ratio(min_size,max_size,image_size)
            detection_gt = np.array(test_gt_roidb[i]['boxes'])*im_scale_test[i]
            detection = np.array(detection_roidb[i])*im_scale_test[i]
            overlaps = bbox_overlaps(
                np.ascontiguousarray(detection, dtype=np.float),
                np.ascontiguousarray(detection_gt, dtype=np.float))
            max_overlaps = overlaps.max(axis=0)
            recall_num=len(np.where(max_overlaps>=j)[0])
            ind_nums+=len(detection_gt)
            recall_nums+=recall_num
        recall[index]=recall_nums/float(ind_nums)
        index+=1
    print 'Detection_proposal_Recall: {}'.format(recall) #测试数据的proposals的recall值
```

# 测试结果

```
IOU 从0.1-1间隔0.1
anchor recall： 
0.98580265  0.94594699  0.89966691  0.85790765  0.80508405  0.58752382 0.37503967 0.15402919  0.02458756  0

stage2 proposal recall:
1 0.99976206  0.99825507  0.99516183  0.98921317  0.97017765 0.88911802  0.45566308  0.04877855  0 

test set proposal recall:
0.99833775  0.99484706  0.9877826   0.97473407  0.95071477  0.89702457 0.74459773  0.32829124  0.03565492  0 
```

# 总结

实验表明在voc2007检测任务中，RPN在IOU为0.5时proposal的recall为95%以上，但检测结果为69mAP，这表明算法的瓶颈在分类问题上。

看我写的辛苦求打赏啊！！！有学术讨论和指点请加微信manutdzou,注明

![20](/public/img/pay.jpg)