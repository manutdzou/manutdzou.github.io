---
layout: post
title: code vectorzation
category: 科研
tags: 
keywords: 
description:
---

```python
def bbox_iou(box1, box2, x1y1x2y2=False):
    """
	box1:nparray(M*4)
	box2:nparray(N*4)
    Returns the IoU of two bounding boxes (M*N)
    """
    M, N = len(box1), len(box2)
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = np.maximum(np.expand_dims(b1_x1,axis=1), b2_x1)
    inter_rect_y1 = np.maximum(np.expand_dims(b1_y1,axis=1), b2_y1)
    inter_rect_x2 = np.minimum(np.expand_dims(b1_x2,axis=1), b2_x2)
    inter_rect_y2 = np.minimum(np.expand_dims(b1_y2,axis=1), b2_y2)
    # Intersection area
    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * np.maximum(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = np.tile(((b1_x2 - b1_x1) * (b1_y2 - b1_y1)).reshape(-1,1),[1,N])
    b2_area = np.tile(((b2_x2 - b2_x1) * (b2_y2 - b2_y1)).reshape(1,-1),[M,1])

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)
```


看我写的辛苦求打赏啊！！！有学术讨论和指点请加微信manutdzou,注明

![20](/public/img/pay.jpg)