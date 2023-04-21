---
layout: post
title: python小程序
category: 示例代码
tags: code
keywords: python代码
description: 
---

# python连续帧图片写视频

```python
import cv2,os
dictionary='show'
size = (1280,672)
fps=20
video=cv2.VideoWriter('demo_show.avi', cv2.cv.CV_FOURCC('M','J','P','G'), fps,size)
for i in range(5782):
    name = str(i)+'.jpg'
    im_path = os.path.join('/home/dlg',dictionary,name)
    im = cv2.imread(im_path)
    video.write(im)
print 'done'
```

# python 用opencv检测视频中人脸

```python
import cv2
import os
import sys
 
OUTPUT_DIR = './my_faces'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
face_haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)
count = 0
while True:
    print(count)
    if count < 10000:
        _, img = cam.read()
	gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_haar.detectMultiScale(gray_image, 1.3, 5)
	for face_x,face_y,face_w,face_h in faces:
	    face = img[face_y:face_y+face_h, face_x:face_x+face_w]
	    face = cv2.resize(face, (64, 64))
	    cv2.imshow('img', face)
	    cv2.imwrite(os.path.join(OUTPUT_DIR, str(count)+'.jpg'), face)
	    count += 1
	key = cv2.waitKey(30) & 0xff
	if key == 27:
            break
    else:
        break
```

# 用python requests模块下载数据

```python
import requests
import tarfile
 
url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
save_path = "my_path"
if not os.path.exists(save_path):
	os.makedirs(save_path)
 
filename = "save_file"
filepath = os.path.join(save_path, filename)
# 下载
if not os.path.exists(filepath):
	print("downloading...", filename)
	r = requests.get(url)
	with open(filepath, 'wb') as f:
	    f.write(r.content)
# 解压
tarfile.open(filepath, 'r:gz').extractall(filepath)
```

# python opencv3 跟踪API

```python
import cv2
import sys
 
if __name__ == '__main__' :
 
    # Set up tracker.
    # Instead of MIL, you can also use
    # BOOSTING, KCF, TLD, MEDIANFLOW or GOTURN
     
    tracker = cv2.Tracker_create("MIL")
 
    # Read video
    video = cv2.VideoCapture("1.mp4")
 
    # Exit if video not opened.
    if not video.isOpened():
        print "Could not open video"
        sys.exit()
 
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print 'Cannot read video file'
        sys.exit()
     
    # Define an initial bounding box
    #bbox = (287, 23, 86, 320)
 
    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(frame, False)
 
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
 
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break
         
        # Update tracker
        ok, bbox = tracker.update(frame)
 
        # Draw bounding box
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0,0,255))
 
        # Display result
        cv2.imshow("Tracking", frame)
 
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break
```

# python opencv 画图

```
#!/usr/bin/env python  
  
import numpy as np  
import cv2  
  
img = np.zeros((512,512,3), np.uint8)  
  
cv2.line(img, (0,0), (511, 511), (255,0,0), 5) #line color (BGR)  
  
cv2.rectangle(img, (384,0), (510, 128), (0, 255, 0), 3)  
  
cv2.circle(img, (447, 63), 63, (0,0,255), -1) #linewidth -1 means fill circle using setted color  
  
cv2.ellipse(img, (256,256), (100,50),45,0,270,(0,0,255),-1) #椭圆的第二个参数是椭圆中心，第三个参数是椭圆长轴和短轴对应的长度，第四个参数45是顺时针旋转45度， 第五个参数是从0度开始，顺时针画270的弧，第七个参数是颜色，最后一个是用颜色填充椭圆内部  
font = cv2.FONT_HERSHEY_SIMPLEX  
cv2.putText(img, 'Hello', (10,500), font, 4, (255,255,255), 2)  
  
cv2.imshow('image', img)  
cv2.waitKey(0)
```