---
layout: post
title: tensorrt docker制作
category: 技术
tags: docker使用
keywords: ubuntu
description:
---

# 从docker hub下载基础镜像

```
docker login输入账号密码

docker search cuda10.2

docker pull wildbrother/cuda10.2-cudnn8-runtime-ubuntu18.04

systemctl status nvidia-docker

systemctl start nvidia-docker

nvidia-docker run -t -i -d IMAGE ID /bin/bash

nvidia-docker run -t -i -d --shm-size=512g -v /dfsdata2/jinyi_data:/jinyi_data torch1.5.1-cuda10.1-cudnn7-devel:ddp /bin/bash

#or

sudo docker run --runtime=nvidia -it --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -e NVIDIA_VISIBLE_DEVICES=0 -w /  IMAGE ID

docker exec -it CONTAINER ID /bin/bash
```

## 安装opencv编译依赖

```
#cmake-3.19.2安装

apt-get autoremove cmake

apt install build-essential

apt-get install libssl-dev

下载解压cmake-3.19.2.tar.gz

./bootstrap

make 

make install

#protobuf安装

apt-get install autoconf autogen

wget http://ftp.gnu.org/gnu/automake/automake-1.16.tar.gz
tar xvfz automake-1.16.tar.gz
cd automake-1.16
./configure --prefix=/usr/local/automake/1_16
make
make install

apt-get install aptitude  
aptitude install libtool

下载解压protobuf-all-3.6.1.tar.gz

./autogen.sh

./configure

make 

make install
```

## opencv编译安装

```
cd build
cmake -D CMAKE_BUILD_TYPE=RELEAS -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D WITH_CUDA=ON -D BUILD_opencv_cudacodec=OFF -D BUILD_opencv_xfeatures2d=OFF -D OPENCV_PC_FILE_NAME=opencv.pc -D OPENCV_GENERATE_PKGCONFIG=YES -D WITH_V4L=ON -D WITH_GSTREAMER=ON -D OPENCV_EXTRA_MODULES_PATH=/root/opencv_contrib-4.2.0/modules -D PYTHON_EXECUTABLE=$(which python3) -D WITH_CUDNN=ON -D OPENCV_DNN_CUDA=ON -D CUDA_ARCH_BIN=7.5 ..

make
make install

apt-get install -y pkg-config
```

## 查看cudnn版本

```
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
```

##  安装nvdecode环境

```
git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
cd nv-codec-headers && sudo make install

git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg/
apt-get -y install build-essential pkg-config checkinstall git libfaac-dev libgpac-dev ladspa-sdk-dev libunistring-dev libbz2-dev \
  libjack-jackd2-dev libmp3lame-dev libsdl2-dev libopencore-amrnb-dev libopencore-amrwb-dev libvpx-dev libx264-dev libx265-dev libxvidcore-dev libopenal-dev libopus-dev \
  libsdl1.2-dev libtheora-dev libva-dev libvdpau-dev libvorbis-dev libx11-dev \
  libxfixes-dev texi2html yasm zlib1g-dev
  
./configure --enable-nonfree --enable-gpl --enable-version3 --enable-libmp3lame --enable-libvpx --enable-libopus --enable-opencl --enable-libxcb --enable-opengl --enable-nvenc --enable-vaapi --enable-vdpau --enable-ffplay --enable-ffprobe --enable-libxvid --enable-libx264 --enable-libx265 --enable-openal --enable-openssl --enable-cuda-nvcc --enable-cuvid --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64
make install
```