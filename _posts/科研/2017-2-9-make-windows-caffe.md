---
layout: post
title: windows caffe implement
category: 科研
tags: 深度学习
keywords: windows caffe
description:
---

# windows7+cuda8+cudnn5.1+vs2015+Anaconda2配置windows caffe

由于windows版本下的caffe有强烈的环境要求，对于装过其他vs版本的电脑来说卸载vs简直是个噩梦，博主卸载了一天vs2010和vs2013没卸干净，导致后续vs2015直接无法安装，所以一怒之下重装系统（非常建议重装系统）。下面分享一下我两天的环境配置历程。

## 重装系统

重新安装windows7系统由于vs2015的环境需求win7要sp1版本以上。下载软碟通将win7系统做成U盘启动，对于双系统机器windows7+ubuntu的机器，只需要把新系统安装在原来的windows7下，然后利用EasyBCD重新添加Ubuntu的启动项就行。在这里还遇到一个小插曲，系统装好以后鼠键没有了反应，主要是因为主板USB驱动不兼容问题，在主板设置将USB调成自动就行。至此一个纯净的windows7系统准备完成。

## 安装vs2015

我下载了一个社区版本的vs2015，安装时候可以选择自己需要的工具，安装过程比较久2到3个小时。这里需要说明cuda的安装依赖于vs,它需要往vs里面添加cuda的配置，所以vs安装要先于cuda.

## 安装cuda8

直接官网安装win7下的cuda8,安装cuda也遇到一个坑由于我先安装了显卡驱动，我发现我的显卡驱动比cuda8里的驱动版本要新，导致cuda无法安装。所以建议直接裸机安装cuda8里面所有东西,一路yes下去，会自动把环境变量添加到系统。安装完成后cmd nvcc-v可以显示编译器信息，然后编译cuda sample运行里面的deviceQuery.exe 可以显示如下信息表示安装完成

![1](/public/img/posts/windows caffe/1.png)

## 安装cudnn5.1

直接官网注册下载适配的cudnn解压里面东西按对应目录分别放置在C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0对应的文件夹里面。

## 安装python环境

由于自动配置lib时候需要用到python脚本所以需要安装python，这里推荐Anaconda Python 2.7。注意安装时候选择将路径添加到系统环境变量否则后续项目找不到python解释器

## 安装CMake3.4以上版本以及Ninja

自动添加CMake路径到系统环境变量

## 配置windows caffe

下载windows版caffe, cmd下

```
git clone https://github.com/BVLC/caffe.git
cd caffe
git checkout windows
```

下面配置scripts\build_win.cmd文件

```
:: Set python 2.7 with conda as the default python
if !PYTHON_VERSION! EQU 2 (
    set CONDA_ROOT=C:\Users\ZouJinyi\Anaconda2
)
```

这儿将python的路径修改为自己的python路径

```
:: Set the appropriate CMake generator
:: Use the exclamation mark ! below to delay the
:: expansion of CMAKE_GENERATOR
if %WITH_NINJA% EQU 0 (
    if "%MSVC_VERSION%"=="14" (
        set CMAKE_GENERATOR=Visual Studio 14 2015 Win64
    )
```

这儿需要根据你的vs版本以及cmake的识别进行修改，这儿一定要选Win64,测试发现不使用Win64时候Ninja可以通过编译，CMake编译出错

```
-DCUDNN_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0 ^
```

在cmake命令中添加cudnn支持，也就是添加cudnn安装位置

到这儿已经完成了一个由GPU和cudnn支持同时支持python和python layer的caffe，在cmd中运行scripts\build_win.cmd，这个命令首先会调用download_prebuilt_dependencies.py自动安装需要的lib,这个功能非常好用，省去了配置编译环境的各种大坑。同时也可以发现如果用vs2015编译Caffe.sln时可以使用NuGet来自动搜索下载依赖库，超爽有木有。编译完成后如下：

![2](/public/img/posts/windows caffe/2.png)

![3](/public/img/posts/windows caffe/3.png)

我们来测试一下caffe的可执行文件C:\Projects\caffe\build\examples\cpp_classification\classification.exe,执行exe需要将所有需要的dll放到一起，或者可以在libraries里的prependpath.bat添加作用到的路径

```
@echo off
:: Prepend the path variable
set PATH=%~dp0bin;%~dp0lib;%~dp0x64\vc14\bin;%~dp0..\examples\cpp_classification\;%PATH%
```

![4](/public/img/posts/windows caffe/4.png)

同时pycaffe也编译完成，在caffe\python\caffe中生成_caffe.pyd以及所有的动态依赖库。将python下整个caffe文件夹拷贝到Anaconda2\Lib\site-packages。至此pycaffe在整个windows系统中可用。

![5](/public/img/posts/windows caffe/5.png)

至于matcaffe同样只需要开启matcaffe编译就行。

到这就能实现在windows下使用caffe的全部功能。特别是Windows下pycaffe的使用将和linux下无异。

由于windows下的Caffe.sln是vs2013下的工程所以在vs2015下无法编译通过，所以需要用cmake重新生成sln文件，方法是禁用ninja编译

```
) else (
    :: Change the settings here to match your setup
    :: Change MSVC_VERSION to 12 to use VS 2013
    if NOT DEFINED MSVC_VERSION set MSVC_VERSION=14
    :: Change to 1 to use Ninja generator (builds much faster)
    if NOT DEFINED WITH_NINJA set WITH_NINJA=0
```

编译完成后在build下会生成vs2015下的Caffe.sln

接着我将尝试在这个基础上开发基于windows系统的基于Windows caffe的exe程序。将会在后续开发中记录。

`搭建开发环境到写完总结用了三个晚上的时间，每晚到凌晨2点啊非常辛苦啊，希望各位大咖在参考时候能小小的打赏一下，下面是我的微信收款哟，嘿嘿，好给我继续写下去的动力有学术讨论和指点请加微信manutdzou,注明`

![6](/public/img/pay.jpg)
