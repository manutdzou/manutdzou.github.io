---
layout: post
title: windows编译tensorflow
category: tensorflow
tags: 深度学习
keywords: tf学习
description: tf学习
---


# windows编译tensorflow

本测试环境Windows7+vs2015+cuda8.0+cudnn5

首先安装CMAKE:3.13.0(好像大于3.8就可以)

安装Python3.5(这个很重要，不管需不需要编译python lib都需要python3.5,而且目前只能是python3.5,在最后生成dll时候需要用到python)

安装swigwin3.0.12(这个大多数教程都说需要安装，好像实际可能并未使用到)

下载tensorflow源码，切换到r1.4分支，对应版本和机器环境有配对关系，需要完全对应否则有编译失败风险

```
https://github.com/tensorflow/tensorflow.git
git checkout r1.4
```

打开cmake-gui,设置cmake文件的路径和需要编译的位置，点击configure和generate生成项目解决方案。这里为了减少编译只选择必须要的项目编译

打开CMakeLists.txt,如下修改只编译需要的lib

```
# Options
option(tensorflow_VERBOSE "Enable for verbose output" OFF)
option(tensorflow_ENABLE_GPU "Enable GPU support" ON)
option(tensorflow_ENABLE_SSL_SUPPORT "Enable boringssl support" OFF)
option(tensorflow_ENABLE_GRPC_SUPPORT "Enable gRPC support" OFF)
option(tensorflow_ENABLE_HDFS_SUPPORT "Enable HDFS support" OFF)
option(tensorflow_ENABLE_JEMALLOC_SUPPORT "Enable jemalloc support" OFF)
option(tensorflow_BUILD_CC_EXAMPLE "Build the C++ tutorial example" OFF)
option(tensorflow_BUILD_PYTHON_BINDINGS "Build the Python bindings" OFF)
option(tensorflow_BUILD_ALL_KERNELS "Build all OpKernels" ON)
option(tensorflow_BUILD_CONTRIB_KERNELS "Build OpKernels from tensorflow/contrib/..." ON)
option(tensorflow_BUILD_CC_TESTS "Build cc unit tests " OFF)
option(tensorflow_BUILD_PYTHON_TESTS "Build python unit tests " OFF)
option(tensorflow_BUILD_MORE_PYTHON_TESTS "Build more python unit tests for contrib packages" OFF)
option(tensorflow_BUILD_SHARED_LIB "Build TensorFlow as a shared library" ON)
option(tensorflow_OPTIMIZE_FOR_NATIVE_ARCH "Enable compiler optimizations for the native processor architecture (if available)" ON)
option(tensorflow_WIN_CPU_SIMD_OPTIONS "Enables CPU SIMD instructions")
option(tensorflow_ENABLE_SNAPPY_SUPPORT "Enable SNAPPY compression support" OFF)
```

下面按大多数教程指出改为

```
if (tensorflow_OPTIMIZE_FOR_NATIVE_ARCH)
  include(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
  if (COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
  else()
    CHECK_CXX_COMPILER_FLAG("/arch:AVX" COMPILER_OPT_ARCH_AVX_SUPPORTED)
    if(COMPILER_OPT_ARCH_AVX_SUPPORTED)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX")
    endif()
  endif()
endif()
```

千万要确保PYTHON_EXECUTABLE的python.exe为python3.5,如下图

![1](/public/img/posts/tensorflow build/1.png)

打开vs2015下的project,只能用release下x64 build编译，点击ALL_BUILD编译全部项目。

这里为了保证有些编译错误，需要把re2\src\re2\re2\testing\search_test.cc和re2\src\re2\re2\testing\re2_test.cc和re2\src\re2\re2\testing\re2_test.cc改为ANSI编码(实际上不改也应该不会影响最后生成dll).

另外在实际编译的时候有可能create_def_file.py会报Python FileNotFoundError: [WinError 2] 系统找不到指定的文件的错误，这时候需要在lib中找到subprocess.py,搜索class Popen(object):将__init__中的shell=False修改为shell=True. 其他可能会报compiler is out of heap space,我的电脑8G内存并未遇到。

最后编译成功，在Release下生成tensorflow.lib和tensorflow.dll.

![2](/public/img/posts/tensorflow build/2.png)