---
layout: post
title: How to use glog
category: 技术
tags: 编程语言
keywords: glog
description: 库使用
---

# How to use glog

Glog是来自Google的一个轻量级日志库，它提供基于C++风格的流的日志API,以及各种辅助的宏。

## Windows下安装Glog

glog下载地址 https://github.com/google/glog，在vs中打开google-glog.sln，在Release下编译解决方案获得libglog.dll和liblog.lib两个文件(dll运行时使用，lib编译时使用)，将这两个文件拷贝到所需项目中，并拷贝./glog-master/glog-master/src/windows/glog目录到所需项目，至此Glog安装完毕。

## Windows下使用Glog

在项目的Properties->Configuration Properties->C/C++->General->Additional Include Directories中添加glog头文件所在路径，在Properties->Configuration Properties->Linker->Input->Additional Dependencies中添加libglog.lib的路径。

```c++
#include "glog/logging.h"
#include <iostream>

using namespace std;

// 日志支持类型（按严重性递增）
// INFO = GLOG_INFO
// WARNING = GLOG_WARNING,
// ERROR = GLOG_ERROR
// FATAL = GLOG_FATAL;

int _tmain(int argc, char* argv[])
{

// Initialize Google's logging library. 

google::InitGoogleLogging(argv[0]);

FLAGS_log_dir = "SavePath"; //指定glog输出文件路径（输出格式为 "<program name>.<hostname>.<user name>.log.<severity level>.<date>.<time>.<pid>"）

google::SetLogDestination(google::INFO,"../my_info_"); //第一个参数为日志级别，第二个参数表示输出目录及日志文件名前缀。

FLAGS_alsologtostderr = true; // 日志输出到stderr（终端屏幕），同时输出到日志文件。 FLAGS_logtostderr = true 日志输出到stderr，不输出到日志文件。

FLAGS_colorlogtostderr = true; //输出彩色日志到stderr

FLAGS_minloglevel = 0; //将大于等于该级别的日志同时输出到stderr和指定文件。日志级别 INFO, WARNING, ERROR, FATAL 的值分别为0、1、2、3。

LOG(INFO)<< "info message..." << endl;

LOG(WARNING)<< "warning message..." << endl;

LOG(ERROR)<< "error message..." << endl;

LOG(FATAL)<< "fatal message..." << endl; //打印完信息后程序终止报错

}
``` 

## 按条件或次数打印日志

```c++
LOG_IF(INFO, num_cookies > 10) << "Got lots of cookies"; //上面的日志只有在满足 num_cookies > 10 时才会打印。

LOG_EVERY_N(INFO, 10) << "Got the " << google::COUNTER << "th cookie"; //执行的第1、11、21、...次时打印日志。 (google::COUNTER 用来表示是哪一次执行)

LOG_IF_EVERY_N(INFO, (size > 1024), 10) << "Got the " << google::COUNTER << "th big cookie"; //执行满足条件的第1、11、21、...次。

LOG_FIRST_N(INFO, 20) << "Got the " << google::COUNTER << "th cookie"; //打印前20次。
```

## 调试模式

调试模式的日志宏只在调试模式下有效，在非调试模式会被清除。可以避免生产环境的程序由于大量的日志而变慢。

```c++
DLOG(INFO) << "Found cookies";

DLOG_IF(INFO, num_cookies > 10) << "Got lots of cookies";

DLOG_EVERY_N(INFO, 10) << "Got the " << google::COUNTER << "th cookie";
```
## CHECK宏

```c++
int number = 3;
CHECK(number == 4) << "fatal message...";
```

当变量number不等于4时会报FATAL的错 F0913 17:49:18.869132  8036 Glog1_test.cpp:30] Check failed: number == 4 fatal message...

有各种用于相等/不等检查的宏： CHECK_EQ, CHECK_NE, CHECK_LE, CHECK_LT, CHECK_GE, CHECK_GT 。它们比较两个值，在不满足期望时打印包括这两个值的 FATAL 日志

```c++
int number = 3;
CHECK_EQ(number, 4) << "fatal message...";
```

报错 F0913 18:00:52.602131  6632 Glog1_test.cpp:30] Check failed: number == 4 (3 vs. 4) fatal message...

## 异常信号处理

glog提供了比较方便的程序异常处理机制。例如，当程序出现SIGSEGV异常信号时，glog的默认异常处理过程会导出非常有用的异常信息。异常处理过程可以通过google::InstallFailureSignalHandler()来自定义。

使用google::InstallFailureSignalHandler(); 和 google::InstallFailureWriter(&FatalMessageDump); 可以在程序出现严重错误时将详细的错误信息打印出来，但是使用默认编译的glog将会出现找不到此函数定义的问题，类似于：

```
error LNK2019: 无法解析的外部符号 "__declspec(dllimport) void __cdecl google::InstallFailureWriter(void (__cdecl*)(char const *,int))" (__imp_?InstallFailureWriter@google@@YAXP6AXPBDH@Z@Z)，该符号在函数 "public: void __thiscall EnvironmentManager::Initialize(int,char * *,int)" (?Initialize@EnvironmentManager@@QAEXHPAPADH@Z) 中被引用
error LNK2019: 无法解析的外部符号 "__declspec(dllimport) void __cdecl google::InstallFailureSignalHandler(void)" (__imp_?InstallFailureSignalHandler@google@@YAXXZ)，该符号在函数 "public: void __thiscall EnvironmentManager::Initialize(int,char * *,int)" (?Initialize@EnvironmentManager@@QAEXHPAPADH@Z) 中被引用
```

这个时候只需要在默认的glog工程中，将signalhandler.cc 纳入到libglog工程中，重新编译生成dll个lib文件即可。