---
layout: post
title: use windows caffe like opencv
category: 科研
tags: 深度学习
keywords: windows caffe
description:
---

# use windows caffe like opencv

如何像Opencv一样方便地使用caffe就要搞好include,lib,dll建立正确的属性表,这个过程看似方便其实充满了不少的坑，下面详细描述我建立库的过程。

首先建立一个vs2015下的解决方案，我想把cpp_classification这个程序脱离caffe的文件编译系统单独编译，以后的工作也就能这样基于这个新建的环境做模型向windows的转移。下面以release版本的配置环境为例：

## 详细过程

在属性管理器Release64下新建一个caffe_release的配置文件，方便以后新建项目时候不需要重复配置环境。

在C/C++->常规->附加包含目录中添加包含头文件

```
C:\Users\ZouJinyi\Anaconda2\include
C:\caffe\build\include
C:\caffe\include
C:\caffe\build\libraries\include
C:\caffe\build\libraries\include\boost-1_61
C:\caffe\build\libraries\include\google
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include
```

在链接器->常规->附加库目录中添加所需lib的路径

```
C:\Users\ZouJinyi\Anaconda2\libs
C:\caffe\build\lib\Release
C:\caffe\build\libraries\lib
C:\caffe\build\libraries\x64\vc14\lib
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64
```

在链接器->输入->附加依赖项中添加所需lib

```
opencv_imgcodecs310.lib
opencv_highgui310.lib
opencv_core310.lib
python27.lib
caffe.lib
proto.lib
opencv_imgproc310.lib
caffehdf5.lib
caffehdf5_cpp.lib
caffehdf5_hl.lib
caffehdf5_hl_cpp.lib
caffezlib.lib
caffezlibstatic.lib
glog.lib
leveldb.lib
libboost_date_time-vc140-mt-1_61.lib
libboost_filesystem-vc140-mt-1_61.lib
boost_python-vc140-mt-1_61.lib
libboost_chrono-vc140-mt-1_61.lib
libboost_system-vc140-mt-1_61.lib
boost_thread-vc140-mt-1_61.lib
gflags.lib
libopenblas.dll.a
libcaffehdf5.lib
libcaffehdf5_cpp.lib
libcaffehdf5_hl.lib
libcaffehdf5_hl_cpp.lib
libprotobuf.lib
libprotobuf-lite.lib
libprotoc.lib
lmdb.lib
snappy.lib
snappy_static.lib
cublas.lib
cublas_device.lib
cuda.lib
cudadevrt.lib
cudnn.lib
cudart.lib
cufft.lib
cudart_static.lib
cufftw.lib
cusparse.lib
cusolver.lib
curand.lib
nppc.lib
OpenCL.lib
```

在添加lib时候遇到一个大坑，可能是由于不同平台问题，编译时候报了一个缺少libboost_date_time-vc140-mt-1_61.lib，libboost_filesystem-vc140-mt-1_61.lib的错误。但找来找去没有这个lib,于是将一个名字类似但头上缺少lib的lib文件添加lib名字绕过这个错误。奇怪的是用CMake或者Ninja或者CMake编译后生成的Caffe.sln编译都没有报这个错，真是神奇。以上lib应该包含了所需的全部库文件，如果还报缺少lib错误就继续往这里添加所需的lib。

至此属性表建好了，然后将cpp_classification.cpp的内容拷贝到新建项目中，编译通过生成exe文件，将上面添加lib对应的dll放到exe文件夹中，如果还有缺失，报缺少什么添加什么。接下去在cmd添加参数运行这个exe，遇到第二个大坑。程序再次报错Check failed: registry.count(t ype) == 1 (0 vs. 1) Unknown layer type，一个莫名其妙的错误。这个错误在linux中只有当所用layer不是caffe中已经定义好的layer才会出错，但现在用的都是基本的layer。只能继续google，找到解决方案。当单独分离程序时候需要在头文件中添加

```
extern INSTANTIATE_CLASS(InputLayer)
```

用什么层加什么名字，简单粗暴，具体原因不详！如果遇到未注册的layer需要添加REGISTER_LAYER_CLASS(InputLayer)，一般不需要。终于程序正常运行了。

下面贴出我分离出来的项目

头文件

```
// stdafx.h : 标准系统包含文件的包含文件，
// 或是经常使用但不常更改的
// 特定于项目的包含文件
//

#pragma once

#include "targetver.h"

#include <stdio.h>
#include <tchar.h>

#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef WITH_PYTHON_LAYER
#include <boost/python.hpp>
#endif
#include <string>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/layers/input_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/dropout_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/lrn_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/tanh_layer.hpp"
#include "caffe/proto/caffe.pb.h"

#ifdef USE_CUDNN
#include "caffe/layers/cudnn_conv_layer.hpp"
#include "caffe/layers/cudnn_lcn_layer.hpp"
#include "caffe/layers/cudnn_lrn_layer.hpp"
#include "caffe/layers/cudnn_pooling_layer.hpp"
#include "caffe/layers/cudnn_relu_layer.hpp"
#include "caffe/layers/cudnn_sigmoid_layer.hpp"
#include "caffe/layers/cudnn_softmax_layer.hpp"
#include "caffe/layers/cudnn_tanh_layer.hpp"
#endif

#ifdef WITH_PYTHON_LAYER
#include "caffe/layers/python_layer.hpp"
#endif

using namespace caffe;  // NOLINT(build/namespaces)

extern INSTANTIATE_CLASS(InputLayer);
extern INSTANTIATE_CLASS(InnerProductLayer);
extern INSTANTIATE_CLASS(DropoutLayer);
extern INSTANTIATE_CLASS(ConvolutionLayer);
extern INSTANTIATE_CLASS(ReLULayer);
extern INSTANTIATE_CLASS(PoolingLayer);
extern INSTANTIATE_CLASS(LRNLayer);
extern INSTANTIATE_CLASS(SoftmaxLayer);
// TODO:  在此处引用程序需要的其他头文件
```

cpp文件

```
// caffe_test.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"

#ifdef USE_OPENCV

using std::string;
/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier {
public:
	Classifier(const string& model_file,
		const string& trained_file,
		const string& mean_file,
		const string& label_file);

	std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

private:
	void SetMean(const string& mean_file);

	std::vector<float> Predict(const cv::Mat& img);

	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img,
		std::vector<cv::Mat>* input_channels);

private:
	shared_ptr<Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
	cv::Mat mean_;
	std::vector<string> labels_;
};

Classifier::Classifier(const string& model_file,
	const string& trained_file,
	const string& mean_file,
	const string& label_file) {
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif

	/* Load the network. */
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

	/* Load the binaryproto mean file. */
	SetMean(mean_file);

	/* Load labels. */
	std::ifstream labels(label_file.c_str());
	CHECK(labels) << "Unable to open labels file " << label_file;
	string line;
	while (std::getline(labels, line))
		labels_.push_back(string(line));

	Blob<float>* output_layer = net_->output_blobs()[0];
	CHECK_EQ(labels_.size(), output_layer->channels())
		<< "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
	const std::pair<float, int>& rhs) {
	return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
	std::vector<std::pair<float, int> > pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(std::make_pair(v[i], static_cast<int>(i)));
	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

	std::vector<int> result;
	for (int i = 0; i < N; ++i)
		result.push_back(pairs[i].second);
	return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
	std::vector<float> output = Predict(img);

	N = std::min<int>(labels_.size(), N);
	std::vector<int> maxN = Argmax(output, N);
	std::vector<Prediction> predictions;
	for (int i = 0; i < N; ++i) {
		int idx = maxN[i];
		predictions.push_back(std::make_pair(labels_[idx], output[idx]));
	}

	return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
	BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

	/* Convert from BlobProto to Blob<float> */
	Blob<float> mean_blob;
	mean_blob.FromProto(blob_proto);
	CHECK_EQ(mean_blob.channels(), num_channels_)
		<< "Number of channels of mean file doesn't match input layer.";

	/* The format of the mean file is planar 32-bit float BGR or grayscale. */
	std::vector<cv::Mat> channels;
	float* data = mean_blob.mutable_cpu_data();
	for (int i = 0; i < num_channels_; ++i) {
		/* Extract an individual channel. */
		cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
		channels.push_back(channel);
		data += mean_blob.height() * mean_blob.width();
	}

	/* Merge the separate channels into a single image. */
	cv::Mat mean;
	cv::merge(channels, mean);

	/* Compute the global mean pixel value and create a mean image
	* filled with this value. */
	cv::Scalar channel_mean = cv::mean(mean);
	mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_,
		input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Preprocess(img, &input_channels);

	net_->Forward();

	/* Copy the output layer to a std::vector */
	Blob<float>* output_layer = net_->output_blobs()[0];
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->channels();
	return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation and we
* don't need to rely on cudaMemcpy2D. The last preprocessing
* operation will write the separate channels directly to the input
* layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void Classifier::Preprocess(const cv::Mat& img,
	std::vector<cv::Mat>* input_channels) {
	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;

	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	cv::Mat sample_normalized;
	cv::subtract(sample_float, mean_, sample_normalized);

	/* This operation will write the separate BGR planes directly to the
	* input layer of the network because it is wrapped by the cv::Mat
	* objects in input_channels. */
	cv::split(sample_normalized, *input_channels);

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		== net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}

int main(int argc, char** argv) {
	if (argc != 6) {
		std::cerr << "Usage: " << argv[0]
			<< " deploy.prototxt network.caffemodel"
			<< " mean.binaryproto labels.txt img.jpg" << std::endl;
		return 1;
	}

	::google::InitGoogleLogging(argv[0]);

	string model_file = argv[1];
	string trained_file = argv[2];
	string mean_file = argv[3];
	string label_file = argv[4];
	Classifier classifier(model_file, trained_file, mean_file, label_file);

	string file = argv[5];

	std::cout << "---------- Prediction for "
		<< file << " ----------" << std::endl;

	cv::Mat img = cv::imread(file, -1);
	CHECK(!img.empty()) << "Unable to decode image " << file;
	std::vector<Prediction> predictions = classifier.Classify(img);

	/* Print the top N predictions. */
	for (size_t i = 0; i < predictions.size(); ++i) {
		Prediction p = predictions[i];
		std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
			<< p.first << "\"" << std::endl;
	}
}
#else
int main(int argc, char** argv) {
	LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
```

需要在预定义中定义USE_OPENCV的宏。

下面贴出运行结果

![1](/public/img/posts/libcaffe使用/1.png)

终于可以基于libcaffe来开发上层建筑了，想象一下在ubuntu下训练好模型然后在windows下包装添加用户界面以供使用是多么方便和炫酷的一件事情。

看我写的辛苦求打赏啊！！！有学术讨论和指点请加微信manutdzou,注明

![20](/public/img/pay.jpg)

![2](/public/img/pay.jpg)