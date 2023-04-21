---
layout: post
title: windows下配置caffe_ssd
category: 科研
tags: 深度学习
keywords: windows caffe ssd
description:
---

# 本文主要描述如何在vs2015上使用caffe配置编译ssd运行在windows平台

首先下载windows版caffe, cmd下

```
git clone https://github.com/BVLC/caffe.git
cd caffe
git checkout windows
```

下载caffe的ssd程序https://github.com/conner99/caffe，这个版本在windows编译上做好了修改的，将conner99/caffe中的include和src文件夹替换caffe原有的include和src.我们需要在CMakeLists.txt中关闭cudnn支持，因为可能会有部分不兼容

```
caffe_option(USE_CUDNN "Build Caffe with cuDNN library support" OFF)
```

在common.hpp和bbox_util.hpp中添加glog的宏定义防止报错

```
#ifndef GLOG_NO_ABBREVIATED_SEVERITIES 
#define GLOG_NO_ABBREVIATED_SEVERITIES 
#endif
#include <glog/logging.h>
```

在detection_output_layer.hpp注释掉boost正则化防止报错和Forward_gpu这个函数

```
//#include <boost/regex.hpp>

//virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//   const vector<Blob<Dtype>*>& top);
/// @brief Not implemented
```

在detection_output_layer.cpp注释掉所有使用了boost的代码

删除bbox_util.cu和detection_output_layer.cu

删除tools，python和examples里面CMakeLists.txt的内容使得不编译文件夹内项目，因为本人测试时候出现若干未可知的错误。同样原因不编译python接口。

将主文件夹内CMakeLists.txt修改为

```
caffe_option(BUILD_python "Build Python wrapper" OFF)
caffe_option(BUILD_python_layer "Build the Caffe Python layer" OFF)
```

使用cmake或者ninja编译生成libcaffe

至此生成了支持ssd特殊layer的libcaffe,然后封装检测器函数，在vs2015中建立project，caffelib的配置前面已经介绍，完全一样

stdafx.h

```C++
// stdafx.h : 标准系统包含文件的包含文件，
// 或是经常使用但不常更改的
// 特定于项目的包含文件
//

#pragma once

#include "targetver.h"

#include <stdio.h>
#include <tchar.h>

#include <caffe/caffe.hpp>
#ifdef USE_OPENCV //USE_OPENCV = 1
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef WITH_PYTHON_LAYER //WITH_PYTHON_LAYER = 0
#include <boost/python.hpp>
#endif

#include <string>
//all layers used in SSD
#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/layers/input_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/normalize_layer.hpp"
#include "caffe/layers/split_layer.hpp"
#include "caffe/layers/permute_layer.hpp"
#include "caffe/layers/flatten_layer.hpp"
#include "caffe/layers/prior_box_layer.hpp"
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/reshape_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/detection_output_layer.hpp"

#include "caffe/proto/caffe.pb.h"

#ifdef USE_CUDNN //USE_CUDNN = 0
#include "caffe/layers/cudnn_conv_layer.hpp"
#include "caffe/layers/cudnn_pooling_layer.hpp"
#include "caffe/layers/cudnn_relu_layer.hpp"
#include "caffe/layers/cudnn_softmax_layer.hpp"
#endif

#ifdef WITH_PYTHON_LAYER // WITH_PYTHON_LAYER=0
#include "caffe/layers/python_layer.hpp"
#endif

using namespace caffe;  // NOLINT(build/namespaces)

extern INSTANTIATE_CLASS(InputLayer);
extern INSTANTIATE_CLASS(ConvolutionLayer);
REGISTER_LAYER_CLASS(Convolution);
extern INSTANTIATE_CLASS(ReLULayer);
REGISTER_LAYER_CLASS(ReLU);
extern INSTANTIATE_CLASS(PoolingLayer);
REGISTER_LAYER_CLASS(Pooling);
extern INSTANTIATE_CLASS(NormalizeLayer);
extern INSTANTIATE_CLASS(SplitLayer);
extern INSTANTIATE_CLASS(PermuteLayer);
extern INSTANTIATE_CLASS(FlattenLayer);
extern INSTANTIATE_CLASS(PriorBoxLayer);
extern INSTANTIATE_CLASS(ConcatLayer);
extern INSTANTIATE_CLASS(ReshapeLayer);
extern INSTANTIATE_CLASS(SoftmaxLayer);
REGISTER_LAYER_CLASS(Softmax);
extern INSTANTIATE_CLASS(DetectionOutputLayer);

// TODO:  在此处引用程序需要的其他头文件
```

ssd.cpp

```C++
// ssd.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"


// This is a demo code for using a SSD model to do detection.
// The code is modified from examples/cpp_classification/classification.cpp.
// Usage:
//    ssd_detect [FLAGS] model_file weights_file list_file
//
// where model_file is the .prototxt file defining the network architecture, and
// weights_file is the .caffemodel file containing the network parameters, and
// list_file contains a list of image files with the format as follows:
//    folder/img1.JPEG
//    folder/img2.JPEG
// list_file can also contain a list of video files with the format as follows:
//    folder/video1.mp4
//    folder/video2.mp4
//
#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)

class Detector {
public:
	Detector(const string& model_file,
		const string& weights_file,
		const string& mean_file,
		const string& mean_value);

	std::vector<vector<float> > Detect(const cv::Mat& img);

private:
	void SetMean(const string& mean_file, const string& mean_value);

	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img,
		std::vector<cv::Mat>* input_channels);

private:
	shared_ptr<Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
	cv::Mat mean_;
};

Detector::Detector(const string& model_file,
	const string& weights_file,
	const string& mean_file,
	const string& mean_value) {
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif

	/* Load the network. */
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(weights_file);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

	/* Load the binaryproto mean file. */
	SetMean(mean_file, mean_value);
}

std::vector<vector<float> > Detector::Detect(const cv::Mat& img) {
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
	Blob<float>* result_blob = net_->output_blobs()[0];
	const float* result = result_blob->cpu_data();
	const int num_det = result_blob->height();
	vector<vector<float> > detections;
	for (int k = 0; k < num_det; ++k) {
		if (result[0] == -1) {
			// Skip invalid detection.
			result += 7;
			continue;
		}
		vector<float> detection(result, result + 7);
		detections.push_back(detection);
		result += 7;
	}
	return detections;
}

/* Load the mean file in binaryproto format. */
void Detector::SetMean(const string& mean_file, const string& mean_value) {
	cv::Scalar channel_mean;
	if (!mean_file.empty()) {
		CHECK(mean_value.empty()) <<
			"Cannot specify mean_file and mean_value at the same time";
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
		channel_mean = cv::mean(mean);
		mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
	}
	if (!mean_value.empty()) {
		CHECK(mean_file.empty()) <<
			"Cannot specify mean_file and mean_value at the same time";
		stringstream ss(mean_value);
		vector<float> values;
		string item;
		while (getline(ss, item, ',')) {
			float value = std::atof(item.c_str());
			values.push_back(value);
		}
		CHECK(values.size() == 1 || values.size() == num_channels_) <<
			"Specify either 1 mean_value or as many as channels: " << num_channels_;

		std::vector<cv::Mat> channels;
		for (int i = 0; i < num_channels_; ++i) {
			/* Extract an individual channel. */
			cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
				cv::Scalar(values[i]));
			channels.push_back(channel);
		}
		cv::merge(channels, mean_);
	}
}

/* Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation and we
* don't need to rely on cudaMemcpy2D. The last preprocessing
* operation will write the separate channels directly to the input
* layer. */
void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
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

void Detector::Preprocess(const cv::Mat& img,
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

DEFINE_string(mean_file, "",
	"The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "104,117,123",
	"If specified, can be one value or can be same as image channels"
	" - would subtract from the corresponding channel). Separated by ','."
	"Either mean_file or mean_value should be provided, not both.");
DEFINE_string(file_type, "video",
	"The file type in the list_file. Currently support image and video.");
DEFINE_string(out_file, "",
	"If provided, store the detection results in the out_file.");
DEFINE_double(confidence_threshold, 0.6,
	"Only store detections with score higher than the threshold.");

int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);
	// Print output to stderr (while still logging)
	FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif

	gflags::SetUsageMessage("Do detection using SSD mode.\n"
		"Usage:\n"
		"    ssd_detect [FLAGS] model_file weights_file list_file\n");
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	if (argc < 4) {
		gflags::ShowUsageWithFlagsRestrict(argv[0], "examples/ssd/ssd_detect");
		return 1;
	}

	const string& model_file = argv[1];
	const string& weights_file = argv[2];
	const string& mean_file = FLAGS_mean_file;
	const string& mean_value = FLAGS_mean_value;
	const string& file_type = FLAGS_file_type;
	const string& out_file = FLAGS_out_file;
	const float confidence_threshold = FLAGS_confidence_threshold;

	// Initialize the network.
	Detector detector(model_file, weights_file, mean_file, mean_value);

	// Set the output mode.
	std::streambuf* buf = std::cout.rdbuf();
	std::ofstream outfile;
	if (!out_file.empty()) {
		outfile.open(out_file.c_str());
		if (outfile.good()) {
			buf = outfile.rdbuf();
		}
	}
	std::ostream out(buf);

	// Process image one by one.
	std::ifstream infile(argv[3]);
	std::string file;
	while (infile >> file) {
		if (file_type == "image") {
			cv::Mat img = cv::imread(file, -1);
			CHECK(!img.empty()) << "Unable to decode image " << file;
			std::vector<vector<float> > detections = detector.Detect(img);

			/* Print the detection results. */
			for (int i = 0; i < detections.size(); ++i) {
				const vector<float>& d = detections[i];
				// Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
				CHECK_EQ(d.size(), 7);
				const float score = d[2];
				if (score >= confidence_threshold) {
					out << file << " ";
					out << static_cast<int>(d[1]) << " ";
					out << score << " ";
					out << static_cast<int>(d[3] * img.cols) << " ";
					out << static_cast<int>(d[4] * img.rows) << " ";
					out << static_cast<int>(d[5] * img.cols) << " ";
					out << static_cast<int>(d[6] * img.rows) << std::endl;
					cv::rectangle(img, cv::Rect(d[3] * img.cols, d[4] * img.rows, d[5] * img.cols - d[3] * img.cols, d[6] * img.rows - d[4] * img.rows), cv::Scalar(255, 0, 0), 2);
				}
			}
			cv::imshow("img", img);
			cv::waitKey(0);
		}
		else if ((file_type == "video"&&file != "0")) {
			cv::VideoCapture cap(file);
			if (!cap.isOpened()) {
				LOG(FATAL) << "Failed to open video: " << file;
			}
			cv::Mat img;
			int frame_count = 0;
			while (true) {
				bool success = cap.read(img);
				if (!success) {
					LOG(INFO) << "Process " << frame_count << " frames from " << file;
					break;
				}
				CHECK(!img.empty()) << "Error when read frame";
				std::vector<vector<float> > detections = detector.Detect(img);

				/* Print the detection results. */
				for (int i = 0; i < detections.size(); ++i) {
					const vector<float>& d = detections[i];
					// Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
					CHECK_EQ(d.size(), 7);
					const float score = d[2];
					if (score >= confidence_threshold) {
						out << file << "_";
						out << std::setfill('0') << std::setw(6) << frame_count << " ";
						out << static_cast<int>(d[1]) << " ";
						out << score << " ";
						out << static_cast<int>(d[3] * img.cols) << " ";
						out << static_cast<int>(d[4] * img.rows) << " ";
						out << static_cast<int>(d[5] * img.cols) << " ";
						out << static_cast<int>(d[6] * img.rows) << std::endl;
					}
				}
				++frame_count;
			}
			if (cap.isOpened()) {
				cap.release();
			}
		}
		else if (file_type == "video"&&file == "0") {
			cv::VideoCapture cap(0);
			if (!cap.isOpened()) {
				LOG(FATAL) << "Failed to open video: " << file;
			}
			cv::Mat img;
			int frame_count = 0;
			while (true) {
				bool success = cap.read(img);
				if (!success) {
					LOG(INFO) << "Process " << frame_count << " frames from " << file;
					break;
				}
				CHECK(!img.empty()) << "Error when read frame";
				std::vector<vector<float> > detections = detector.Detect(img);

				/* Print the detection results. */
				for (int i = 0; i < detections.size(); ++i) {
					const vector<float>& d = detections[i];
					// Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
					CHECK_EQ(d.size(), 7);
					const float score = d[2];
					if (score >= confidence_threshold) {
						out << file << "_";
						out << std::setfill('0') << std::setw(6) << frame_count << " ";
						out << static_cast<int>(d[1]) << " ";
						out << score << " ";
						out << static_cast<int>(d[3] * img.cols) << " ";
						out << static_cast<int>(d[4] * img.rows) << " ";
						out << static_cast<int>(d[5] * img.cols) << " ";
						out << static_cast<int>(d[6] * img.rows) << std::endl;
						cv::rectangle(img, cv::Rect(d[3] * img.cols, d[4] * img.rows, d[5] * img.cols - d[3] * img.cols, d[6] * img.rows - d[4] * img.rows), cv::Scalar(255, 0, 0), 2);
					}
				}
				cv::imshow("img", img);
				if ((char)cv::waitKey(1) == 'q')
					break;
				++frame_count;
			}
			if (cap.isOpened()) {
				cap.release();
			}
		}
		else {
			LOG(FATAL) << "Unknown file_type: " << file_type;
		}
	}
	return 0;
}
#else
int main(int argc, char** argv) {
	LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
```

以上代码实现了检测图片，视频以及检测摄像头内容的SSD算法。执行代码：

```
ssd.exe C:\Users\ZouJinyi\Desktop\SSD_300x300_ft\deploy.prototxt C:\Users\ZouJinyi\Desktop\SSD_300x300_ft\VGG_VOC0712_SSD_300x300_ft_iter_120000.caffemodel C:\Users\ZouJinyi\Desktop\SSD_300x300_ft\list.txt
```

检测图像

list.txt给出图片路径，一行一张图片

```
DEFINE_string(file_type, "image",
	"The file type in the list_file. Currently support image and video.");
```

检测视频

list.txt给出视频路径，一行一个视频

```
DEFINE_string(file_type, "video",
	"The file type in the list_file. Currently support image and video.");
```

检测摄像头视频

list.txt填0

```
DEFINE_string(file_type, "video",
	"The file type in the list_file. Currently support image and video.");
```

看我写的辛苦求打赏啊！！！有学术讨论和指点请加微信manutdzou,注明

![3](/public/img/pay.jpg)
