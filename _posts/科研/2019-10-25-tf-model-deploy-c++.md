---
layout: post
title: tensorflow C++ deploy
category: 科研
tags: 
keywords: 
description:
---

记录一次tensorflow模型的C++部署，将推理速度提升10倍。由于涉及到项目，这里只记录部署的思想并只展示部分代码。

首先需要做模型解密，我们参考[TFSecured](https://github.com/manutdzou/TFSecured)根据之前的密钥解码模型。

```
static std::unique_ptr<tensorflow::Session> session;

void CreateSession(const std::string &model_path)
{
	session.reset(tensorflow::NewSession(tensorflow::SessionOptions()));
	tensorflow::GraphDef graph_def;

	const std::string key = "CCCC88QM8IPZDAJDBD6Y2816V0CCCC"; //解析加密模型
	TF_CHECK_OK(tfsecured::GraphDefDecryptAES(model_path, &graph_def, key));

	//TF_CHECK_OK(tensorflow::ReadBinaryProto(tensorflow::Env::Default(), model_path, &graph_def));//解析普通模型
	TF_CHECK_OK(session->Create(graph_def));
}
```

正常inference时一般是先处理数据然后再输入模型,这是个串行的过程，时间是处理数据和模型推理相加。我们可以优化这个过程，并行化数据处理和模型推理。我们使用Provider-Consumer方式。我们首先建立一个线程安全的队列用于存储数据，起一个或多个线程来处理数据放入线程安全的队列中。然后另起一个或多个线程将队列中的数据取出喂给模型预测。

线程安全的队列

```
template <typename T>
class Queue
{
public:

	T pop()
	{
		std::unique_lock<std::mutex> mlock(mutex_);
		while (queue_.empty())
		{
			cond_.wait(mlock);
		}
		auto val = queue_.front();
		queue_.pop();
		return val;
	}

	void pop(T& item)
	{
		std::unique_lock<std::mutex> mlock(mutex_);
		while (queue_.empty())
		{
			cond_.wait(mlock);
		}
		item = queue_.front();
		queue_.pop();
	}

	void push(const T& item)
	{
		std::unique_lock<std::mutex> mlock(mutex_);
		queue_.push(item);
		mlock.unlock();
		cond_.notify_one();
	}

	int size()
	{
		std::unique_lock<std::mutex> mlock(mutex_);
		return queue_.size();
	}

	bool empty()
	{
		std::unique_lock<std::mutex> mlock(mutex_);
		return queue_.empty();
	}

	Queue() = default;
	Queue(const Queue&) = delete;            // disable copying
	Queue& operator=(const Queue&) = delete; // disable assignment

private:
	mutable std::queue<T> queue_;
	std::mutex mutex_;
	std::condition_variable cond_;
};
```

调用tensorflow模型

```
tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ batchsize, 256, 256, 3 }));

float *pTensor = input_tensor.flat<float>().data();
memcpy(pTensor, tmp_atom.data, batchsize * count * sizeof(float));

std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = { { input_name, input_tensor }, };

std::vector<tensorflow::Tensor> outputs;

TF_CHECK_OK(session->Run(inputs, { output_name }, {}, &outputs));
```

至于Consumer和Provider函数涉及到项目不方便贴代码。

只能说相比于使用python的tensorflow，C++下的部署实在快太多了，如果对于时间瓶颈在数据处理的任务，加上一些优化技巧提升5-10倍很容易。

看我写的辛苦求打赏啊！！！有学术讨论和指点请加微信manutdzou,注明

![20](/public/img/pay.jpg)