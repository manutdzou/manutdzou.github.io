---
layout: post
title: tensorflow使用LSTM
category: tensorflow
tags: 深度学习
keywords: tf学习
description: tf学习
---

# 使用静态rnn处理时序数据

```
import tensorflow as tf
from tensorflow.contrib import rnn

x=tf.placeholder("float",[None,time_steps,length])
y=tf.placeholder("float",[None,n_classes])

input=tf.unstack(x ,time_steps,1)

lstm_layer=rnn.BasicLSTMCell(num_units,forget_bias=1)
#lstm_layer=rnn.LSTMCell(num_units,use_peepholes=True,forget_bias=1)

outputs,_=rnn.static_rnn(lstm_layer,input,dtype="float32")
```

# 使用动态rnn处理时序数据

```
import tensorflow as tf
from tensorflow.contrib import rnn

x=tf.placeholder("float",[None,time_steps,length])
y=tf.placeholder("float",[None,n_classes])

#lstm_layer=rnn.BasicLSTMCell(num_units,forget_bias=1)
lstm_layer=rnn.LSTMCell(num_units,use_peepholes=True,forget_bias=1)

initial_state = lstm_layer.zero_state(batch_size, dtype=tf.float32)
outputs,_=tf.nn.dynamic_rnn(lstm_layer,x,initial_state=initial_state,time_major=False,dtype="float32")
```

# 使用GRU单元

```
import tensorflow as tf
from tensorflow.contrib import rnn

x=tf.placeholder("float",[None,time_steps,length])
y=tf.placeholder("float",[None,n_classes])

gru_layer=tf.nn.rnn_cell.GRUCell(num_units)

initial_state = gru_layer.zero_state(batch_size, dtype=tf.float32)
outputs,_=tf.nn.dynamic_rnn(gru_layer,x,initial_state=initial_state,time_major=False,dtype="float32")
```

# 使用多个LSTM cell

```
import tensorflow as tf
from tensorflow.contrib import rnn

x=tf.placeholder("float",[None,time_steps,length])
y=tf.placeholder("float",[None,n_classes])

lstm_layers = [rnn.LSTMCell(num_units,forget_bias=1) for num_units in multi_units]

multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_layers)
init_state = multi_rnn_cell.zero_state(batch_size, dtype=tf.float32)
outputs,_=tf.nn.dynamic_rnn(multi_rnn_cell,x,initial_state = init_state,dtype="float32")
```

# 使用双向LSTM

```
import tensorflow as tf
from tensorflow.contrib import rnn

x=tf.placeholder("float",[None,time_steps,length])
y=tf.placeholder("float",[None,n_classes])

fw_lstm_layer=rnn.LSTMCell(num_units,use_peepholes=True,forget_bias=1)
bw_lstm_layer=rnn.LSTMCell(num_units,use_peepholes=True,forget_bias=1)

initial_state_fw = fw_lstm_layer.zero_state(batch_size, dtype=tf.float32)
initial_state_bw = bw_lstm_layer.zero_state(batch_size, dtype=tf.float32)
outputs,_=tf.nn.bidirectional_dynamic_rnn(fw_lstm_layer,bw_lstm_layer,x,initial_state_fw=initial_state_fw,initial_state_bw=initial_state_bw,time_major=False,dtype="float32")
fw_bw=tf.concat(outputs, 2)
```

# 使用conv_lstm

```
import tensorflow as tf
from tensorflow.contrib import rnn

# 5-D tensor
x = tf.placeholder(tf.float32, [None, time_step, n_input, n_input,channel])
y = tf.placeholder(tf.float32, [None, n_classes])

def convlstm(x):
    convlstm_layer= tf.contrib.rnn.ConvLSTMCell(
                conv_ndims=2,
                input_shape=[28, 28, channel],
                output_channels=32,
                kernel_shape=[3, 3],
                use_bias=True,
                skip_connection=False,
                forget_bias=1.0,
                initializers=None,
                name="conv_lstm_cell")
    
    initial_state = convlstm_layer.zero_state(batch_size, dtype=tf.float32)
    outputs,_=tf.nn.dynamic_rnn(convlstm_layer,x,initial_state=initial_state,time_major=False,dtype="float32")
    return outputs

lstm_out = convlstm(x)
```

