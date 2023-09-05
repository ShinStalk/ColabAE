""" Wrapper functions for TensorFlow layers.

Author: Charles R. Qi
Date: November 2017
"""

import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Conv2DTranspose, Dense, BatchNormalization, Activation, Layer, Dropout


def _variable_on_cpu(name, shape, initializer, use_fp16=False):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device("/cpu:0"):
    dtype = tf.float16 if use_fp16 else tf.float32
    var = tf.compat.v1.get_variable(name, shape, initializer=initializer, dtype=dtype, use_resource=False)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    use_xavier: bool, whether to use xavier initializer

  Returns:
    Variable Tensor
  """
  if use_xavier:
    initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
  else:
    initializer = tf.compat.v1.truncated_normal_initializer(stddev=stddev)
  var = _variable_on_cpu(name, shape, initializer)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.compat.v1.add_to_collection('losses', weight_decay)
  return var


def conv1d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=1,
           padding='SAME',
           data_format='NHWC',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=None,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None):
  """ 1D convolution with non-linear operation.

  Args:
    inputs: 3-D tensor variable BxLxC
    num_output_channels: int
    kernel_size: int
    scope: string
    stride: int
    padding: 'SAME' or 'VALID'
    data_format: 'NHWC' or 'NCHW'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]

  Returns:
    Variable tensor
  """
  with tf.compat.v1.variable_scope(scope) as sc:
    assert(data_format=='NHWC' or data_format=='NCHW')
    if data_format == 'NHWC':
      num_in_channels = inputs.shape[-1]
    elif data_format=='NCHW':
      num_in_channels = inputs.shape[1]
    kernel_shape = [kernel_size,
                    num_in_channels, num_output_channels]
    kernel = _variable_with_weight_decay('weights',
                                         shape=kernel_shape,
                                         use_xavier=use_xavier,
                                         stddev=stddev,
                                         wd=weight_decay)
    outputs = tf.nn.conv1d(input=inputs, filters=kernel,
                           stride=stride,
                           padding=padding,
                           data_format=data_format)
    biases = _variable_on_cpu('biases', [num_output_channels],
                              tf.compat.v1.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases, data_format=data_format)

    if bn:
      outputs = batch_norm_for_conv1d(outputs, bn_decay=bn_decay)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs


def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           data_format='NHWC',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=None,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None):
  """ 2D convolution with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    data_format: 'NHWC' or 'NCHW'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]

  Returns:
    Variable tensor
  """
  with tf.compat.v1.variable_scope(scope) as sc:
      kernel_h, kernel_w = kernel_size
      assert(data_format=='NHWC' or data_format=='NCHW')
      if data_format == 'NHWC':
        num_in_channels = inputs.shape[-1]
      elif data_format=='NCHW':
        num_in_channels = inputs.shape[-1]
      kernel_shape = [kernel_h, kernel_w,
                      num_in_channels, num_output_channels]

      # Selecting the initializer based on use_xavier
      if use_xavier:
          initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
      else:
          initializer = tf.compat.v1.truncated_normal_initializer(stddev=stddev)

      # Creating kernel variable directly
      kernel = tf.Variable(initializer(shape=kernel_shape), name='weights')

      stride_h, stride_w = stride
      outputs = tf.nn.conv2d(input=inputs, filters=kernel,
                             strides=[1, stride_h, stride_w, 1],
                             padding=padding,
                             data_format=data_format)
      biases = _variable_on_cpu('biases', [num_output_channels],
                                tf.compat.v1.constant_initializer(0.0))
      outputs = outputs + biases

      if bn:
        outputs = batch_norm_for_conv2d(outputs, bn_decay=bn_decay)

      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return outputs

class conv2d_v2(Layer):
    def __init__(self,
                 num_output_channels,
                 kernel_size,
                 scope,
                 padding='SAME',
                 stride=[1, 1],
                 data_format='NHWC',
                 use_xavier=True,
                 stddev=1e-3,
                 activation_fn=tf.nn.relu,
                 bn=False,
                 bn_decay=None, **kwargs):
        super(conv2d_v2, self).__init__(name=scope, **kwargs)
        self.num_output_channels = num_output_channels
        self.kernel_size = kernel_size
        self.scope = scope
        self.stride = stride
        self.padding = padding
        self.data_format = data_format
        self.use_xavier = use_xavier
        self.stddev = stddev
        self.activation_fn = activation_fn
        self.bn = bn
        self.bn_decay = bn_decay
        self.kernel = None
        self.biases = None
        if self.bn:
            self.bn_layer = BatchNormalization(momentum=0.99, scale=True, center=True)

    def build(self, input_shape):
        if self.data_format == 'NHWC':
            num_in_channels = input_shape[-1]
        elif self.data_format == 'NCHW':
            num_in_channels = input_shape[1]

        kernel_shape = [self.kernel_size[0], self.kernel_size[1], num_in_channels, self.num_output_channels]

        # Selecting the initializer based on use_xavier
        initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform") if self.use_xavier else tf.keras.initializers.TruncatedNormal(stddev=self.stddev)

        # Creating kernel and biases
        self.kernel = self.add_weight(name='weights_'+self.scope, shape=kernel_shape, initializer=initializer, trainable=True)
        self.biases = self.add_weight(name='biases_'+self.scope, shape=[self.num_output_channels], initializer=tf.keras.initializers.Zeros(), trainable=True)

        super(conv2d_v2, self).build(input_shape)

    def call(self, inputs, **kwargs):
        stride_h, stride_w = self.stride
        outputs = tf.nn.conv2d(input=inputs, filters=self.kernel, strides=[1, stride_h, stride_w, 1], padding=self.padding, data_format=self.data_format)
        outputs = outputs + self.biases

        if self.bn:
           outputs = self.bn_layer(outputs)

        if self.activation_fn is not None:
            outputs = self.activation_fn(outputs)
        return outputs


def conv2d_transpose(inputs,
                     num_output_channels,
                     kernel_size,
                     scope,
                     stride=[1, 1],
                     padding='SAME',
                     use_xavier=True,
                     stddev=1e-3,
                     weight_decay=None,
                     activation_fn=tf.nn.relu,
                     bn=False,
                     bn_decay=None,
                     data_format='NHCW'):
  """ 2D convolution transpose with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]

  Returns:
    Variable tensor

  Note: conv2d(conv2d_transpose(a, num_out, ksize, stride), a.shape[-1], ksize, stride) == a
  """
  with tf.compat.v1.variable_scope(scope) as sc:
      kernel_h, kernel_w = kernel_size
      num_in_channels = inputs.shape[-1]
      kernel_shape = [kernel_h, kernel_w,
                      num_output_channels, num_in_channels] # reversed to conv2d

      # Selecting the initializer based on use_xavier
      if use_xavier:
          initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
      else:
          initializer = tf.compat.v1.truncated_normal_initializer(stddev=stddev)

      # Creating kernel variable directly
      kernel = tf.Variable(initializer(shape=kernel_shape), name='weights')

      stride_h, stride_w = stride
      
      # from slim.convolution2d_transpose
      def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
          dim_size *= stride_size

          if padding == 'VALID' and dim_size is not None:
            dim_size += max(kernel_size - stride_size, 0)
          return dim_size

      # caculate output shape
      batch_size = inputs.shape[0]
      height = inputs.shape[1]
      width = inputs.shape[2]
      out_height = get_deconv_dim(height, stride_h, kernel_h, padding)
      out_width = get_deconv_dim(width, stride_w, kernel_w, padding)
      output_shape = [batch_size, out_height, out_width, num_output_channels]

      outputs = tf.nn.conv2d_transpose(inputs, kernel, output_shape,
                             [1, stride_h, stride_w, 1],
                             padding=padding)
      biases = _variable_on_cpu('biases', [num_output_channels],
                                tf.compat.v1.constant_initializer(0.0))
      outputs = tf.nn.bias_add(outputs, biases)

      if bn:
        outputs = batch_norm_for_conv2d(outputs, bn_decay=bn_decay)

      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return outputs


class conv2d_transpose_v2(Layer):
    def __init__(self,
                 num_output_channels,
                 kernel_size,
                 scope,
                 stride=[1, 1],
                 padding='SAME',
                 use_xavier=True,
                 stddev=1e-3,
                 activation_fn=tf.keras.activations.relu,
                 bn=False,
                 bn_decay=None, **kwargs):

        super(conv2d_transpose_v2, self).__init__(name=scope, **kwargs)
        self.num_output_channels = num_output_channels
        self.kernel_size = kernel_size
        self.scope = scope
        self.stride = stride
        self.padding = padding
        self.use_xavier = use_xavier
        self.stddev = stddev
        self.activation_fn = activation_fn
        self.bn = bn
        self.bn_decay = bn_decay
        self.kernel = None
        self.biases = None
        if self.bn:
            self.bn_layer = BatchNormalization(momentum=0.99, scale=True, center=True)

    def build(self, input_shape):
        kernel_h, kernel_w = self.kernel_size
        num_in_channels = input_shape[-1]
        kernel_shape = [kernel_h, kernel_w, self.num_output_channels, num_in_channels]  # reversed to conv2d

        # Selecting the initializer based on use_xavier
        initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform") if self.use_xavier else tf.keras.initializers.TruncatedNormal(
            stddev=self.stddev)

        # Creating kernel and biases
        self.kernel = self.add_weight(name='weights_' + self.scope, shape=kernel_shape, initializer=initializer, trainable=True)
        self.biases = self.add_weight(name='biases_' + self.scope, shape=[self.num_output_channels], initializer=tf.keras.initializers.Zeros(), trainable=True)

        super(conv2d_transpose_v2, self).build(input_shape)

    def call(self, inputs, **kwargs):
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride

        # from slim.convolution2d_transpose
        def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
            dim_size *= stride_size

            if padding == 'VALID' and dim_size is not None:
                dim_size += max(kernel_size - stride_size, 0)
            return dim_size

        # caculate output shape
        batch_size = inputs.shape[0]
        height = inputs.shape[1]
        width = inputs.shape[2]
        out_height = get_deconv_dim(height, stride_h, kernel_h, self.padding)
        out_width = get_deconv_dim(width, stride_w, kernel_w, self.padding)
        output_shape = [batch_size, out_height, out_width, self.num_output_channels]

        #TODO replace with Conv2DTranspose Keras layer (see: https://stackoverflow.com/questions/55825822/why-does-tf-keras-layers-conv2dtranspose-need-no-output-shape-compared-to-tf-nn)
        outputs = tf.nn.conv2d_transpose(input=inputs, filters=self.kernel, output_shape=output_shape, strides=[1, stride_h, stride_w, 1], padding=self.padding)
        outputs = outputs + self.biases

        if self.bn:
           outputs = self.bn_layer(outputs)

        if self.activation_fn is not None:
            outputs = self.activation_fn(outputs)
        return outputs


def conv3d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=None,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None):
  """ 3D convolution with non-linear operation.

  Args:
    inputs: 5-D tensor variable BxDxHxWxC
    num_output_channels: int
    kernel_size: a list of 3 ints
    scope: string
    stride: a list of 3 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]

  Returns:
    Variable tensor
  """
  with tf.compat.v1.variable_scope(scope) as sc:
    kernel_d, kernel_h, kernel_w = kernel_size
    num_in_channels = inputs.shape[-1]
    kernel_shape = [kernel_d, kernel_h, kernel_w,
                    num_in_channels, num_output_channels]
    kernel = _variable_with_weight_decay('weights',
                                         shape=kernel_shape,
                                         use_xavier=use_xavier,
                                         stddev=stddev,
                                         wd=weight_decay)
    stride_d, stride_h, stride_w = stride
    outputs = tf.nn.conv3d(inputs, kernel,
                           [1, stride_d, stride_h, stride_w, 1],
                           padding=padding)
    biases = _variable_on_cpu('biases', [num_output_channels],
                              tf.compat.v1.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases)
    
    if bn:
      outputs = batch_norm_for_conv3d(outputs, bn_decay=bn_decay)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs

def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=None,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None):
  """ Fully connected layer with non-linear operation.
  
  Args:
    inputs: 2-D tensor BxN
    num_outputs: int
  
  Returns:
    Variable tensor of size B x num_outputs.
  """
  with tf.compat.v1.variable_scope(scope) as sc:
    num_input_units = inputs.shape[-1]
    weights = _variable_with_weight_decay('weights',
                                          shape=[num_input_units, num_outputs],
                                          use_xavier=use_xavier,
                                          stddev=stddev,
                                          wd=weight_decay)
    outputs = tf.matmul(inputs, weights)
    biases = _variable_on_cpu('biases', [num_outputs],
                             tf.compat.v1.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases)
     
    if bn:
      outputs = batch_norm_for_fc(outputs, bn_decay)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs


class fully_connected_v2(Layer):
    def __init__(self, num_outputs, use_xavier=True, stddev=1e-3, activation_fn=tf.nn.relu, bn=False, bn_decay=None, **kwargs):
        super(fully_connected_v2, self).__init__(**kwargs)
        self.num_outputs = num_outputs
        self.use_xavier = use_xavier
        self.stddev = stddev
        self.activation_fn = activation_fn
        self.bn = bn
        self.bn_decay = bn_decay
        self._weights = None
        self._biases = None
        if self.bn:
            self.bn_layer = BatchNormalization(momentum=0.99, scale=True, center=True)

    def build(self, input_shape):
        print(f'build fc input_shape: {input_shape}')
        num_input_units = input_shape[-1]

        # Initializing the weights
        if self.use_xavier:
            initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        else:
            initializer = tf.keras.initializers.TruncatedNormal(stddev=self.stddev)

        self._weights = self.add_weight(name='weights', shape=[num_input_units, self.num_outputs], initializer=initializer)
        self._biases = self.add_weight(name='biases', shape=[self.num_outputs], initializer='zeros')

        super(fully_connected_v2, self).build(input_shape)

    def call(self, inputs, **kwargs):
        print(f'call fc inputs.shape: {inputs.shape}')
        outputs = tf.matmul(inputs, self._weights)
        outputs = outputs + self._biases

        # Applying the specific batch normalization function if required
        if self.bn:
            outputs = self.bn_layer(outputs)

        if self.activation_fn is not None:
            outputs = self.activation_fn(outputs)

        return outputs


def max_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
  """ 2D max pooling.

  Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 2 ints
    stride: a list of 2 ints
  
  Returns:
    Variable tensor
  """
  with tf.compat.v1.variable_scope(scope) as sc:
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    outputs = tf.nn.max_pool2d(input=inputs,
                             ksize=[1, kernel_h, kernel_w, 1],
                             strides=[1, stride_h, stride_w, 1],
                             padding=padding,
                             name=sc.name)
    return outputs

def avg_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
  """ 2D avg pooling.

  Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 2 ints
    stride: a list of 2 ints
  
  Returns:
    Variable tensor
  """
  with tf.compat.v1.variable_scope(scope) as sc:
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    outputs = tf.nn.avg_pool2d(input=inputs,
                             ksize=[1, kernel_h, kernel_w, 1],
                             strides=[1, stride_h, stride_w, 1],
                             padding=padding,
                             name=sc.name)
    return outputs


def max_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID'):
  """ 3D max pooling.

  Args:
    inputs: 5-D tensor BxDxHxWxC
    kernel_size: a list of 3 ints
    stride: a list of 3 ints
  
  Returns:
    Variable tensor
  """
  with tf.compat.v1.variable_scope(scope) as sc:
    kernel_d, kernel_h, kernel_w = kernel_size
    stride_d, stride_h, stride_w = stride
    outputs = tf.nn.max_pool3d(inputs,
                               ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                               strides=[1, stride_d, stride_h, stride_w, 1],
                               padding=padding,
                               name=sc.name)
    return outputs

def avg_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID'):
  """ 3D avg pooling.

  Args:
    inputs: 5-D tensor BxDxHxWxC
    kernel_size: a list of 3 ints
    stride: a list of 3 ints
  
  Returns:
    Variable tensor
  """
  with tf.compat.v1.variable_scope(scope) as sc:
    kernel_d, kernel_h, kernel_w = kernel_size
    stride_d, stride_h, stride_w = stride
    outputs = tf.nn.avg_pool3d(inputs,
                               ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                               strides=[1, stride_d, stride_h, stride_w, 1],
                               padding=padding,
                               name=sc.name)
    return outputs


def batch_norm_template(inputs, bn_decay):
  """ Batch normalization on convolutional maps and beyond...
  Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
  
  Args:
      inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
      scope:         string, variable scope
      moments_dims:  a list of ints, indicating dimensions for moments calculation
      bn_decay:      float or float tensor variable, controling moving average weight
      data_format:   'NHWC' or 'NCHW'
  Return:
      normed:        batch-normalized maps
  """
  bn_decay = bn_decay if bn_decay is not None else 0.99
  return BatchNormalization(momentum=bn_decay, scale=True, center=True)(inputs)


def batch_norm_for_fc(inputs, bn_decay):
  """ Batch normalization on FC data.
  
  Args:
      inputs:      Tensor, 2D BxC input
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, bn_decay)


def batch_norm_for_conv1d(inputs, bn_decay):
  """ Batch normalization on 1D convolutional maps.
  
  Args:
      inputs:      Tensor, 3D BLC input maps
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
      data_format: 'NHWC' or 'NCHW'
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, bn_decay)



  
def batch_norm_for_conv2d(inputs, bn_decay):
  """ Batch normalization on 2D convolutional maps.
  
  Args:
      inputs:      Tensor, 4D BHWC input maps
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
      data_format: 'NHWC' or 'NCHW'
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, bn_decay)


def batch_norm_for_conv3d(inputs, bn_decay):
  """ Batch normalization on 3D convolutional maps.
  
  Args:
      inputs:      Tensor, 5D BDHWC input maps
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, bn_decay)


def dropout(inputs,
            is_training,
            scope,
            keep_prob=0.5,
            noise_shape=None):
  """ Dropout layer.

  Args:
    inputs: tensor
    is_training: boolean tf.Variable
    scope: string
    keep_prob: float in [0,1]
    noise_shape: list of ints

  Returns:
    tensor variable
  """
  with tf.compat.v1.variable_scope(scope) as sc:
    outputs = tf.cond(pred=is_training,
                      true_fn=lambda: tf.nn.dropout(inputs, noise_shape=noise_shape, rate=1 - (keep_prob)),
                      false_fn=lambda: inputs)
    return outputs


def tf2_conv2d(inputs, num_output_channels, kernel_size):
    conv = Conv2D(num_output_channels, kernel_size)(inputs)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    return conv


def tf2_conv2d_transpose(inputs, num_output_channels, kernel_size, stride=(1,1)):
    conv = Conv2DTranspose(num_output_channels, kernel_size, strides=stride)(inputs)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    return conv


def tf2_fully_connected(inputs, num_outputs):
    dense_layer = Dense(num_outputs)(inputs)
    dense_layer = BatchNormalization()(dense_layer)
    dense_layer = Activation('relu')(dense_layer)

    return dense_layer
