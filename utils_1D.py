#-*- coding:utf-8 -*-
#'''
# Created on 18-8-14 下午4:39
#
# @Author: Greg Gao(laygin)
#'''
import os
from keras import backend as K
#from keras.applications.imagenet_utils import _obtain_input_shape
#from keras_applications.imagenet_utils import _obtain_input_shape

from keras.models import Model
from keras.engine.topology import get_source_inputs
#from keras.layers import Activation, Add, Concatenate, Conv2D, GlobalMaxPooling2D
from keras.layers import *
from keras.layers import Activation, Add, Concatenate, Conv1D, GlobalMaxPooling1D
#from keras.layers import GlobalAveragePooling2D,Input, Dense
from keras.layers import GlobalAveragePooling1D,Input, Dense
#from keras.layers import MaxPool2D,AveragePooling2D, BatchNormalization, Lambda, DepthwiseConv2D
from keras.layers import MaxPool1D
from keras.layers import AveragePooling1D
from keras.layers import BatchNormalization
from keras.layers import Lambda

from keras.layers import DepthwiseConv2D
from keras.regularizers import l2
#from keras.layers import DepthwiseConv1D
#create 1D depthwise conv ........
def relu6(x):
  return K.relu(x, max_value=6)

def _depthwise_conv_block(
        x, k, padding='same', use_bias=False,#num_filter,
        dilation_rate=1, intermediate_activation=False, name='{}/3x3dwconv',
        strides=1, l2_reg=1e-5):
  # TODO(fchollet): Implement DepthwiseConv1D
  x = Lambda(lambda x: K.expand_dims(x, 1))(x)
  x = DepthwiseConv2D(
      (1, k), padding=padding, use_bias=use_bias,
      dilation_rate=dilation_rate, strides=strides,
      kernel_regularizer=l2(l2_reg))(x)
  x = Lambda(lambda x: K.squeeze(x, 1))(x)
  return x

import numpy as np

def channel_split(x, name=''):
    # equipartition
    in_channles = x.shape.as_list()[-1]
    ip = in_channles // 2
    c_hat = Lambda(lambda z: z[:, :, 0:ip], name='%s/sp%d_slice' % (name, 0))(x)
    c = Lambda(lambda z: z[:, :, ip:], name='%s/sp%d_slice' % (name, 1))(x)
    return c_hat, c

def channel_shuffle(x):
    width, channels = x.shape.as_list()[1:]
    channels_per_split = channels // 2
    x = K.reshape(x, [-1, width, 2, channels_per_split])
    x = K.permute_dimensions(x, (0,1,2,3))
    x = K.reshape(x, [-1, width, channels])
    return x


def shuffle_unit(inputs, out_channels, bottleneck_ratio,strides=2,stage=1,block=1):
    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        raise ValueError('Only channels last supported')

    prefix = 'stage{}/block{}'.format(stage, block)
    bottleneck_channels = int(out_channels * bottleneck_ratio)
    if strides < 2:
        print('1 split')
        c_hat, c = channel_split(inputs, '{}/spl'.format(prefix))
        inputs = c

    x = Conv1D(bottleneck_channels, kernel_size=1, strides=1, padding='same', name='{}/1x1conv_1'.format(prefix))(inputs)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_1'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_1x1conv_1'.format(prefix))(x)
    x = _depthwise_conv_block(x,k=3, strides=strides, padding='same', name='{}/3x3dwconv'.format(prefix))#(x)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv'.format(prefix))(x)
    x = Conv1D(bottleneck_channels, kernel_size=1,strides=1,padding='same', name='{}/1x1conv_2'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_2'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_1x1conv_2'.format(prefix))(x)

    if strides < 2:
        ret = Concatenate(axis=bn_axis, name='{}/concat_1'.format(prefix))([x, c_hat])
    else:
        s2 = _depthwise_conv_block(inputs,k=3, strides=2, padding='same', name='{}/3x3dwconv_2'.format(prefix))#(inputs)
        s2 = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv_2'.format(prefix))(s2)
        s2 = Conv1D(bottleneck_channels, kernel_size=1,strides=1,padding='same', name='{}/1x1_conv_3'.format(prefix))(s2)
        s2 = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_3'.format(prefix))(s2)
        s2 = Activation('relu', name='{}/relu_1x1conv_3'.format(prefix))(s2)
        ret = Concatenate(axis=bn_axis, name='{}/concat_2'.format(prefix))([x, s2])

    ret = Lambda(channel_shuffle, name='{}/channel_shuffle'.format(prefix))(ret)
    print('shuffle unit ended')
    return ret


def block(x, channel_map, bottleneck_ratio, repeat=1, stage=1):
    x = shuffle_unit(x, out_channels=channel_map[stage-1],
                      strides=2,bottleneck_ratio=bottleneck_ratio,stage=stage,block=1)

    for i in range(1, repeat+1):
        x = shuffle_unit(x, out_channels=channel_map[stage-1],strides=1,
                          bottleneck_ratio=bottleneck_ratio,stage=stage, block=(1+i))

    return x
