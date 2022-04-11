import numpy as np
from keras.utils import plot_model
#from keras.applications.imagenet_utils import _obtain_input_shape
# from keras_applications.imagenet_utils import _obtain_input_shape

from keras.engine.topology import get_source_inputs
from keras.layers import Input, Conv1D, MaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Activation, Dense, Dropout
from keras.models import Model
import keras.backend as K
from utils_1D import block, _depthwise_conv_block, relu6


def ShuffleNet1D_V2(include_top=True,
                 input_tensor=None,
                 scale_factor=1.0,
                 pooling='max',
                 input_shape=(500,1),
                 load_model=None,
                 num_shuffle_units=[1],#[3,7,3],
                 bottleneck_ratio=1,
                 classes=8):
    if K.backend() != 'tensorflow':
        raise RuntimeError('Only tensorflow supported for now')
    name = 'ShuffleNet1D_V2_{}_{}_{}'.format(scale_factor, bottleneck_ratio, "".join([str(x) for x in num_shuffle_units]))
#    input_shape = _obtain_input_shape(input_shape, default_size=(500,1), min_size=(200,1), require_flatten=include_top,
#                                      data_format=K.image_data_format())
#    out_dim_stage_two = {0.5:48, 1:116, 1.5:176, 2:244}
    out_dim_stage_two = {0.5:32, 0.5:32, 1:32, 1:32}

    if pooling not in ['max', 'avg']:
        raise ValueError('Invalid value for pooling')
    if not (float(scale_factor)*4).is_integer():
        raise ValueError('Invalid value for scale_factor, should be x over 4')
    exp = np.insert(np.arange(len(num_shuffle_units), dtype=np.float32), 0, 0)  # [0., 0., 1., 2.]
    out_channels_in_stage = 2**exp
    out_channels_in_stage *= out_dim_stage_two[bottleneck_ratio]  #  calculate output channels for each stage
    out_channels_in_stage[0] = 32#24  # first stage has always 24 output channels
    out_channels_in_stage *= scale_factor
    out_channels_in_stage = out_channels_in_stage.astype(int)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # create shufflenet architecture
    x = Conv1D(filters=out_channels_in_stage[0], kernel_size= 3, padding='same', use_bias=False, strides= 2,
               activation='relu', name='conv1')(img_input)
    x = MaxPool1D(pool_size=3, strides=2, padding='same', name='maxpool1')(x)

    # create stages containing shufflenet units beginning at stage 2
    for stage in range(len(num_shuffle_units)):
        repeat = num_shuffle_units[stage]
        x = block(x, out_channels_in_stage,
                   repeat=repeat,
                   bottleneck_ratio=bottleneck_ratio,
                   stage=stage + 2)

    if bottleneck_ratio < 2:
        k = 16#1024
    else:
        k = 16#2048
    x = Conv1D(k, kernel_size=1, padding='same', strides=1, name='1x1conv5_out', activation='relu')(x)
    x = Conv1D(k, kernel_size=1, padding='same', strides=1, name='1x1conv6_out', activation='relu')(x)
    x = Conv1D(2*k, kernel_size=1, padding='same', strides=1, name='1x1conv7_out', activation='relu')(x)

    if pooling == 'avg':
        x = GlobalAveragePooling1D(name='global_avg_pool')(x)
    elif pooling == 'max':
        x = GlobalMaxPooling1D(name='global_max_pool')(x)

    if include_top:
        x = Dense(2*k, name='fc1')(x)
        x = Activation('relu', name='relu1')(x)  ##Changed from softmax (Huruy)
        x = Dense(k, name='fc2')(x)
        # x = Dropout(0.1)(x)
        x = Activation('relu', name='relu2')(x)  ##Changed from softmax (Huruy)
        x = Dense(classes, name='fc3')(x)  #Dense(classes, name='fc')(x)
        x = Activation('sigmoid', name='sigmoid')(x)  ##Changed from softmax (Huruy)

    if input_tensor:
        inputs = get_source_inputs(input_tensor)

    else:
        inputs = img_input

    model = Model(inputs, x, name=name)

    if load_model:
        model.load_weights('', by_name=True)

    return model

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    model = ShuffleNet1D_V2(include_top=True, input_shape=(500, 1), bottleneck_ratio=1)
    plot_model(model, to_file='ShuffleNet1D_V2_new.png', show_layer_names=True, show_shapes=True)


    pass


