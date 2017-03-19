from __future__ import absolute_import, division, print_function

from keras.models import Model
from keras.layers import merge, Input, Flatten, GlobalAveragePooling2D, Lambda
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, AveragePooling2D, Dense, Activation
from keras.layers.normalization import BatchNormalization
import numpy as np


rows, cols, channels = 512, 512, 3
bn_axis = 3
weights_path = 'weights/imagenet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


# def preprocess_input(x):
#     ''' Substract Imagenet mean and reverse channel axis
#         x[:, :, :, 0] -= 103.939
#         x[:, :, :, 1] -= 116.779
#         x[:, :, :, 2] -= 123.68
#         # 'RGB'->'BGR'
#         x = x[:, :, :, ::-1]
#     '''
#     return (x - np.array([103.939, 116.779, 123.68]))[...,::-1]


def identity_block(input_tensor, kernel_size, filters, stage, block):
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      border_mode='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, subsample=strides,
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides,
                             name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x


def ResNet50(include_top=False, input_shape=(rows, cols, channels), weights=None):
    img_input = Input(shape=input_shape)

    # x = Lambda(preprocess_input)(img_input)
    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    if include_top:
        x = AveragePooling2D((7, 7), name='avg_pool')(x)
        x = Flatten()(x)
        x = Dense(1000, activation='softmax', name='fc1000')(x)

    model = Model(img_input, x)

    if weights == 'imagenet':
        print('Load Model Weights From:')
        print(weights_path)
        model.load_weights(weights_path)

    return model


def build_base(weights=None):
    body = ResNet50(input_shape=(rows, cols, channels), weights=weights)
    return body


def build_model(nb_classes, weights='imagenet'):
    body = ResNet50(input_shape=(rows, cols, channels), weights=weights)
    for layer in body.layers:
        layer.trainable = False

    head = body.output
    head = BatchNormalization(axis=3)(head)
    head = GlobalAveragePooling2D(name='avg_pool')(head)
    head = Dense(nb_classes, activation="softmax")(head)

    model = Model(body.input, head)

    return model


def build_finetune(nb_classes, weights=None, layer_name=None):
    model = build_model(nb_classes, weights)
    flag = False
    for layer in model.layers:
        # print(layer.name)
        layer.trainable = flag
        if layer.name == layer_name:
            flag = True

    return model
