from __future__ import absolute_import, division, print_function

from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization, Lambda, Activation, merge
from keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D


rows, cols, channels = 512, 512, 3
bn_axis = 3
weights_path = 'weights/imagenet/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'


# def preprocess_input(x):
#     return (x / 255. - 0.5) * 2.


def Xception(include_top=False, input_shape=(rows,cols,channels), weights=None):
    img_input = Input(shape=input_shape, name='input')

    # x = Lambda(preprocess_input)(img_input)
    x = Conv2D(32, 3, 3, subsample=(2, 2), bias=False, name='block1_conv1')(img_input)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(64, 3, 3, bias=False, name='block1_conv2')(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)

    residual = Conv2D(128, 1, 1, subsample=(2, 2),
                      border_mode='same', bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, 3, 3, border_mode='same', bias=False, name='block2_sepconv1')(x)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv2D(128, 3, 3, border_mode='same', bias=False, name='block2_sepconv2')(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same', name='block2_pool')(x)
    x = merge([x, residual], mode='sum')

    residual = Conv2D(256, 1, 1, subsample=(2, 2),
                      border_mode='same', bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv2D(256, 3, 3, border_mode='same', bias=False, name='block3_sepconv1')(x)
    x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = SeparableConv2D(256, 3, 3, border_mode='same', bias=False, name='block3_sepconv2')(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same', name='block3_pool')(x)
    x = merge([x, residual], mode='sum')

    residual = Conv2D(728, 1, 1, subsample=(2, 2),
                      border_mode='same', bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block4_sepconv1_act')(x)
    x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name='block4_sepconv1')(x)
    x = BatchNormalization(name='block4_sepconv1_bn')(x)
    x = Activation('relu', name='block4_sepconv2_act')(x)
    x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name='block4_sepconv2')(x)
    x = BatchNormalization(name='block4_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same', name='block4_pool')(x)
    x = merge([x, residual], mode='sum')

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name=prefix + '_sepconv1')(x)
        x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name=prefix + '_sepconv2')(x)
        x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name=prefix + '_sepconv3')(x)
        x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

        x = merge([x, residual], mode='sum')

    residual = Conv2D(1024, 1, 1, subsample=(2, 2),
                      border_mode='same', bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block13_sepconv1_act')(x)
    x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name='block13_sepconv1')(x)
    x = BatchNormalization(name='block13_sepconv1_bn')(x)
    x = Activation('relu', name='block13_sepconv2_act')(x)
    x = SeparableConv2D(1024, 3, 3, border_mode='same', bias=False, name='block13_sepconv2')(x)
    x = BatchNormalization(name='block13_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same', name='block13_pool')(x)
    x = merge([x, residual], mode='sum')

    x = SeparableConv2D(1536, 3, 3, border_mode='same', bias=False, name='block14_sepconv1')(x)
    x = BatchNormalization(name='block14_sepconv1_bn')(x)
    x = Activation('relu', name='block14_sepconv1_act')(x)

    x = SeparableConv2D(2048, 3, 3, border_mode='same', bias=False, name='block14_sepconv2')(x)
    x = BatchNormalization(name='block14_sepconv2_bn')(x)
    x = Activation('relu', name='block14_sepconv2_act')(x)

    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(1000, activation='softmax', name='predictions')(x)

    # Create model
    model = Model(img_input, x)

    # load weights
    if weights == 'imagenet':
        print('Load Model Weights From:')
        print(weights_path)
        model.load_weights(weights_path)

    return model


def build_base(nb_classes, weights=None):
    body = Xception(input_shape=(rows, cols, channels), weights=weights)
    head = body.output
    head = BatchNormalization(axis=3)(head)
    head = GlobalAveragePooling2D(name='avg_pool')(head)
    head = Dense(nb_classes, activation='softmax', name='predictions')(head)

    model = Model(body.input, head)

    return model


def build_model(nb_classes, weights='imagenet'):
    body = Xception(input_shape=(rows, cols, channels), weights=weights)
    for layer in body.layers:
        layer.trainable = False

    head = body.output
    head = BatchNormalization(axis=3)(head)
    head = GlobalAveragePooling2D(name='avg_pool')(head)
    head = Dense(nb_classes, activation='softmax', name='predictions')(head)

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
