from __future__ import absolute_import, division, print_function

import os
import h5py
import datetime
import argparse
from utils import utils
import utils.image as T
import numpy as np
import pandas as pd

from keras.models import Model, model_from_json
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.callbacks import CSVLogger, EarlyStopping, LearningRateScheduler
from keras.utils.visualize_util import plot
from keras.optimizers import Adam


def getFeatures(name):
    TRAIN_DIR = 'data/train'
    VALID_DIR = 'data/valid'
    WEIGHT_DIR = os.path.join('weights', name)
    batch_size = 32

    base_model = model_from_json(open(os.path.join('weights', name + '.json')).read())
    weight_files = [WEIGHT_DIR + '/' + i for i in os.listdir(WEIGHT_DIR) if 'finetune_' + name in i]
    best_file = sorted(weight_files)[-1]
    print("LOADING WEIGHTS:")
    print(best_file)
    base_model.load_weights(best_file)
    model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)

    if name == 'resnet50':
        preprocess_input = utils.resnet_preprocess_input
    else:
        preprocess_input = utils.ception_preprocess_input
    gen = T.ImageDataGenerator(preprocessing_function=preprocess_input)
    train_batches = gen.flow_from_directory(TRAIN_DIR, model.input_shape[1:3],
                                            shuffle=False, batch_size=batch_size)
    valid_batches = gen.flow_from_directory(VALID_DIR, model.input_shape[1:3],
                                            shuffle=False, batch_size=batch_size)


    print('Get train features for model {name}'.format(name=name))
    train_feature = model.predict_generator(train_batches, train_batches.nb_sample)
    print(train_feature.shape)
    print('Get valid features for model {name}'.format(name=name))
    valid_feature = model.predict_generator(valid_batches, valid_batches.nb_sample)
    print(valid_feature.shape)

    with h5py.File('features/' + name + '_train.h5') as hf:
        hf.create_dataset("train", data=train_feature)
        hf.create_dataset("valid", data=valid_feature)
        hf.create_dataset("train_label", data=to_categorical(train_batches.classes))
        hf.create_dataset("valid_label", data=to_categorical(valid_batches.classes))


def main():
    parser = argparse.ArgumentParser(description='Process training.')
    parser.add_argument('-m', dest='table', help='', nargs='+')
    args = parser.parse_args()
    names = args.table
    for name in names:
        getFeatures(name)


if __name__ == '__main__':
    main()