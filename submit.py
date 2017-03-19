from __future__ import absolute_import, division, print_function

import os
import argparse
from utils import utils
import pandas as pd
import datetime
import utils.image as T
from keras.models import model_from_json


def submit(name):
    TEST_DIR = 'data/test'
    WEIGHT_DIR = os.path.join('weights', name)
    batch_size = 32

    model = model_from_json(open(os.path.join('weights', name + '.json')).read())
    weight_files = [WEIGHT_DIR + '/' + i for i in os.listdir(WEIGHT_DIR) if 'finetune_' + name in i]
    best_file = sorted(weight_files)[-1]
    print("LOADING WEIGHTS ...")
    model.load_weights(best_file)

    if name == 'resnet50':
        preprocess_input = utils.resnet_preprocess_input
    else:
        preprocess_input = utils.ception_preprocess_input
    gen = T.ImageDataGenerator(preprocessing_function=preprocess_input)
    test_batches = gen.flow_from_directory(TEST_DIR, model.input_shape[1:3], batch_size=batch_size,
                                           shuffle=False, class_mode=None)
    y_test = model.predict_generator(test_batches, test_batches.nb_sample)
    y_test = y_test.clip(min=0.0125, max=0.9875)

    timestamp = datetime.datetime.now().strftime('[%y-%m-%d_%H:%M:%S]')
    subm = pd.read_csv("submitions/sample_submission.csv")
    ids = [int(x.split(os.path.sep)[1].split(".")[0]) for x in test_batches.filenames]

    for i in range(len(ids)):
        subm.loc[subm.id == ids[i], "label"] = y_test[:, 1][i]

    subm.to_csv('submitions/' + name + timestamp + '.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description='Process training.')
    parser.add_argument('-m', type=str)
    args = parser.parse_args()
    name = args.m
    submit(name)


if __name__=='__main__':
    main()