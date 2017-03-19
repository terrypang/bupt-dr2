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
from sklearn.metrics import confusion_matrix


VALID_DIR = 'data/valid'
WEIGHT_DIR = os.path.join('weights', 'blend')
LOG_DIR = os.path.join('logs', 'blend')
objective = 'categorical_crossentropy'
nb_epoch = 100
batch_size = 32
nb_classes = 5

# optimizer = Adam(lr=1e-2, momentum=0.9)


def blend(names):
    X_train_list = []
    X_valid_list = []
    for name in names:
        print('Load feature file From:')
        print('features/' + name + '_train.h5')
        with h5py.File('features/' + name + '_train.h5') as hf:
            X_train = hf["train"][:]
            X_valid = hf["valid"][:]
        X_train_list.append(X_train)
        X_valid_list.append(X_valid)

    X_train = np.hstack(tuple(X_train_list))
    X_valid = np.hstack(tuple(X_valid_list))

    with h5py.File('features/' + names[0] + '_train.h5') as hf:
        y_train = hf["train_label"][:]
        y_valid = hf["valid_label"][:]

    model = Sequential([
        Dense(1024, input_shape=X_train.shape[1:], activation="relu"),
        Dropout(0.5),
        Dense(1024, activation="relu"),
        Dropout(0.5),
        Dense(nb_classes, activation="softmax")
    ])
    model.compile(loss=objective, optimizer='adam', metrics=['accuracy'])
    model.summary()
    plot(model, to_file=os.path.join('weights', 'blend' + '.png'), show_shapes=True)

    timestamp = datetime.datetime.now().strftime('[%y-%m-%d_%H:%M:%S]')
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    log_file = os.path.join(LOG_DIR, 'blend' + timestamp)
    csv_logger = CSVLogger(log_file + '.csv')
    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)
    checkpoint_path = os.path.join(WEIGHT_DIR, 'blend' + '_' +
                                   "weights{timestamp}.{epoch:02d}-{val_loss:.4f}.hdf5")
    checkpoint = utils.ModelCheckpoint(checkpoint_path, monitor='val_loss', mode='min',
                                       save_best_only=True, save_weights_only=True, max_num=5)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
    lrate = LearningRateScheduler(utils.step_decay)
    callbacks_list = [checkpoint, csv_logger, early_stopping, lrate]

    model.fit(X_train, y_train, callbacks=callbacks_list,
              validation_data=(X_valid, y_valid),
              batch_size=batch_size,
              nb_epoch=nb_epoch)

    weight_files = [WEIGHT_DIR + '/' + i for i in os.listdir(WEIGHT_DIR) if 'blend' in i]
    best_file = sorted(weight_files)[-1]
    print('Load Model Weights From:')
    print(best_file)
    model.load_weights(best_file)
    #
    # X_test_list = []
    # for name in names:
    #     with h5py.File('features/' + name + '_test.h5') as hf:
    #         X_test = hf["test"][:]
    #     X_test_list.append(X_test)
    # X_test = np.hstack(tuple(X_test_list))
    #
    # y_test = model.predict(X_test)
    # y_test = y_test.clip(min=0.0125, max=0.9875)
    #
    # gen = T.ImageDataGenerator()
    # test_batches = gen.flow_from_directory(TEST_DIR, model.input_shape[1:3], shuffle=False,
    #                                        batch_size=batch_size, class_mode=None)
    #
    # subm = pd.read_csv("submitions/sample_submission.csv")
    # ids = [int(x.split(os.path.sep)[1].split(".")[0]) for x in test_batches.filenames]
    #
    # for i in range(len(ids)):
    #     subm.loc[subm.id == ids[i], "label"] = y_test[:, 1][i]
    #
    # subm.to_csv('submitions/' + 'blend' + timestamp + '.csv', index=False)

    print("confusion matrix")
    gen = T.ImageDataGenerator()
    test_batches = gen.flow_from_directory(VALID_DIR, model.input_shape[1:3], batch_size=batch_size,
                                           shuffle=False, class_mode=None)
    y_pred = model.predict_generator(test_batches, test_batches.nb_sample)
    y_pred = y_pred.argmax(axis=1).astype(int)
    print('Confution Matrix:')
    print(confusion_matrix(test_batches.classes, y_pred))
    kappa = utils.kappa(test_batches.classes, y_pred)
    print('Kappa: {kappa}'.format(kappa=kappa))


def blendAll(names):
    X_train_list = []
    X_valid_list = []
#     for name in names:
#         print('Load feature file From:')
#         print('features/' + name + '_train.h5')
#         with h5py.File('features/' + name + '_train.h5') as hf:
#             X_train = hf["train"][:]
#             X_valid = hf["valid"][:]
#         X_train_list.append(X_train)
#         X_valid_list.append(X_valid)
#
#     X_train = np.hstack(tuple(X_train_list))
#     X_valid = np.hstack(tuple(X_valid_list))
#     print(X_train.shape)
#     print(X_valid.shape)
#
#     X_train = np.vstack((X_train, X_valid))
#     print(X_train.shape)
#
#     with h5py.File('features/' + names[0] + '_train.h5') as hf:
#         y_train = hf["train_label"][:]
#         y_valid = hf["valid_label"][:]
#     y_train = np.vstack((y_train, y_valid))
#
#     model = Sequential([
#         Dense(1024, input_shape=X_train.shape[1:], activation="relu"),
#         Dropout(0.5),
#         Dense(1024, activation="relu"),
#         Dropout(0.5),
#         Dense(nb_classes, activation="softmax")
#     ])
#     model.compile(loss=objective, optimizer='nadam', metrics=['accuracy'])
#
#     timestamp = datetime.datetime.now().strftime('[%y-%m-%d_%H:%M:%S]')
#     if not os.path.exists(LOG_DIR):
#         os.makedirs(LOG_DIR)
#     log_file = os.path.join(LOG_DIR, 'blend' + timestamp)
#     csv_logger = CSVLogger(log_file + '.csv')
#     if not os.path.exists(WEIGHT_DIR):
#         os.makedirs(WEIGHT_DIR)
#     checkpoint_path = os.path.join(WEIGHT_DIR, 'blend' + '_' +
#                                    "weights{timestamp}.{epoch:02d}-{loss:.4f}.hdf5")
#     checkpoint = utils.ModelCheckpoint(checkpoint_path, monitor='loss', mode='min',
#                                        save_best_only=True, save_weights_only=True, max_num=5)
#     early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1, mode='min')
#     lrate = LearningRateScheduler(utils.step_decay)
#     callbacks_list = [checkpoint, csv_logger, early_stopping, lrate]
#
#     model.fit(X_train, y_train, callbacks=callbacks_list,
#               batch_size=batch_size,
#               nb_epoch=nb_epoch)
#
#     weight_files = [WEIGHT_DIR + '/' + i for i in os.listdir(WEIGHT_DIR) if 'blend' in i]
#     best_file = sorted(weight_files)[-1]
#     print('Load Model Weights From:')
#     print(best_file)
#     model.load_weights(best_file)
#
#     X_test_list = []
#     for name in names:
#         with h5py.File('features/' + name + '_test.h5') as hf:
#             X_test = hf["test"][:]
#         X_test_list.append(X_test)
#     X_test = np.hstack(tuple(X_test_list))
#
#     y_test = model.predict(X_test)
#     y_test = y_test.clip(min=0.0125, max=0.9875)
#
#     gen = T.ImageDataGenerator()
#     test_batches = gen.flow_from_directory(TEST_DIR, model.input_shape[1:3], shuffle=False,
#                                            batch_size=batch_size, class_mode=None)
#
#     subm = pd.read_csv("submitions/sample_submission.csv")
#     ids = [int(x.split(os.path.sep)[1].split(".")[0]) for x in test_batches.filenames]
#
#     for i in range(len(ids)):
#         subm.loc[subm.id == ids[i], "label"] = y_test[:, 1][i]
#
#     subm.to_csv('submitions/' + 'blend' + timestamp + '.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description='Process training.')
    parser.add_argument('-m', dest='table', help='', nargs='+')
    parser.add_argument('-a', default=False, type=bool)
    args = parser.parse_args()
    names = args.table
    isAll = args.a
    if isAll:
        blendAll(names)
    else:
        blend(names)


if __name__ == '__main__':
    main()