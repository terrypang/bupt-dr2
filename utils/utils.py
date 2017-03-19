from __future__ import absolute_import, print_function

import os
import numpy as np
import warnings
from keras.callbacks import Callback
import utils.image as T
from PIL import Image as pil_image
import multiprocessing
import signal
import math
import datetime
import shutil
from sklearn.cross_validation import StratifiedShuffleSplit
import pandas as pd
from tqdm import tqdm
from utils.quadratic_weighted_kappa import quadratic_weighted_kappa
from keras.preprocessing import image

pool = None

class ModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, max_num=5):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.max_num = max_num
        self.file_list = []

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        timestamp = datetime.datetime.now().strftime('[%y-%m-%d_%H:%M:%S]')
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(timestamp=timestamp, epoch=epoch, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                        self.file_list.append(filepath)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)
                self.file_list.append(filepath)

            if len(self.file_list) > self.max_num:
                file = self.file_list[0]
                os.remove(file)
                self.file_list = self.file_list[1:]


def ception_preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def resnet_preprocess_input(x):
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
    # 'RGB'->'BGR'
    x = x[:, :, ::-1]
    return x


# def preprocess_img(img):
#     img = img.astype(np.float32) / 255.0
#     img -= 0.5
#     return img * 2


def resize_image(img, size):
    """
    Resize PIL image

    Resizes image to be square with sidelength size. Pads with black if needed.
    """
    # Resize
    n_x, n_y = img.size
    if n_y > n_x:
        n_y_new = size
        n_x_new = int(size * n_x / n_y + 0.5)
    else:
        n_x_new = size
        n_y_new = int(size * n_y / n_x + 0.5)

    img_res = img.resize((n_x_new, n_y_new), resample=pil_image.BICUBIC)

    # Pad the borders to create a square image
    img_pad = pil_image.new('RGB', (size, size), (128, 128, 128))
    ulc = ((size - n_x_new) // 2, (size - n_y_new) // 2)
    img_pad.paste(img_res, ulc)

    return img_pad


# def load_images(images, input_shape, preprocess_input=resnet_preprocess_input):
#     rows, cols, channels = input_shape
#     count = len(images)
#     data = np.ndarray((count, rows, cols, channels), dtype=np.uint8)
#
#     for i, image_file in enumerate(images):
#         image = pil_image.open(image_file)
#         # image = image.resize((rows, cols))
#         image = resize_image(image, rows)
#         image = np.array(image)
#         data[i] = preprocess_input(image)
#         if i % 500 == 0: print('Processed {} of {}'.format(i, count))
#
#     # data = preprocess_img(data)
#
#     return data


def load_image(path, target_size=(512, 512), preprocess_input=None):
    img = image.load_img(path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def setup_generator(processes=None, preprocess_input=None):
    global pool
    try:
        pool.terminate()
    except:
        pass

    if processes:
        pool = multiprocessing.Pool(processes=processes, initializer=init_worker)
    else:
        num_cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=num_cores)

    train_datagen = T.ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.15,
        fill_mode="constant",
        channel_shift_range=10,
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        preprocessing_function=preprocess_input,
        pool=pool
    )
    valid_datagen = T.ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.05,
        fill_mode="constant",
        channel_shift_range=10,
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        preprocessing_function=preprocess_input,
        pool=pool
    )

    return train_datagen, valid_datagen


def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.3
    epochs_drop = 5.0
    lrate = initial_lrate * math.pow(drop, math.floor(epoch / epochs_drop))
    return lrate

def data_split(datapath, test_size):
    TRAIN_DIR = 'data/train'
    VALID_DIR = 'data/valid'
    raw_images = [os.path.join(datapath, i) for i in os.listdir(datapath)]
    names = [os.path.basename(x).split('.')[0] for x in raw_images]
    labels = pd.read_csv('data/trainLabels.csv', index_col=0).loc[names].values.flatten()

    # labels = []
    # for i in raw_images:
    #     if 'dog' in i:
    #         labels.append(1)
    #     else:
    #         labels.append(0)

    sss = StratifiedShuffleSplit(labels, 1, test_size=test_size, random_state=0)
    for train_index, valid_index in sss:
        if os.path.exists(TRAIN_DIR):
            shutil.rmtree(TRAIN_DIR)
        if os.path.exists(VALID_DIR):
            shutil.rmtree(VALID_DIR)

        for i in range(5):
            os.makedirs(TRAIN_DIR + '/' + str(i))
            os.makedirs(VALID_DIR + '/' + str(i))

        for i in tqdm(train_index):
            src_file = raw_images[i]
            if labels[i] == 0:
                shutil.copyfile(src_file, TRAIN_DIR + '/0/' + os.path.basename(src_file))
            elif labels[i] == 1:
                shutil.copyfile(src_file, TRAIN_DIR + '/1/' + os.path.basename(src_file))
            elif labels[i] == 2:
                shutil.copyfile(src_file, TRAIN_DIR + '/2/' + os.path.basename(src_file))
            elif labels[i] == 3:
                shutil.copyfile(src_file, TRAIN_DIR + '/3/' + os.path.basename(src_file))
            elif labels[i] == 4:
                shutil.copyfile(src_file, TRAIN_DIR + '/4/' + os.path.basename(src_file))
        for i in tqdm(valid_index):
            src_file = raw_images[i]
            if labels[i] == 0:
                shutil.copyfile(src_file, VALID_DIR + '/0/' + os.path.basename(src_file))
            elif labels[i] == 1:
                shutil.copyfile(src_file, VALID_DIR + '/1/' + os.path.basename(src_file))
            elif labels[i] == 2:
                shutil.copyfile(src_file, VALID_DIR + '/2/' + os.path.basename(src_file))
            elif labels[i] == 3:
                shutil.copyfile(src_file, VALID_DIR + '/3/' + os.path.basename(src_file))
            elif labels[i] == 4:
                shutil.copyfile(src_file, VALID_DIR + '/4/' + os.path.basename(src_file))


def kappa(y_true, y_pred):
    min_rating = 0
    max_rating = 4
    # y_true = y_true.argmax(axis=1).astype(int)
    # y_pred = y_pred.argmax(axis=1).astype(int)
    return quadratic_weighted_kappa(y_true, y_pred, min_rating, max_rating)
