from __future__ import absolute_import, division, print_function

import os
import datetime
import argparse
import importlib
from utils import utils
from keras.optimizers import SGD
from keras.callbacks import CSVLogger, EarlyStopping, LearningRateScheduler
from sklearn.metrics import confusion_matrix
import utils.image as T


TRAIN_DIR = 'data/train'
VALID_DIR = 'data/valid'
TEST_DIR = 'data/test'
MODEL_DIR = 'models'

objective = 'categorical_crossentropy'
# optimizer = RMSprop(lr=1e-4)
optimizer = SGD(lr=1e-4, momentum=0.9)
nb_epoch = 100
batch_size = 32
nb_classes = 5
processes = 4


def finetune(name):
    WEIGHT_DIR = os.path.join('weights', name)
    LOG_DIR = os.path.join('logs', name)

    model = importlib.import_module(MODEL_DIR + '.' + name)
    # model = model.build_finetune(nb_classes=nb_classes, layer_name='merge_11')   # xception
    # model = model.build_finetune(nb_classes=nb_classes, layer_name='merge_3')  # xception
    # model = model.build_finetune(nb_classes=nb_classes, layer_name='merge_13')   # resnet50
    model = model.build_finetune(nb_classes=nb_classes, layer_name='merge_4')   # resnet50
    # model = model.build_finetune(nb_classes=nb_classes, layer_name='mixed8')   # inception
    # model = model.build_finetune(nb_classes=nb_classes, layer_name='mixed2')   # inception
    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    weight_finetune_files = [WEIGHT_DIR + '/' + i for i in os.listdir(WEIGHT_DIR) if 'finetune_' + name in i]
    if weight_finetune_files:
        best_file = sorted(weight_finetune_files)[-1]
    else:
        weight_files = [WEIGHT_DIR + '/' + i for i in os.listdir(WEIGHT_DIR) if name in i]
        best_file = sorted(weight_files)[-1]
    print("LOADING WEIGHTS:")
    print(best_file)
    model.load_weights(best_file)

    if name == 'resnet50':
        preprocess_input = utils.resnet_preprocess_input
    else:
        preprocess_input = utils.ception_preprocess_input
    train_datagen, valid_datagen = utils.setup_generator(processes=processes, preprocess_input=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=model.input_shape[1:3],
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical')
    validation_generator = valid_datagen.flow_from_directory(
        VALID_DIR,
        target_size=model.input_shape[1:3],
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical')

    timestamp = datetime.datetime.now().strftime('[%y-%m-%d_%H:%M:%S]')
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    log_file = os.path.join(LOG_DIR, name + timestamp)
    csv_logger = CSVLogger(log_file + '.csv')
    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)
    checkpoint_path = os.path.join(WEIGHT_DIR, 'finetune_' + name + '_' +
                                   "weights{timestamp}.{epoch:02d}-{val_loss:.4f}.hdf5")
    checkpoint = utils.ModelCheckpoint(checkpoint_path, monitor='val_loss', mode='min',
                                       save_best_only=True, save_weights_only=True, max_num=5)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
    lrate = LearningRateScheduler(utils.step_decay)
    callbacks_list = [checkpoint, csv_logger, early_stopping, lrate]

    model.fit_generator(
        train_generator,
        samples_per_epoch=train_generator.nb_sample,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=validation_generator.nb_sample,
        callbacks=callbacks_list,
        class_weight={0: 1, 1: 3, 2: 2, 3: 4, 4: 5}
        # class_weight={0: 0.27219467, 1: 2.87548097, 2: 1.32743764, 3: 8.0467354, 4: 9.9220339}
    )

    print("confusion matrix")
    gen = T.ImageDataGenerator(preprocessing_function=preprocess_input)
    test_batches = gen.flow_from_directory(VALID_DIR, model.input_shape[1:3], batch_size=batch_size,
                                           shuffle=False, class_mode=None)
    y_pred = model.predict_generator(test_batches, test_batches.nb_sample)
    y_pred = y_pred.argmax(axis=1).astype(int)
    print('Confution Matrix:')
    print(confusion_matrix(test_batches.classes, y_pred))
    kappa = utils.kappa(test_batches.classes, y_pred)
    print('Kappa: {kappa}'.format(kappa=kappa))


def main():
    parser = argparse.ArgumentParser(description='Process training.')
    parser.add_argument('-m', type=str)
    args = parser.parse_args()
    name = args.m
    finetune(name)


if __name__ == '__main__':
    main()