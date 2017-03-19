from keras.applications.vgg16 import VGG16
import importlib
import utils.image as T
from utils import utils
from models import xception
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.layers.core import Lambda
from keras.models import Sequential
import keras.backend as K
import tensorflow as tf
import numpy as np
import sys
import cv2
import csv
import os
import argparse
import shutil
from glob import glob
from sklearn.metrics import confusion_matrix

name = 'resnet50'
MODEL_DIR = 'models'
VALID_DIR = 'data/valid'
batch_size = 32
nb_classes = 5
K.set_learning_phase(0)

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(path):
    img_path = path
    img = image.load_img(img_path, target_size=(512, 512))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def grad_cam(input_model, image, category_index, layer_name):
    model = Sequential()
    model.add(input_model)

    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    model.add(Lambda(target_layer,
                     output_shape = target_category_loss_output_shape))

    loss = K.sum(model.layers[-1].output)

    # print([l.name for l in model.layers[0].layers])
    conv_output =  [l for l in model.layers[0].layers if l.name == layer_name][0].output
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (512, 512))
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)
    # print(image)
    # image /= 2.
    # image += 0.5
    # image *= 255.
    # image -= np.min(image)
    # image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam)

def getResult(y_pred):
    if y_pred == 0:
        return 'No DR'
    elif y_pred == 1:
        return 'Mild'
    elif y_pred ==2:
        return 'Moderate'
    elif y_pred == 3:
        return 'Severe'
    elif y_pred == 4:
        return 'Proliferative DR'


def main(path):
    model = importlib.import_module(MODEL_DIR + '.' + name)
    model = model.build_model(nb_classes=nb_classes)

    WEIGHT_DIR = os.path.join('weights', name)
    weight_finetune_files = [WEIGHT_DIR + '/' + i for i in os.listdir(WEIGHT_DIR) if 'finetune_' + name in i]
    best_file = sorted(weight_finetune_files)[-1]
    print("LOADING WEIGHTS:")
    print(best_file)
    model.load_weights(best_file)

    # print("confusion matrix")
    # gen = T.ImageDataGenerator(preprocessing_function=utils.resnet_preprocess_input)
    # test_batches = gen.flow_from_directory(VALID_DIR, model.input_shape[1:3], batch_size=batch_size,
    #                                        shuffle=False, class_mode=None)
    # y_pred = model.predict_generator(test_batches, test_batches.nb_sample)
    # y_pred = y_pred.argmax(axis=1).astype(int)
    # print('Confution Matrix:')
    # print(confusion_matrix(test_batches.classes, y_pred))
    # kappa = utils.kappa(test_batches.classes, y_pred)
    # print('Kappa: {kappa}'.format(kappa=kappa))

    base_path = '/home/terrypang/Desktop/DR-Check/data'
    workspace = os.path.join(base_path, 'workspace_' + path)
    cam_path = os.path.join(base_path, 'cam_' + path)
    if os.path.exists(cam_path):
        shutil.rmtree(cam_path)
    os.mkdir(cam_path)
    with open(base_path + '/' + path + '_cam.csv', 'w', newline='') as cam_f:
        fieldnames = ['file', 'value', 'result']
        writer = csv.DictWriter(cam_f, fieldnames=fieldnames)
        writer.writeheader()
        with open(base_path + '/' + path + '_blend.csv', newline='') as blend_f:
            reader = csv.DictReader(blend_f)
            field = reader.fieldnames
            for rowDict in reader:
                filename = rowDict[field[0]]
                value = float(rowDict[field[1]])
                value = np.clip(value, 0, 4)
                value = np.round(value).astype(int)

                preprocessed_input = load_image(os.path.join(workspace, filename))
                print('File: {file}'.format(file=filename))
                predicted = model.predict(preprocessed_input)[0]
                cam = grad_cam(model, preprocessed_input, value, 'activation_49')
                cv2.imwrite(os.path.join(cam_path, filename), cam)
                writer.writerow({'file': filename, 'value': predicted[value], 'result': getResult(value)})


    # base_path = '/home/terrypang/Desktop/DR-Check/data'
    # workspace = os.path.join(base_path, 'workspace_' + path)
    # cam_path = os.path.join(base_path, 'cam_' + path)
    # if os.path.exists(cam_path):
    #     shutil.rmtree(cam_path)
    # os.mkdir(cam_path)
    # fs = sorted(glob('{}/*'.format(workspace)))
    #
    # with open(base_path + '/' + path + '_cam.csv', 'w', newline='') as f:
    #     fieldnames = ['file', 'value', 'result']
    #     writer = csv.DictWriter(f, fieldnames=fieldnames)
    #     writer.writeheader()
    #
    #     for f in fs:
    #         preprocessed_input = load_image(f)
    #         print('File: {file}'.format(file=f))
    #         predicted = model.predict(preprocessed_input)[0]
    #
    #         top1_class = np.argmax(predicted)
    #         top1_value = predicted[top1_class]
    #         print('Top1: {top1_class}, Value: {top1_value}'.format(top1_class=top1_class, top1_value=top1_value))
    #         cam = grad_cam(model, preprocessed_input, top1_class, 'activation_49')
    #         top1_name = os.path.basename(f).split('.')[0] + '_top1_' + str(top1_class) + '.jpg'
    #         cv2.imwrite(cam_path + '/' + top1_name, cam)
    #         writer.writerow({'file': top1_name, 'value': top1_value, 'result': getResult(top1_class)})
    #
    #         predicted[top1_class] = -1
    #         top2_class = np.argmax(predicted)
    #         top2_value = predicted[top2_class]
    #         print('Top2: {top2_class}, Value: {top2_value}'.format(top2_class=top2_class, top2_value=top2_value))
    #         cam = grad_cam(model, preprocessed_input, top2_class, 'activation_49')
    #         top2_name = os.path.basename(f).split('.')[0] + '_top2_' + str(top2_class) + '.jpg'
    #         cv2.imwrite(cam_path + '/' + top2_name, cam)
    #         writer.writerow({'file': top2_name, 'value': top2_value, 'result': getResult(top2_class)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process training.')
    parser.add_argument('-p', type=str)
    args = parser.parse_args()
    path = args.p
    main(path)
    # test()