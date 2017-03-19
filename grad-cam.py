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
import os
from sklearn.metrics import confusion_matrix

MODEL_DIR = 'models'
VALID_DIR = 'data/valid'
name = 'resnet50'
batch_size = 32
nb_classes = 5
K.set_learning_phase(0)

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    # return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)
    # return x / (K.abs(K.mean(x)) + 1e-5)
    return tf.nn.l2_normalize(x, dim=0, epsilon=5e-10)

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

    # x = input_model.output
    # array([[8.01009664e-05, 6.64381834e-04, 9.68862474e-01,
    #         3.03799510e-02, 1.30711987e-05]])
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    model.add(Lambda(target_layer,
                     output_shape = target_category_loss_output_shape))

    # target_layer.output is: array([[ 0.        ,  0.        ,  0.96886247,  0.        ,  0.        ]])
    # loss = 0.96886247
    loss = K.sum(model.layers[-1].output)

    # print([l.name for l in model.layers[0].layers])
    # shape: (1, 16, 16, 2048)
    conv_output =  [l for l in model.layers[0].layers if l.name == layer_name][0].output

    # Returns the gradients of `conv_output` (list of tensor variables) with regard to `loss`.
    # Returns a list of `sum(dy/dx)` for each x in `xs`. There is only one image, the index is [0].
    # shape: (1, 16, 16, 2048)
    # do L2 normalization
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])

    # image to [model.layers[0].input], and get [conv_output, grads]
    output, grads_val = gradient_function([image])
    # change shape to (16, 16, 2048)
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    # weight is the mean gredient value of each channel
    # shpae: (2048,)
    weights = np.mean(grads_val, axis = (0, 1))
    # shpae: (16, 16)
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    # i is the channel index, w is the weight of this chanel.
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    # resize cam to 512x512
    cam = cv2.resize(cam, (512, 512))
    # make all pixel value >= 0
    cam = np.maximum(cam, 0)
    # cam = cam / np.max(cam)

    # Return to origin image [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)
    # print(image)
    # image /= 2.
    # image += 0.5
    # image *= 255.
    # image -= np.min(image)
    # image = np.minimum(image, 255)

    # get ColorMaps
    cam = cv2.applyColorMap(np.uint8(cam), cv2.COLORMAP_JET)
    # add two image
    cam = np.float32(cam) + np.float32(image)
    # color adjustment
    cam = 255 * cam / np.max(cam)
    # make cam pixel value in [0, 255]
    return np.uint8(cam)


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

# filename = 'data/valid/3/5789_left.jpg'
filename = 'data/train/2/5789_right.jpg'

preprocessed_input = load_image(filename)

predicted = model.predict(preprocessed_input)
predicted_class = np.argmax(predicted)
print('Predict: {pred}'.format(pred=predicted))
print('Predict class: {pred}'.format(pred=predicted_class))
cam  = grad_cam(model, preprocessed_input, predicted_class, 'activation_49')
cv2.imwrite("cam-49.jpg", cam)
