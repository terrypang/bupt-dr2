from __future__ import division, print_function, absolute_import

import numpy as np
from glob import glob
from tqdm import tqdm
from keras.preprocessing import image
import os
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from utils import utils


def main():

    RAW_DIR = 'data/convert_1024/'
    filenames = glob('{}/*'.format(RAW_DIR))

    bs = 100
    batches = [filenames[i * bs : (i + 1) * bs]
               for i in range(int(len(filenames) / bs) + 1)]

    STDs, MEANs = [], []
    Us, EVs = [], []
    for batch in tqdm(batches):
        images = np.array([image.img_to_array(image.load_img(f, target_size=(512, 512))) for f in batch])
        X = images.reshape(-1, 3)
        STD = np.std(X, axis=0)
        MEAN = np.mean(X, axis=0)
        STDs.append(STD)
        MEANs.append(MEAN)

        X = np.subtract(X, MEAN)
        X = np.divide(X, STD)
        cov = np.dot(X.T, X) / X.shape[0]
        U, S, V = np.linalg.svd(cov)
        ev = np.sqrt(S)
        Us.append(U)
        EVs.append(ev)

    print('STD')
    print(np.mean(STDs, axis=0))
    print('MEAN')
    print(np.mean(MEANs, axis=0))
    print('U')
    print(np.mean(Us, axis=0))
    print('EV')
    print(np.mean(EVs, axis=0))

    raw_images = [os.path.join(RAW_DIR, i) for i in os.listdir(RAW_DIR)]
    names = [os.path.basename(x).split('.')[0] for x in raw_images]
    labels = pd.read_csv('data/trainLabels.csv', index_col=0).loc[names].values.flatten()
    cw = compute_class_weight('balanced', range(5), labels)
    print(cw)

    utils.data_split(datapath=RAW_DIR, test_size=0.1)


if __name__ == '__main__':
    main()