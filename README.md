### 0. Environment
* Intel I7-7655K, 32G DDR4, Nvidia Titan X
* Xubuntu 16.04.1, CUDA 8.0, cudnn 5.1

### 1. Install the following components
* python 2.7
* numpy
* scipy
* pandas
* matplotlib
* scikit-learn
* pillow
* tensorflow
* keras
* h5py
* pydot

### 2. Folder structure
````bash
├── data
│   ├── raw
│   ├── test
│       └── test
├── features
├── logs
├── models
├── submitions
├── utils
├── weights
├── submit.py
└── train.py
````

### 2. Download model weight files to 'weights/imagenet'
* https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5
* https://github.com/fchollet/deep-learning-models/releases/download/v0.2/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
* https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5

### 3. Download kaggle compitition data
* Download from: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data
* Extract all images from train.zip to 'data/raw'
* Extract all images from test.zip to 'data/test/test'
* Copy 'sample_submission.csv' to 'submitions'

### 4. Train single models and get results
* Run 'python train.py -m [inception|resnet50|xception]' to get transform learning model
* Run 'python finetune.py -m [inception|resnet50|xception]' to fine tune model
* All model weight files are in the folder 'weights'
* Run 'python submit.py -m [inception|resnet50|xception]' to get submition
* All submitions are in the folder 'submitions'

### 5. Train blend model and get results
* Run 'python blend.py -m xception resnet50 inception' to get blend result with validation
* Run 'python blend.py -m xception resnet50 inception -a true' to get blend result without validation
