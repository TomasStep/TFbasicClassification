#TensorFlow example of basic Linear Model
import tensorflow as tf
import numpy as np

print(tf.__version__)

from tensorflow.contrib.learn.python.learn.datasets import base

#Data files
IRIS_TRAINING = "data_train.csv"
IRIS_TEST = "data_test.csv"

#Load datasets
training_set = base.load_csv_with_header(filename=IRIS_TRAINING,
                                         features_dtype=np.float32,
                                         target_dtype=np.int)
test_set = base.load_csv_with_header(filename=IRIS_TEST,
                                     features_dtype=np.float32,
                                     target_dtype=np.int)

print(training_set.data)

print(training_set.target)
