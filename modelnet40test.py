import argparse
import imp
import time
import logging
import math
import os

import numpy as np
from path import Path
import theano
import theano.tensor as T
import theano.sandbox.cuda.basic_ops as sbcuda
import lasagne

import sys

sys.path.append('C:/Users/p2admin/documents/max/projects/voxel-cppn')
import imp

checkpoints = imp.load_source('utils', 'C:/Users/p2admin/documents/max/projects/voxel-cppn/vrn/utils/checkpoints.py')
metrics_logging = imp.load_source('utils',
                                  'C:/Users/p2admin/documents/max/projects/voxel-cppn/vrn/utils/metrics_logging.py')
# from utils import checkpoints, metrics_logging
from collections import OrderedDict
import matplotlib


# Define the testing functions
def make_testing_functions():
    # Input Array
    X = T.TensorType('float32', [False] * 5)('X')

    # # Shared Variable for input array
    # X_shared = lasagne.utils.shared_empty(5, dtype='float32')

    # Output layer
    l_out = model['l_out']

    # Get output
    y_hat_deterministic = lasagne.layers.get_output(l_out, X, deterministic=True)

    prob = T.sum(y_hat_deterministic, axis=0)

    # Compile Functions
    obj_fun = theano.function([X], [prob])

    return obj_fun



# Load config module
config_path = 'c:/users/p2admin/documents/max/projects/voxel-cppn/vrn/Discriminative/VRN.py'
data_path = 'c:/users/p2admin/documents/max/projects/voxel-cppn/vrn/datasets/modelnet40_rot_test.npz'
config_module = imp.load_source('config', config_path)
cfg = config_module.cfg
# Find weights file
weights_fname = str(config_path)[:-3] + '.npz'
# Get Model
model = config_module.get_model()
# Load weights
metadata = checkpoints.load_weights(weights_fname, model['l_out']) # model weights changed after this...inside load_weights, another model file is read from weights_fname

# Compile functions
print('Compiling theano functions...')
# category_id=0
obj_fun = make_testing_functions()

# Prepare data
data=np.load(data_path)
feat=data['features']

# Get chunks
chunk_index = 0
test_chunk_size = 12
upper_range = min(len(feat),(chunk_index+1)*test_chunk_size)
X = np.asarray(feat[chunk_index * test_chunk_size:upper_range, :, :, :, :], dtype=np.float32)
X = 4.0 * X - 1.0
prob = obj_fun(X)

print prob

