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
def make_testing_functions(model, category_id):
    # Input Array
    X = T.TensorType('float32', [False]*5)('X')

    # Get output
    obj = T.sum(lasagne.layers.get_output(model['l_out'], X, deterministic=True), axis=0)

    # Compile Functions
    obj_fun = theano.function([X], obj)

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

# Compile functions
print('Compiling theano functions...')
obj_fun = make_testing_functions(model=model, category_id=0)

# Prepare data
# X = generate() ## CPPN
X = np.random.randint(low=0, high=1, size=(1,1,32,32,32)).astype('float32')*3.0-1.0

pred = obj_fun(X)

print pred

