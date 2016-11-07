from __future__ import print_function
import random

import numpy as np
import theano
import theano.tensor as T
import theano.sandbox.cuda.basic_ops as sbcuda
import lasagne

import sys

import os
from neat import nn, population, statistics

import imp
sys.path.append('/home/p2admin/Documents/Malcolm/voxel-cppn')
# from utils import checkpoints, metrics_logging
checkpoints = imp.load_source('utils', '/home/p2admin/Documents/Malcolm/voxel-cppn/VRN/utils/checkpoints.py')
metrics_logging = imp.load_source('utils', '/home/p2admin/Documents/Malcolm/voxel-cppn/VRN/utils/metrics_logging.py')
config_path = '/home/p2admin/Documents/Malcolm/voxel-cppn/VRN/Discriminative/VRN.py'
data_path = '/home/p2admin/Documents/Malcolm/voxel-cppn/VRN/datasets/modelnet40_rot_test.npz'

# Define the testing functions
def make_testing_functions(model, category_id):
	# Input Array
	X = T.TensorType('float32', [False]*5)('X')
	# Get output
	obj = T.sum(lasagne.layers.get_output(model['l_out'], X, deterministic=True), axis=0)[category_id]
	# Compile Functions
	obj_fun = theano.function([X], obj)
	return obj_fun

# Load config module
config_module = imp.load_source('config', config_path)
cfg = config_module.cfg

# Find weights 
fileweights_fname = str(config_path)[:-3] + '.npz'

# Get Model
model = config_module.get_model()

# Compile functions
print('Compiling theano functions...')
obj_fun = make_testing_functions(model=model, category_id=0)

# Prepare data
X = np.random.randint(low=0, high=1, size=(1,1,32,32,32)).astype('float32')*4.0-1.0

############
### CPPN ###
############

input = [[[0 for k in xrange(32)] for j in xrange(32)] for i in xrange(32)] 
for i in xrange(100):
    x=random.randrange(0,32,1) 
    y=random.randrange(0,32,1) 
    z=random.randrange(0,32,1) 
    input[x][y][z] = 1

def eval_fitness(genomes):
    for g in genomes:
        net = nn.create_feed_forward_phenotype(g)
        output = net.serial_activate(input)
        outputarray = np.asarray(output)
        outputarray = np.reshape(outputarray,(32,32,32))
        # outputarray = outputarray>.5 ### Make binary array
        temp = np.zeros((1,1,32,32,32),dtype=np.uint8)
        temp[0,0,:,:,:] = outputarray
        g.fitness = obj_fun(temp)

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'main_config')
pop = population.Population(config_path)
pop.run(eval_fitness, 4)

print('Number of evaluations: {0}'.format(pop.total_evaluations))

# Display the most fit genome.
winner = pop.statistics.best_genome()
print('\nBest genome:\n{!s}'.format(winner))
