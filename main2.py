from __future__ import print_function
import random
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
import os

from neat import nn, population, statistics

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(sys.path[0],'VRN'))

from utils import checkpoints
from utils import metrics_logging
import imp

from collections import OrderedDict
import matplotlib

np.set_printoptions(threshold=np.nan)

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
basepath=str(os.path.dirname(os.path.realpath(__file__)))
configure_path = basepath+'/VRN/Discriminative/VRN.py'
config_module = imp.load_source('config', configure_path)
cfg = config_module.cfg

# Find weights 
fileweights_fname = str(configure_path)[:-3] + '.npz'

# Get Model
model = config_module.get_model()

# Compile functions
print('Compiling theano functions...')
obj_fun = make_testing_functions(model=model, category_id=0)

#CPPN
#make x, y, and z vectors of 100 elements with values between 0 and 31
#input=[x,y,z]

input = [[16, 16, 16],[16,16,19],[16,16,13],[16,19,16],[16,13,16],[13,16,16],[19,16,16]]

threshold = 1 
def eval_fitness(genomes):
    for g in genomes:
	net = nn.create_feed_forward_phenotype(g)
	for inputs in input:
	    output = net.serial_activate(inputs)        
            outputarray = np.asarray(output)
            outputarray = np.reshape(outputarray,(32,32,32))
            outputarray[outputarray<threshold]=-1
            outputarray[outputarray>=threshold]=3
            #print(outputarray[20,20,:])
            temp = np.zeros((1,1,32,32,32),dtype=np.uint8)
            temp[0,0,:,:,:] = outputarray
            temp=temp.astype('float32') 
	        pred=obj_fun(temp)
            pred=float(pred)
            #print(pred)
            g.fitness = pred

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'main_config')
pop = population.Population(config_path)
pop.run(eval_fitness, 200)

print('Number of evaluations: {0}'.format(pop.total_evaluations))

# Display the most fit genome.
winner = pop.statistics.best_genome()
print('\nBest genome:\n{!s}'.format(winner))
