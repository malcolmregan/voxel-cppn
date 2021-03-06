from __future__ import print_function
from objplot import plotsave
import numpy as np
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
import imp

np.set_printoptions(threshold=np.nan)

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
basepath=str(os.path.dirname(os.path.realpath(__file__)))
configure_path = basepath+'/VRN/Discriminative/VRN.py'
data_path = basepath+'/VRN/datasets/CPPNGenerated'
config_module = imp.load_source('config', configure_path)
cfg = config_module.cfg
# Find weights file
weights_fname = str(configure_path)[:-3] + '.npz'
# Get Model
model = config_module.get_model()
# Load weights
metadata = checkpoints.load_weights(weights_fname, model['l_out']) # model weights changed after this...inside load_weights, another model file is read from weights_fname


# Compile functions
print('Compiling theano functions...')
obj_fun = make_testing_functions()

#CPPN
inp=[0]*(32*32*32)
n=0
for i in range(0, 32):
    for j in range(0, 32):
        for k in range(0, 32):
            inp[n]=(i,j,k)
            n=n+1

def get_fitness(g, inp, CLASS):
    net = nn.create_feed_forward_phenotype(g)
    outputarray = [0]*32*32*32
    for inputs in inp:
        output = net.serial_activate(inputs)
        outputarray[inputs[0]+inputs[1]*32+inputs[2]*32*32] = output[0]
    outputarray = np.reshape(outputarray,(32,32,32))
    threshold=.5
    outputarray[outputarray<threshold]=-1
    outputarray[outputarray>=threshold]=3
    temp = np.zeros((1,1,32,32,32),dtype=np.float32)
    temp[0,0,:,:,:] = outputarray
    pred = obj_fun(temp)
    pred = pred[0]
    pred[:CLASS]=np.absolute(pred[:CLASS])          #make all but CLASS positive (yes? no?)
    pred[(CLASS+1):]=np.absolute(pred[(CLASS+1):])  #???
    fitness = (pred[CLASS]-min(pred))/(np.sum(pred-min(pred)))
    if ((fitness>.2) and (os.path.isfile(os.path.join(data_path, "class_{0}_fitness_{1:.4f}".format(CLASS, fitness))))==False):
        filename = "class{0}_fitness{1:.4f}".format(CLASS, fitness)
        #Save data
        np.savez(os.path.join(data_path,filename+'.npz'),**{'features': temp, 'targets': [CLASS]})
        #Save object image
        plotsave(outputarray,os.path.join(data_path,filename+'.png'))
        #Save Prediction array as text file
        file = open(os.path.join(data_path,filename+'.txt'),'w')
        np.savetxt(file, pred)
        file.close
    return fitness


def eval_fitness(genomes):
    for g in genomes:
       g.fitness = get_fitness(g, inp, 0)
      
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'main_config')
pop = population.Population(config_path)
pop.run(eval_fitness, 10000)