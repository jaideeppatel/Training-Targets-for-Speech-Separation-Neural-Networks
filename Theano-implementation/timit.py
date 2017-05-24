#!/usr/bin/env python

"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.
This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html
More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T
import pickle
import matplotlib.pyplot as plt

import lasagne
from models import *

master_path = '/Users/mrins/Documents/Research/ML_in_SP_Project/'

def load_dataset(filename):
    
    path = 'Data/'
    with open(path+filename, 'rb') as f:
        try:
            dataset = pickle.load(f, encoding='latin1')
        except:
            dataset = pickle.load(f)
    return dataset
            
            
def create_train_test_dataset(filename = 'sample.pkl'):
    
    dataset = load_dataset(filename)
            
    x = dataset['X'].T
    s = dataset['S'].T
    n = dataset['N'].T
    ibm = dataset['M'].T
    
    x = np.sqrt(np.real(x)**2 + np.imag(x)**2)
    
    num_examples = x.shape[0]
    num_examples_x_train = int(0.8 * num_examples) # 10% for validation
    num_examples_x_valid = int(0.1 * num_examples)
    
    x_train = x[:num_examples_x_train,:]
    x_val = x[num_examples_x_train:num_examples_x_train+num_examples_x_valid,:]
    x_test = x[num_examples_x_train+num_examples_x_valid:,:]
    y_ibm_train = ibm[:num_examples_x_train,:]
    y_ibm_val = ibm[num_examples_x_train:num_examples_x_train+num_examples_x_valid,:]
    y_ibm_test = ibm[num_examples_x_train+num_examples_x_valid:,:] 
    s_test = s[num_examples_x_train+num_examples_x_valid:,:]
    
#    return x_train, y_ibm_train, x_val, y_ibm_val, x_test, y_ibm_test, s_test
    return x_train, y_ibm_train, x_train, y_ibm_train, x_train, y_ibm_train, s


def create_train_test_dataset_per_speech(i):
                
    x = dataset['X'][i].T
    s = dataset['S'][i].T
    n = dataset['N'][i].T
    ibm = dataset['IBM'][i].T
    
    x = np.sqrt(np.real(x)**2 + np.imag(x)**2)
    
    num_examples = x.shape[0]
    num_examples_x_train = int(0.8 * num_examples) # 20% for validation
    num_examples_x_valid = int(0.1 * num_examples)
    
    x_train = x[:num_examples_x_train,:]
    x_val = x[num_examples_x_train:num_examples_x_train+num_examples_x_valid,:]
    x_test = x[num_examples_x_train+num_examples_x_valid:,:]
    y_ibm_train = ibm[:num_examples_x_train,:]
    y_ibm_val = ibm[num_examples_x_train:num_examples_x_train+num_examples_x_valid,:]
    y_ibm_test = ibm[num_examples_x_train+num_examples_x_valid:,:]    
    
    return x_train, y_ibm_train, x_val, y_ibm_val, x_test, y_ibm_test



# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batchsize`.
# If the size of the data is not a multiple of `batchsize`, it will not
# return the last (remaining) mini-batch.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(num_epochs=200):
    # Load the dataset

    batchsize = 10
    print("Loading data...")

    X_train, y_train, X_val, y_val, X_test, y_test, s_test = create_train_test_dataset()

    # Prepare Theano variables for inputs and targets
    input_var = T.matrix('inputs')
    target_var = T.matrix('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    network = mlp_3(input_var)

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.squared_error(prediction, target_var).mean()
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.momentum(loss, params, learning_rate=0.1, momentum=0.9)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction, target_var).mean()

    train_fn = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast=True)
    val_fn = theano.function([input_var, target_var], test_loss, allow_input_downcast=True)
    test_fn = theano.function([input_var, target_var], test_loss, allow_input_downcast=True)
    predict_op = theano.function([input_var], test_prediction, allow_input_downcast=True)
    
    print("Starting training...")
    train_loss = []
    for epoch in range(num_epochs):
            
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batchsize, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batchsize, shuffle=False):
            inputs, targets = batch
            err = val_fn(inputs, targets)
            val_err += err
            val_batches += 1

        # Then we print the results for this epoch:
        train_loss.append(train_err)
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

    # After training, we compute and print the test error:
    print(train_loss)
    plt.plot(range(len(train_loss)),train_loss)
    plt.show()
    print('Starting testing...')
    test_err = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, batchsize, shuffle=False):
        inputs, targets = batch
        err = test_fn(inputs, targets)
        test_err += err
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))


    # Optionally, you could now dump the network weights to a file like this:
    np.savez(master_path+filename.split('.')[0]+'.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)

    
    
    ibm_pred = predict_op(X_test)
    y_test = y_test
    dict_ibm = {'x_test': X_test.T, 's_test': s_test.T, 'ibm':y_test.T, 'ibm_pred':ibm_pred.T}
    with open(master_path+filename.split('.')[0]+'_ibm_predictions.pkl', 'wb') as handle:
        pickle.dump(dict_ibm, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(y_test)
    print(ibm_pred)
#    


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        main(**kwargs)
