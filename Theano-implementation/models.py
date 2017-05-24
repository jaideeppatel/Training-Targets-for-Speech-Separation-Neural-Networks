# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 14:32:13 2017

@author: mrins
"""

import lasagne


input_size = 513
output_size = 513
batchsize = 10

# 2 hidden layers, 10 units each
def mlp_1(input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, input_size), input_var=input_var)
    l_hid1 = lasagne.layers.DenseLayer(l_in, num_units=10, 
                                       nonlinearity= lasagne.nonlinearities.rectify, 
                                       W=lasagne.init.GlorotUniform())
    l_hid2 = lasagne.layers.DenseLayer(l_hid1, num_units=10,
                                       nonlinearity=lasagne.nonlinearities.rectify,
                                       W=lasagne.init.GlorotUniform())
    l_out = lasagne.layers.DenseLayer(l_hid2, num_units=output_size,
                                      nonlinearity=lasagne.nonlinearities.sigmoid)                                      
    return l_out
    

# 3 hidden layers, 1024 units each
def mlp_2(input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, input_size), input_var=input_var)
    l_hid1 = lasagne.layers.DenseLayer(l_in, num_units=1024, 
                                       nonlinearity= lasagne.nonlinearities.rectify, 
                                       W=lasagne.init.GlorotUniform())
    l_hid2 = lasagne.layers.DenseLayer(l_hid1, num_units=1024,
                                       nonlinearity=lasagne.nonlinearities.rectify,
                                       W=lasagne.init.GlorotUniform())
    l_hid3 = lasagne.layers.DenseLayer(l_hid2, num_units=1024,
                                       nonlinearity=lasagne.nonlinearities.rectify,
                                       W=lasagne.init.GlorotUniform())
    l_out = lasagne.layers.DenseLayer(l_hid3, num_units=output_size,
                                      nonlinearity=lasagne.nonlinearities.softmax)                                      
    return l_out
    
    
    
# Built on model 2 with dropout. This model is used in the paper
def mlp_3(input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, input_size), input_var=input_var)
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)
    l_hid1 = lasagne.layers.DenseLayer(l_in_drop, num_units=1024, 
                                       nonlinearity= lasagne.nonlinearities.rectify, 
                                       W=lasagne.init.GlorotUniform())
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)
    l_hid2 = lasagne.layers.DenseLayer(l_hid1_drop, num_units=1024,
                                       nonlinearity=lasagne.nonlinearities.rectify,
                                       W=lasagne.init.GlorotUniform())
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)
    l_hid3 = lasagne.layers.DenseLayer(l_hid2_drop, num_units=1024,
                                       nonlinearity=lasagne.nonlinearities.rectify,
                                       W=lasagne.init.GlorotUniform())
    l_hid3_drop = lasagne.layers.DropoutLayer(l_hid3, p=0.5)
    l_out = lasagne.layers.DenseLayer(l_hid3_drop, num_units=output_size,
                                      nonlinearity=lasagne.nonlinearities.softmax)                                      
    return l_out
    
    
# cnn : 1 convolution, 1 fc layer
def cnn_1(input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, input_size),
                                        input_var=input_var)
    l_in_reshape = lasagne.layers.ReshapeLayer(l_in, shape=(batchsize, 1, 28, 28))
    l_h1 = lasagne.layers.Conv2DLayer(
            l_in_reshape, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    l_m1 = lasagne.layers.MaxPool2DLayer(l_h1, pool_size=(2, 2))

    l_fc = lasagne.layers.DenseLayer(
            l_m1,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    l_out = lasagne.layers.DenseLayer(
            l_fc,
            num_units=output_size,
            nonlinearity=lasagne.nonlinearities.softmax)

    return l_out
    
    
    
# cnn : 2 convolution, 1 fc layer with dropout, extension of cnn_1
def cnn_2(input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, input_size),
                                        input_var=input_var)
    l_in_reshape = lasagne.layers.ReshapeLayer(l_in, shape=(batchsize, 1, 28, 28))
    
    l_h1 = lasagne.layers.Conv2DLayer(
            l_in_reshape, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    l_m1 = lasagne.layers.MaxPool2DLayer(l_h1, pool_size=(2, 2))
    
    l_h2 = lasagne.layers.Conv2DLayer(
            l_m1, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    l_m2 = lasagne.layers.MaxPool2DLayer(l_h2, pool_size=(2, 2))

    l_fc = lasagne.layers.DenseLayer(
            l_m2,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)
    l_fc_drop = lasagne.layers.DropoutLayer(l_fc, p=0.5)

    l_out = lasagne.layers.DenseLayer(
            l_fc_drop,
            num_units=output_size,
            nonlinearity=lasagne.nonlinearities.softmax)

    return l_out
    
    
# 1 lstm layer with fc
def rnn_1(input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, None, input_size),
                                        input_var=input_var)
                                        
    batch_sz, seq_len, _ = l_in.shape
    n_lstm_units = 30
    l_lstm_1 = lasagne.layers.LSTMLayer(l_in, num_units=n_lstm_units)
    l_lstm_1_reshape = lasagne.layers.ReshapeLayer(l_lstm_1, shape=(-1, n_lstm_units))
    
    l_out = lasagne.layers.DenseLayer(l_lstm_1_reshape, num_units=output_size)
    l_out_reshape = lasagne.layers.ReshapeLayer(l_out, shape=(batch_sz, seq_len, output_size))
    
    return l_out_reshape