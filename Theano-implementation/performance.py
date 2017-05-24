# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 02:19:45 2017

@author: mrins
"""
import numpy as np
import theano
import theano.tensor as T
import lasagne
from models import mlp_3
from resynthesis import *
import pickle

master_path = '/Users/mrins/Documents/Research/ML_in_SP_Project'

def load_prediction():
    path = ''
    pred_file = 'sample.npz' 
    with open(path+pred_file, 'rb') as f:
        try:
            dataset = pickle.load(f, encoding='latin1')
        except:
            dataset = pickle.load(f)
    clean_s = dataset['s_test']
    noisy_x = dataset['x_test']
    ibm = dataset['ibm']    
    ibm_pred = dataset['ibm_pred']
    return [clean_s, noisy_x, ibm, ibm_pred]

     

sampling_rate = 16000
frame_size = 1024
F = generate_dft_coeffs(frame_size)
s_test, x_test, ibm, ibm_pred = load_prediction()
s_test_pred = ibm_pred * x_test
s_test_pred = get_wave(s_test_pred, F)
s_test = get_wave(s_test, F)
print('Writing recovered signal to file')
wav.write(data=s_test, rate=sampling_rate, filename='s.wav')
wav.write(data=s_test_pred, rate=sampling_rate, filename='s_cap.wav')
s_test_pred_normalized = s_test_pred / s_test_pred.var()
wav.write(data=s_test_pred_normalized, rate=sampling_rate, filename='s_cap_norm.wav')

#    # SNR Calculation
print('Computing SNR')
print('\n\n', '-'*20, 'SNR', '-'*20)
s_test = s_test / s_test.max()
s_test_pred = s_test_pred / s_test_pred.var()
s_test_pred_normalized = s_test_pred_normalized / s_test_pred_normalized.max()

SNR = 10 * np.log10(s_test.var() / (s_test - s_test_pred).var())
print('SNR of recovered signal:', SNR)

