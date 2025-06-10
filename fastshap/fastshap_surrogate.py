# %%
import numpy as np
import sys

import tensorflow as tf
from tensorflow.keras.layers import (Input, Layer, Dense, Lambda, 
                                     Dropout, Multiply, BatchNormalization, 
                                     Reshape, Concatenate, Conv2D, Conv1D, Permute, UpSampling1D, UpSampling2D)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers.experimental.preprocessing import Resizing

# %%
import shap

# %%
import sys
from make_data import *
from make_model import *

# %%
import pickle
import pandas as pd
import os

# %%
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

# %%
from fastshap import FastSHAP, EfficiencyNormalization

# %%
import matplotlib.pyplot as plt

# %%
from fastshap.utils import convert_to_linkTF, convert_to_link, ShapleySampler

# %%
import random
import time

# %%
from tensorflow.keras.utils import plot_model

# %% [markdown]
# # Load up model architecture

# %%
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
INPUT_SHAPE = (2500, 8)

num_classes = 2

# %%
model_dir = '.'

# %%
label_dict = {'label': {'reformat': 'one_hot',  
                        'loss_fn': 'CategoricalCrossentropy',
                        'loss_weight': None,
                        'threshold': None,
                        'metrics': ['accuracy', 'AUC']}}

# %%
params = pickle.load(open(model_dir + '/params.pkl', 'rb'))

# %%
input_list = ['ecg', 'age', 'sex', 'race', 'smoking', 'phys_act', 'bmi']

# %%
builder, ds_train, ds_val, ds_test =  make_data(input_list, label_dict, **params)

# %%
# Get model Architecture
surrogate_model = make_model(input_list, builder, label_dict, **params)
# Load Model Weights
surrogate_model.load_weights(os.path.join(model_dir, 'weights.h5'))

# %%
class ECGSurrogate:
    '''
    Wrapper around surrogate model
    Args:
        surrogate_model
        baseline
        width
        superpixel_size,
        superpixel_tabular_size
        model_dir
        
    '''
    
    def __init__(self, 
                 surrogate_model,
                 baseline,
                 baseline_tabular,
                 width, 
                 superpixel_size, 
                 superpixel_tabular_size=1):
        
        # Models
        self.surrogate_model = surrogate_model
        
        # Set up superpixel upsampling.
        self.width = width
        self.superpixel_size = superpixel_size
        self.superpixel_tabular_size = superpixel_tabular_size
        if superpixel_size == 1:
            self.upsample = Lambda(lambda x: x)
        else:
            self.upsample = UpSampling1D(
                size=(superpixel_size))
            

        self.upsample_tabular = Lambda(lambda x: x)

        # Set up number of players.
        self.small_width = (width-superpixel_tabular_size) // superpixel_size

        self.P = self.small_width + superpixel_tabular_size
        # Set baseline masking value
        self.baseline = baseline
        self.baseline_tabular = baseline_tabular
        
        self.model = None
    def model():
        return self.model
        
    def train(self,
              original_model,
              train_data,
              val_data,
              batch_size,
              max_epochs,
              validation_batch_size,
              loss_fn='categorical_crossentropy',
              lr=1e-3,
              min_lr=1e-5,
              lr_factor=0.9,
              lookback=10,
              gpu_device=0,
              verbose=False, 
              model_dir=None):
        
        # model save dir
        if model_dir is None:
            self.save = datetime.now().strftime("%Y%m%d_%H_%M_%S")
            self.model_dir = os.path.join(os.getcwd(), self.save)
        else:
            self.model_dir = model_dir
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
        
        # Device
        if gpu_device is False:
            device = "cpu:0"
        else:
            device = "gpu:" + str(gpu_device)
        
        # Data
        #Check if Provided TF Dataset, if So X should be paired with model predictions
        if isinstance(train_data, tf.data.Dataset): 
            if original_model is not None:
                @tf.function
                def make_prediction_data(x, y):
                    with tf.device(device):
                        y_model = original_model(x)
                    return (x, y_model)

                with tf.device(device):
                    train_data = train_data.map(make_prediction_data)
                    val_data = val_data.map(make_prediction_data)
        else:
            if original_model is not None:
                fx_train = original_model.predict(train_data)
                fx_val = original_model.predict(val_data)
                train_data = tf.data.Dataset.from_tensor_slices((train_data, fx_train)).batch(batch_size)
                val_data = tf.data.Dataset.from_tensor_slices((val_data, fx_val)).batch(validation_batch_size)
            else:
                train_data = tf.data.Dataset.from_tensor_slices(train_data).batch(batch_size)
                val_data = tf.data.Dataset.from_tensor_slices(val_data).batch(validation_batch_size)
        
        
        #################################################################
        
        # Mask Inputs
        self.surrogate_model.trainable = True

#         inputs_dict = {}
        inputs = {}
        input_lists = []
        tabular_concat = []
        S_shape = 0
        # establish input lists
        for i in self.surrogate_model.input.keys():
            input_shape = builder._info().features[i].shape
            input_len = len(self.surrogate_model.input[i].shape) 
            if i == 'ecg':
                model_input = Input(shape=(input_shape[1],input_shape[0], ), dtype='float32', name=i)
            else:
                model_input = Input(shape=input_shape, dtype='float32', name=i)
            inputs[i] = (model_input)
            print(input_shape)
            if len(input_shape) == 0:
                model_input = tf.expand_dims(model_input, -1)
            print(model_input)
            
            if i == 'ecg':
                input_lists.append(model_input)
            else:
                S_shape += model_input.shape[1]
                tabular_concat.append(model_input)
            
            
        # concat
        print(input_lists)
        # concat tabular features
        tab = Concatenate()(tabular_concat)
        tab = tf.expand_dims(tab, -1)
        tab = tf.tile(tab, [1, 1, 8])
        print(tab.shape)
        tab_ecg = Concatenate(axis=1)([tab, input_lists[0]])
        print(tab_ecg.shape)
        S = ShapleySampler(self.P, paired_sampling=False, num_samples=1)(tab_ecg)
        S = Lambda(lambda x: tf.cast(x, tf.float32))(S)
        print(S.shape)
        S = Reshape((self.P,1))(S)
        
        S_tabular = S[:,:S_shape,:]
        S_ecg = S[:,S_shape:,:]
        S_ecg = self.upsample(S_ecg)
        
        S_tabular_baseline = (1-S_tabular)*self.baseline_tabular
        S_ecg_baseline = (1-S_ecg)*self.baseline
        
        S = Concatenate(axis=1)([S_tabular, S_ecg])
        S_baseline = Concatenate(axis=1)([S_tabular_baseline, S_ecg_baseline])
 
        x_S = Lambda(lambda x: x[1]*x[0] + x[2])([tab_ecg, S, S_baseline])
        
        tabular = x_S[:, :S_shape, 0]
        ecg = x_S[:,S_shape:,:]
        
        ecg = self.surrogate_model.get_layer('ecg_model')(ecg)
        
        y = Concatenate()([tabular,ecg]) 
        print(y.shape)
        # dense to out
        for i in range(13, 17): 
            print('connecting dense layers')
            y = self.surrogate_model.layers[i](y)
            print(y.shape)
        self.surrogate_model.get_layer('label')._name = 'label_hidden'
        out = Lambda(lambda x: x, name='label')(y)
        self.model = Model(inputs, out)
        print(self.model.summary())
        
        # Metrics
        METRICS = [ 
          tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='accuracy'),
        ]
        
        # Model Checkpointing
        weights_path = os.path.join(self.model_dir, 'surrogate_weights.h5')
        checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=verbose, 
                                     save_best_only=True, mode='min', save_weights_only = True)
        
        # LR Schedule
        reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=lr_factor, patience=2, 
                                     verbose=1, mode='min', cooldown=1, min_lr=min_lr)
        
        # Early Stopping 
        earlyStop = EarlyStopping(monitor="val_loss", mode="min", patience=lookback) 
        
        # Compile Model
        CALLBACKS = [checkpoint, earlyStop, reduceLR]
        OPTIMIZER = tf.keras.optimizers.Adam(lr)

        self.model.compile(
            loss=loss_fn,
            optimizer=OPTIMIZER,
            metrics=METRICS,
        )
        if not os.path.isfile(os.path.join(model_dir, 'surrogate_weights.h5')):

            # Train Model 
            self.model.fit(x = train_data,
                           epochs = max_epochs,
                           validation_data = val_data,
                           validation_batch_size = validation_batch_size,
                           callbacks = CALLBACKS,
                           verbose=verbose)
        
        
        # Get Checkpointed Model
        self.model.load_weights(weights_path)
        self.model.trainable = False   

# %%
superpixel_size = 25
 
surrogate = ECGSurrogate(surrogate_model = surrogate_model,
                      baseline = 0,
                        baseline_tabular = -1,
                      width = 2514, 
                      superpixel_size = superpixel_size,
                      superpixel_tabular_size = 14
                        )

t = time.time()
surrogate.train(original_model = None,
                train_data = ds_train,
                val_data = ds_val,
                batch_size = params['batch_size'],
                max_epochs = 100,
                validation_batch_size = params['batch_size'],
                loss_fn='CategoricalCrossentropy',
                lr=1e-3,
                min_lr=1e-5,
                lr_factor=0.9,
                lookback=20,
                gpu_device=0,
                verbose=1, 
                model_dir = model_dir)
training_time = time.time() - t

with open(os.path.join(model_dir, 'training_time.pkl'), 'wb') as f:
    pickle.dump(training_time, f)


