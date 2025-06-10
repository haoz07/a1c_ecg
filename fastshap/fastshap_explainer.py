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
import shap
import sys
from make_data import *
from make_model import *
import pickle
import pandas as pd
import os
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from fastshap import FastSHAP, EfficiencyNormalization
import matplotlib.pyplot as plt
from fastshap.utils import convert_to_linkTF, convert_to_link, ShapleySampler
import random
import time
from sklearn.metrics import (roc_curve, precision_recall_curve, auc, roc_auc_score,
                             average_precision_score, accuracy_score, 
                             precision_score, recall_score)
import matplotlib.pyplot as plt
from fastshap_surrogate import ECGSurrogate
from tqdm import tqdm

model_dir = '.'

label_dict = {'label': {'reformat': 'one_hot',  
                        'loss_fn': 'CategoricalCrossentropy',
                        'loss_weight': None,
                        'threshold': None,
                        'metrics': ['accuracy', 'AUC']}}

params = pickle.load(open(model_dir + '/params.pkl', 'rb'))

input_list = ['ecg', 'age', 'sex', 'race', 'smoking', 'phys_act', 'bmi']

builder, ds_train, ds_val, ds_test =  make_data(input_list, label_dict, **params)

# Get model Architecture
surrogate_model = make_model(input_list, builder, label_dict, **params)
# Load Model Weights
surrogate_model.load_weights(os.path.join(model_dir, 'weights.h5'))

superpixel_size  = 25
 
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
                builder=builder,
                val_data = ds_val,
                batch_size = params['batch_size'],
                max_epochs = 100,
                validation_batch_size = params['batch_size'],
                loss_fn='CategoricalCrossentropy',
                lr=1e-3,
                min_lr=1e-5,
                lr_factor=0.9,
                lookback=10,
                gpu_device=0,
                verbose=1, 
                model_dir = model_dir)


surrogate_trained = surrogate.model
surrogate_trained.trainable = False

params_debug = params.copy()
params_debug['batch_size'] = 1

builder_debug, ds_train_debug, ds_val_debug, ds_test_debug =  make_data(input_list, label_dict, **params_debug)

# get one sample
for x, y, z in ds_test_debug:
    print(len(x))
    break

for i in range(10):
    # surrogate model, original model
    print(surrogate_trained.predict(x), surrogate_model.predict(x)) 

inputs = {} # input dictionary
input_lists = [] # ecg input
tabular_concat = [] # tabular inputs
S_shape = 0
# establish input lists
for i in surrogate_model.input.keys():
    input_shape = builder._info().features[i].shape
    input_len = len(surrogate_trained.input[i].shape) 
    if i == 'ecg':
        model_input = Input(shape=(input_shape[1],input_shape[0], ), dtype='float32', name=i)
    else:
        model_input = Input(shape=input_shape, dtype='float32', name=i)
    inputs[i] = (model_input)
    if len(input_shape) == 0:
        model_input = tf.expand_dims(model_input, -1)

    if i == 'ecg':
        input_lists.append(model_input)
    else:
        S_shape += model_input.shape[1]
        tabular_concat.append(model_input)

tab = Concatenate()(tabular_concat)
ecg = surrogate_trained.get_layer('ecg_model')(input_lists[0])

y = Concatenate()([tab,ecg]) 

# linking the rest of the surrogate network
# note change this based on the network, from the 1st dense layer essentially
for l in surrogate_trained.layers[-5:-1]:
    y = l(y)

# value function log(y_1|Xs)
out = Lambda(lambda x: tf.math.log(tf.math.reduce_sum(x[:, 2:], axis=-1, keepdims=True)))(y)

surrogate_model_modified = Model(inputs, out)

surrogate_model_modified.trainable = False

surrogate_model_modified.compile()

surrogate_model_modified.summary()

# Get model Architecture
model_base = make_model(input_list, builder, label_dict, **params)
# Load Model Weights
model_base.load_weights(os.path.join(model_dir, 'weights.h5'))

inputs = {} # input dictionary
input_lists = [] # ecg input
tabular_concat = [] # tabular inputs
S_shape = 0
# establish input lists
for i in model_base.input.keys():
    input_shape = builder._info().features[i].shape
    input_len = len(model_base.input[i].shape) 
    if i == 'ecg':
        model_input = Input(shape=(input_shape[1],input_shape[0], ), dtype='float32', name=i)
    else:
        model_input = Input(shape=input_shape, dtype='float32', name=i)
    inputs[i] = (model_input)
    if len(input_shape) == 0:
        model_input = tf.expand_dims(model_input, -1)

    if i == 'ecg':
        input_lists.append(model_input)
    else:
        S_shape += model_input.shape[1]
        tabular_concat.append(model_input)

tab = Concatenate()(tabular_concat)
ecg = model_base.get_layer('ecg_model')(input_lists[0])

y = Concatenate()([tab,ecg]) 

# linking the rest of the surrogate network
# again, change based on the number of dense layers
for l in model_base.layers[-4:-1]:
    y = l(y)

# this should match number of players (P): 14 + 2500 / 25 = 114
phi = Dense(114, activation="linear")(y)

explainer = Model(inputs, phi)

explainer.trainable = True

explainer.compile()

train_data = ds_train,
val_data = ds_val
batch_size=32
validation_batch_size = 64
num_samples=1
max_epochs=200
model_dir = model_dir
lr=1e-5
verbose = 1
lookback = 5
min_lr=1e-6
lr_factor=0.5
eff_lambda=0
paired_sampling=True
explainer = explainer 
surrogate = surrogate_model_modified
normalization='additive'
link='identity'
baseline = 0
baseline_tabular = -1

# prepare null data
null_data = {}
for i in inputs:
    if i == 'ecg':
        null_data[i] = baseline * np.ones(tuple([1]+list(inputs[i].shape[1:])))
    else:
        null_data[i] = baseline_tabular * np.ones(tuple([1]+list(inputs[i].shape[1:])))


null = np.squeeze(surrogate_model_modified.predict(null_data))
linkfv =  np.vectorize(convert_to_link(link).f)
link = convert_to_linkTF(link)


# here number of players should be separate for tabular and ECG
superpixel_ecg = 25
P_tabular = 14
P_ecg = 2500 // superpixel_ecg # superpixel size 25
# P = 2514//6
D = 1


model_input = inputs


tab = Concatenate()(tabular_concat)
tab = tf.expand_dims(tab, -1)
tab = tf.tile(tab, [1, 1, 8])

tab_ecg = Concatenate(axis=1)([tab, input_lists[0]])
S = ShapleySampler(P_tabular + P_ecg, paired_sampling=paired_sampling, num_samples = num_samples)(tab_ecg)
S = Lambda(lambda x: tf.cast(x, tf.float32), name='S')(S)
if paired_sampling:
    num_samples = 2 * num_samples
else:
    num_samples = num_samples
phi = explainer(model_input)
phi = Reshape((D, P_tabular + P_ecg))(phi)
#Efficency Normalization
phi = Layer(name='phi')(phi)
grand = surrogate_model_modified(model_input)
phi, gap = EfficiencyNormalization(normalization, null, 
                                   link.f, P_tabular+P_ecg)([phi, grand])
phi = Reshape(((P_tabular + P_ecg)*D,))(phi)
# Repeat Phi for Multiple Subset Sampling
phi_repeat = tf.keras.layers.RepeatVector(num_samples)(phi)
phi_repeat = Reshape((num_samples, D, P_tabular + P_ecg),  name='phi_repeat')(phi_repeat)

# Calculate output 
phi_S = Lambda(lambda x: tf.concat([x[0], tf.expand_dims(x[1], 2)], 2))([phi_repeat, S])

out = TimeDistributed(
    Lambda(lambda x: 
           tf.squeeze(tf.matmul(x[:,:D,:], tf.expand_dims(x[:,-1,:], -1)), -1)),
    name = 'label'
)(phi_S)

model_input_repeat = Reshape(tuple([1]+list(tab_ecg.shape[1:])))(tab_ecg)


model_input_repeat = UpSampling2D(size=(num_samples, 1),
                                  name='model_input_repeat')(model_input_repeat)

# Resize Masks
S = TimeDistributed(Reshape((P_ecg + P_tabular, 1)))(S)
S_tabular = TimeDistributed(Lambda(lambda x: x[:,:14,:]))(S)
S_ecg = TimeDistributed(Lambda(lambda x: x[:,14:,:]))(S)
# size should be superpixel size: 25 here
S_ecg = TimeDistributed(UpSampling1D(size=(superpixel_ecg)), name='S_RM')(S_ecg)
S = Concatenate(axis=2)([S_tabular, S_ecg])


# new masking for different baselines
S_tabular_baseline = (1-S_tabular)*baseline_tabular
S_ecg_baseline = (1-S_ecg)*baseline
S_baseline = Concatenate(axis=2)([S_tabular_baseline, S_ecg_baseline])
xs = Lambda(lambda x: x[1]*x[0] + x[2])([model_input_repeat, S, S_baseline])

surrogate_input = Input(shape=(2514,8, ), dtype='float32', name='surrogate_input')
surr_input_lambda = Lambda(lambda x: x)(surrogate_input)
tabular_masked = Lambda(lambda x: x[:,:S_shape, 0])(surr_input_lambda)
ecg_masked = Lambda(lambda x: x[:,S_shape:,:])(surr_input_lambda)
ecg_masked = surrogate_model_modified.get_layer('ecg_model')(ecg_masked)

y = Concatenate()([tabular_masked,ecg_masked]) 
for l in surrogate_model_modified.layers[-5:]:
    y = l(y)
    
surrogate_td = Model(surrogate_input, y)

f_xs = TimeDistributed(surrogate_td, name='f_xs')(xs)
yAdj = TimeDistributed(
    Lambda(lambda x: K.stop_gradient(
        link.f(x) - link.f(tf.constant(null, dtype=tf.float32))
    )), name = 'yAdj'
)(f_xs)

SHAPloss = tf.reduce_mean(tf.keras.losses.MSE(yAdj, out))
EFFloss = eff_lambda*tf.reduce_mean(gap**2) 

explainer_final = Model(model_input, out)

explainer_final.add_loss(SHAPloss)
explainer_final.add_loss(EFFloss)

explainer_final.add_metric(SHAPloss, name='shap_loss', aggregation='mean')
explainer_final.add_metric(EFFloss, name='eff_loss', aggregation='mean')

# Model Checkpointing
explainer_weights_path = os.path.join(model_dir, 'explainer_weights.h5')
checkpoint = ModelCheckpoint(explainer_weights_path, monitor='val_shap_loss', verbose=verbose, 
                             save_best_only=True, mode='min', save_weights_only = True)

# Early Stopping 
earlyStop = EarlyStopping(monitor="val_shap_loss", mode="min", patience=lookback) 

# LR Schedule
reduceLR = ReduceLROnPlateau(monitor='val_shap_loss', factor=lr_factor, patience=2, 
                             verbose=1, mode='min', cooldown=1, min_lr=min_lr)

# Compile Model
CALLBACKS = [checkpoint, earlyStop, reduceLR]
OPTIMIZER = tf.keras.optimizers.Adam(lr)

explainer_final.compile(
    optimizer=OPTIMIZER
)


# Train Model
history = explainer_final.fit(x = ds_train,
                             epochs = max_epochs,
                             validation_data = ds_val,
                             validation_batch_size = validation_batch_size,
                             callbacks = CALLBACKS, 
                             verbose=verbose)


import pickle
pickle.dump(history, open(os.path.join(model_dir, 'history.pkl'), 'wb'))


# explainer_final.load_weights(explainer_weights_path)
explainer_final.load_weights(os.path.join(model_dir, 'explainer_weights.h5'))

base_input = {}
for i in explainer_final.layers[:7]:
    base_input[i.name] = i.input

base_model = Model(base_input, 
                   explainer_final.get_layer('phi').output)


phi = base_model(base_input)


phi = Permute((2,1))(phi)
phi = Reshape((P_tabular+P_ecg, D))(phi)
phi_tabular = phi[:,:14,:]
phi_ecg = phi[:,14:,:]
phi_ecg = UpSampling1D(size=(superpixel_ecg))(phi_ecg) # superpixel size = 25
phi_ecg = Rescaling(1./25, offset=0.0)(phi_ecg)
phi = Concatenate(axis=1)([phi_tabular, phi_ecg])


shap_explainer = Model(base_input, phi)
shap_explainer.trainable = False
shap_explainer.save(os.path.join(model_dir, 'explainer.h5'))


shap_explainer = tf.keras.models.load_model(os.path.join(model_dir, 'explainer.h5'))

normalization='additive'
link='identity'
baseline = 0
baseline_tabular = -1


# prepare null data
null_data = {}
for i in inputs:
    if i == 'ecg':
        null_data[i] = baseline * np.ones(tuple([1]+list(inputs[i].shape[1:])))
    else:
        null_data[i] = baseline_tabular * np.ones(tuple([1]+list(inputs[i].shape[1:])))


null = np.squeeze(surrogate_model_modified.predict(null_data))
linkfv =  np.vectorize(convert_to_link(link).f)
link = convert_to_linkTF(link)


shap_explainer = tf.keras.models.load_model(os.path.join(model_dir, 'explainer.h5'))


builder, ds_train, ds_val, ds_test =  make_data(input_list+['encounter_id'], label_dict, **params)


# all explanation
enc_keys = [x[0]['encounter_id'].numpy() for x in ds_test]
# efficeny normalization
explanations = shap_explainer.predict(ds_test, verbose=1)

prediction = linkfv(surrogate_model_modified.predict(ds_test, verbose=1)) - linkfv(null)
# calculate final 
diff = prediction - explanations.sum(1)
explanations = explanations + np.expand_dims(diff/explanations.shape[1], -1)