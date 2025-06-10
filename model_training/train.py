import sys, os
import pickle

import numpy as np

#Tensorflow
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import tensorflow_probability as tfp

import xgboost as xgb

from sksurv.metrics import concordance_index_censored

# Custom Loss for Thresholded Regression Losses
def regression_loss(loss_fn, threshold):
    if loss_fn == 'mean_absolute_error':
        def custom_mae(y_true, y_pred):
            return K.mean(K.clip(K.abs(y_pred - y_true), 0., threshold), axis=-1)
        
        return custom_mae
    
    if loss_fn == 'logcosh':
        def custom_logcosh(y_true, y_pred):
            def _logcosh(x):
                return x + nn.softplus(-2. * x) - math_ops.cast(math_ops.log(2.), x.dtype)
            return K.mean(_logcosh(K.minimum(y_pred - y_true, threshold)), axis=-1)
        
        return custom_logcosh
    
    if loss_fn == 'mean_squared_error':
        def custom_mse(y_true, y_pred):
            return K.mean(K.clip(K.square(y_pred - y_true), 0., threshold), axis=-1)
        
        return custom_mse

# Custom Loss for Time to Event Analysis
def get_bin(time, boundaries):
    original_shape = time.shape
    time = tf.expand_dims(time,-1)
    boundaries_to_consider = boundaries[1:-1]
    time_cat = tf.cast(time > boundaries_to_consider, tf.float32)
    time_cat = K.sum(time_cat,-1)

    return time_cat

def tte_loss(loss_fn, boundaries):
    if loss_fn == 'failure':
        def failure_nll(y_true, y_pred):
            logeps = 1e-4
            #Initialize Categorical from Logit Preds
            y_pred = tf.clip_by_value(y_pred ,logeps,1.0-logeps)
            y_pred /= tf.reshape(K.sum(y_pred, 1), (-1, 1))
            dist = tfp.distributions.Categorical(probs = y_pred)

            #Split y_true
            U = y_true[:,0]
            Delta = y_true[:,1]

            logeps = 1e-4
            MAX_BIN_IDX = len(boundaries)-1

            #Bin U
            U_bin = get_bin(U, boundaries)
            U_max = U_bin == MAX_BIN_IDX

            # Calulate PMF of Time
            pmf_term = -1.0 * tf.clip_by_value(dist.log_prob(U_bin), -10, 10)

            #Maximizing bins after the censoring time. Might want to include current bins
            ## Want to include bin for when censoring time = 0
    #         survivor_term = -1.0 * dist.log_survival_function(U) 
            survivor_term = -1.0 * K.log(tf.clip_by_value(1. - dist.cdf(U_bin),logeps,1.0-logeps))
            print(survivor_term)

            # Use PMF if Observed or Last Bin else use survior time
            nll = tf.where((Delta==1.) | U_max, pmf_term, survivor_term)

            nll = K.mean(nll, 0)

            return nll

        return failure_nll
    
    if loss_fn == 'failure_only':
        def failure_only_nll(y_true, y_pred):
            logeps = 1e-4
            #Initialize Categorical from Logit Preds
            y_pred = tf.clip_by_value(y_pred ,logeps,1.0-logeps)
            y_pred /= tf.reshape(K.sum(y_pred, 1), (-1, 1))
            dist = tfp.distributions.Categorical(probs = y_pred)

            #Split y_true
            U = y_true[:,0]
            Delta = y_true[:,1]

            logeps = 1e-4
            MAX_BIN_IDX = len(boundaries)-1

            #Bin U
            U_bin = get_bin(U, boundaries)
            U_max = U_bin == MAX_BIN_IDX

            # Calulate PMF of Time
            pmf_term = -1.0 * tf.clip_by_value(dist.log_prob(U_bin), -10, 10)

            #Maximizing bins after the censoring time. Might want to include current bins
            ## Want to include bin for when censoring time = 0
    #         survivor_term = -1.0 * dist.log_survival_function(U) 
            survivor_term = -1.0 * K.log(tf.clip_by_value(1. - dist.cdf(U_bin),logeps,1.0-logeps))
            print(survivor_term)

            # Use PMF if Observed else 0
            nll = tf.where((Delta==1.), pmf_term, tf.zeros_like(pmf_term))

            nll = K.sum(nll, 0)/ tf.clip_by_value(K.sum(Delta), 1, 1e8)

            return nll

        return failure_only_nll
    
    elif loss_fn == 'censorship':
        def censorship_nll(y_true, y_pred):
            logeps = 1e-4
            #Initialize Categorical from Logit Preds
            y_pred = tf.clip_by_value(y_pred ,logeps,1.0-logeps)
            y_pred /= tf.reshape(K.sum(y_pred, 1), (-1, 1))
            dist = tfp.distributions.Categorical(probs = y_pred)

            #Split y_true
            U = y_true[:,0]
            Delta = y_true[:,1]

            logeps = 1e-4
            MAX_BIN_IDX = len(boundaries)-1

            #Bin U
            U_bin = get_bin(U, boundaries)
            U_max = U_bin == MAX_BIN_IDX

            # Calulate PMF of Time
            pmf_term = -1.0 * tf.clip_by_value(dist.log_prob(U_bin), -10, 10)

            #Maximizing bins after the censoring time. Might want to include current bins
            ## Want to include bin for when censoring time = 0
    #         survivor_term = -1.0 * dist.log_survival_function(U)
            Ubin_1m = tf.clip_by_value(tf.math.subtract(U_bin, 1), 0, MAX_BIN_IDX)
            survivor_term = -1.0 * K.log(tf.clip_by_value(1. - dist.cdf(Ubin_1m),logeps,1.0-logeps))
            print(survivor_term)

            # Use PMF if Not Observed else use survior time
            nll = tf.where((Delta==0.), pmf_term, survivor_term)

            nll = K.mean(nll, 0)

            return nll

        return censorship_nll
    
    
# Return Dictionary of Loss Functions
def make_loss_fn(label_dict):
    loss_fn = dict()
    for k, v in label_dict.items():
        if v['loss_fn'] in ['mean_absolute_error', 'mean_squared_error', 'logcosh']:
            if v['threshold'] is not None:
                loss_fn[k] = regression_loss(v['loss_fn'], v['threshold'])
            else:
                loss_fn[k] = v['loss_fn']
        elif v['loss_fn'] in ['failure', 'failure_only', 'censorship']:
            loss_fn[k] = tte_loss(v['loss_fn'], v['boundaries'])
        else:
            loss_fn[k] = v['loss_fn']
            
    return loss_fn

#Define Custom Condordance Metric
def concordance_metric(metric, boundaries):
    '''
    y_test: Includes times (censoring or failure) --> n x 2
    y_pred_time: Includes predicted failure time --> n x 1
    '''
    #Get Times
    times = boundaries[1:-1]
    center_times = [(x + y)/2 for x, y in zip(boundaries[0:-1],boundaries[1:])]
    
    if metric == 'failure':
        def py_failure_condordance(y_true, y_pred):
            y_true = y_true.numpy()
            y_pred = y_pred.numpy()
            
            #Get Prediction Time from Probabilities
            y_pred_bin = y_pred.argmax(1)
            y_pred_time = np.array([center_times[y] for y in y_pred_bin])
            
            # Decompose y_true
            U = y_true[:,0]
            delta = y_true[:,1].astype(bool)
             
            #Unweighted Concordance
            cindex_uw, _, _, _, _ = concordance_index_censored(delta, U, y_pred_time)
            
            return cindex_uw
        
        def failure_condordance(y_true, y_pred):
            
            return tf.py_function(func=py_failure_condordance, inp=[y_true, y_pred], Tout=tf.float64)
        
        return failure_condordance
    

# Return Dictionary of Metrics Functions
def make_metrics_fn(label_dict):
    metrics_fn = dict()
    for k, v in label_dict.items():
        if v['metrics'] in ['failure', 'censorship']:
            metrics_fn[k] = concordance_metric(v['metrics'], v['boundaries'])
        else:
            metrics_fn[k] = v['metrics']
            
    return metrics_fn

# Trains Neural Network Model
def train_nn(model, model_dir, train_data, val_data, label_dict, **params):
    #1. Save Dir
    output_weights_path = os.path.join(model_dir, 'weights.h5')
    output_weights_path
    
    #2. Compile Model
    ## Get Loss FN + weights
    loss_fn = make_loss_fn(label_dict)
    loss_weight = {k: v['loss_weight'] for k,v in label_dict.items()}
    
    ## Get Metrics
    metrics_fn = make_metrics_fn(label_dict)
    
    ## Compile
#     # Comment 3/14/2024, depending on the tf version, you'll need weighted_metrics instead of metrics to properly evaluate
    model.compile(
        loss=loss_fn,
        optimizer=optimizers.Adam(params['learning_rate']),
        metrics=metrics_fn,
        loss_weights = loss_weight
    )
    
#     model.compile(
#         loss=loss_fn,
#         optimizer=optimizers.Adam(params['learning_rate']),
#         weighted_metrics=metrics_fn,
#         loss_weights = loss_weight
#     )
    
    #3. Checkpoint + Callbacks
    checkpoint = ModelCheckpoint(
         output_weights_path,
         save_weights_only=False,
         save_best_only=True,
         verbose=1,
    )
    reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=int(params['epochs']/10), 
                                 verbose=1, mode='min', cooldown=1, min_lr=params['learning_rate']/100)
    earlyStop = EarlyStopping(monitor="val_loss", mode="min", patience=int(params['epochs']/5)) 
    callbacks = [checkpoint, reduceLR, earlyStop]
    
    #4. Fit Model
    model.fit(
        train_data,
        epochs=params['epochs'],
        validation_data=val_data,
        callbacks = callbacks
    )
    
    #5. Load best model
    model = tf.keras.models.load_model(output_weights_path, compile=False)
    
    return model

#Trains the ML (sklearn) Models
def train_ml(model, model_dir, train_data, **params):
    output_weights_path = os.path.join(model_dir, 'weights.pkl')
    output_weights_path
    
    #2. Fit Model
    model.fit(train_data[0], np.argmax(train_data[1],1).squeeze())
    pickle.dump(model, open(output_weights_path, 'wb'))
    
    return model

#Trains the XGB Models
def train_xgb(model_params, model_dir, train_data, val_data, **params):
    #1. Set Path
    output_weights_path = os.path.join(model_dir, 'weights.model')
    output_weights_path
    
    #2. Make DMatrix
    if 'reg' in model_params['objective']:
        train_data = xgb.DMatrix(train_data[0], label=train_data[1].squeeze())
        val_data = xgb.DMatrix(val_data[0], label=val_data[1].squeeze())
    else:
        train_data = xgb.DMatrix(train_data[0], label=np.argmax(train_data[1],1).squeeze())
        val_data = xgb.DMatrix(val_data[0], label=np.argmax(val_data[1],1).squeeze())

    #2. Fit Model
    model = xgb.train(
        model_params,
        train_data,
        num_boost_round=params['num_boost_round'],
        evals=[(val_data, "Val")],
        early_stopping_rounds=params['early_stopping_rounds'])
    
    #3. Save Model
    model.save_model(output_weights_path)
    
    return model

# Master Train Function
def train(model, model_dir, train_data, val_data, label_dict, **params):
    
    if params['model_type'] == 'nn': 
        model = train_nn(model, model_dir, train_data, val_data, label_dict, **params)
        
    elif params['model_type'] == 'lr':
        model = train_ml(model, model_dir, train_data, **params)
        
    elif params['model_type'] == 'xgb':
        model = train_xgb(model, model_dir, train_data, val_data, **params)
        
    return model