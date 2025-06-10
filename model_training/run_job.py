import sys, os
import random
import argparse
import pickle

#Import Functions to Build Pipeline
from make_data import make_data
from make_model import make_model
from train import train 
from evaluate import eval_model
from save import make_directories, save_results

#Import Numpy, Tensorflow
import numpy as np
import tensorflow as tf

#Set Seed
tf.random.set_seed(0)
np.random.seed(0)
random.seed(0)

from random import uniform
from time import sleep

#Select GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Command Line Arguements
parser = argparse.ArgumentParser(description='ECG Experiment')
parser.add_argument('--arg_file', type=str, default='', metavar='a',
                    help='Path to File with Grid Search Arguments')
parser.add_argument('--index', type=int, default=9999, metavar='i',
                    help='Index for Job Array')
parser.add_argument('--verbose', type=int, default=1, metavar='v',
                    help='Prints Outputs')
args = parser.parse_args()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Get Index (Either from argument or from SLURM JOB ARRAY)
if 'SLURM_ARRAY_TASK_ID' in os.environ:
    args.index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    print('SLURM_ARRAY_TASK_ID found..., using index %s' % args.index)
else:
    print('no SLURM_ARRAY_TASK_ID... using index %s' % args.index)
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Get Arguments
with open(args.arg_file, "rb") as arg_file:
    args.arg_file = pickle.load(arg_file)

input_list, label_dict, params = args.arg_file[args.index]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Run Experiment
## Generate Saving Directories
#Time Delay
sleep(uniform(0,10))
exp_dir, model_dir = make_directories(**params)
if args.verbose > 0:
    print('Experiment Directory: {}, Model Directory: {}'.format(exp_dir, model_dir))

## Make Dataset
builder, ds_train, ds_val, ds_test =  make_data(input_list, label_dict, **params)
if args.verbose > 0:
    print('Dataset Loaded!')

## Build Model
model = make_model(input_list, builder, label_dict, **params)
if args.verbose > 0:
    print('Model Built!')
    try:
        print(model.summary())
    except:
        print('ML model!')

## Train Model
model = train(model, model_dir, ds_train, ds_val, label_dict, **params)
if args.verbose > 0:
    print('Model Trained!')

## Evaluate Model
results_dict_val = eval_model(model, ds_val, model_dir, label_dict, 'val', **params)
results_dict_test = eval_model(model, ds_test, model_dir, label_dict, 'test', **params)
if args.verbose > 0:
    print('Results Generated!')
    print(results_dict_val)
    print(results_dict_test)

## Save
#Time Delay
sleep(uniform(0,10))
save_results(results_dict_val, results_dict_test, exp_dir, model_dir, input_list, label_dict, **params)
print('Done! {:-)>')
    
## Delete weights file at the end
## Temporary measure for neural networks only
# os.remove(os.path.join(model_dir, 'weights.h5'))

