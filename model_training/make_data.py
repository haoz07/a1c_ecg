import sys, importlib, inspect, os, glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_datasets as tfds


# Function to Load DataLoader Class
def fetch_dataloader(task, version):
    '''
    Input: Task Names and Version
    Returns: The initialized the Dataset Class
    '''
    
    base_dir = '.'
    if task in [x[len(base_dir)+1:] for x in glob.glob(base_dir+'/*')]:
        dataloader_dir = os.path.join(base_dir, task)
        if version in [x[len(dataloader_dir)+1:-3] for x in glob.glob(dataloader_dir+'/*')]:
            sys.path.insert(0, dataloader_dir)
            
            dataloader_class = inspect.getmembers(importlib.import_module(version), inspect.isclass)[0][0]
            cmd = 'from {} import {} as builder'.format(version, dataloader_class) 
            exec(cmd, globals())
            
            b = builder()
            datasets = b.as_dataset(as_supervised=False)
            
            return datasets, b

# Function to Batch and Apply Reformatting to tfds Dataset
def batch_data(dataset, fn, batch_size):
    dataset = dataset.map(fn)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

# Function to Reformat and Select Inputs and Labels From Dataset
def format_dataset(datasets, builder, label_dict, input_list, batch_size, **params):
    ds_train = datasets['train']
    ds_val= datasets['validation']
    ds_test = datasets['test']
    
    # Function to Extract Inputs, Outputs, and Sample Weights
    output_weights = False
    def reformat(input_dict):
        output_dict = {}
        labels = {}
        for i in builder._info().features: 
            if i in label_dict.keys():
                reformat = label_dict[i]['reformat']
                #For Categorical Labels
                if reformat == 'one_hot':
                    CLASSES = builder._info().features[i].num_classes
                    labels[i] = tf.one_hot(input_dict[i], depth = CLASSES)   
                # To Discretize Continuout Labels
                # TO-DO: check if type is a number and not just int
                # from numbers import Number
                # isinstance(n, Number)
                elif type(reformat) == int:
                    CLASSES = 2
                    labels[i] = tf.one_hot(int(input_dict[i] > reformat), depth = CLASSES)
                # extreme delta
                elif reformat == 'extreme':
                    if 'risq' in i:
                        d_s = 'delta_qtcrisq'
                    elif 'frederica' in i:
                        d_s = 'delta_qtcfrederica'
                    else:
                        d_s = 'delta_qtc'
                        
                    CLASSES = 2
                    delta_qtc = input_dict[d_s] > 60
                    labels[i] = tf.one_hot(int((input_dict[i] > 1) or delta_qtc), depth = CLASSES)
                    
                else:
                    labels[i] = input_dict[i]
            elif i in input_list:
                if i == 'ecg':
                    output_dict[i] = tf.dtypes.cast(K.permute_dimensions(input_dict['ecg'], pattern=(1, 0)), tf.float32)
                    if 'single_lead' in params.keys():
                        if params['single_lead']:
                            output_dict[i] = tf.expand_dims(output_dict[i][:,0], -1)
                else:
                    output_dict[i] = input_dict[i] 
        
        # Add Sample Weights to Tuple if Appropreate
        if 'weight_column' in params.keys():
            print(output_weights)
            if (params['weight_column'] is not None) & output_weights:
                sample_weights = input_dict[params['weight_column']]
                return (output_dict, labels, sample_weights)

        return (output_dict, labels)
    
    # Function to Oversample Data
    def n_UpSample(num_to_repeat):
        num_to_repeat_integral = tf.cast(int(num_to_repeat), tf.float64)
        residue = tf.cast(num_to_repeat - num_to_repeat_integral, tf.float64)
        num_to_repeat = num_to_repeat_integral + tf.cast(tf.random.uniform(shape=(), dtype=tf.float64) <= residue, tf.float64)
        
        return tf.cast(num_to_repeat, tf.int64)
    
    # Handle Resampling/Re-Weighting
    if ('weight_column' in params.keys()) & ('sampling_method' in params.keys()):
        if (params['weight_column'] is not None):
            # Format Training Set
            if params['sampling_method'] == 'oversampling':
                ds_train = ds_train.flat_map(lambda x: tf.data.Dataset.from_tensor_slices({k:[v] for k,v in x.items()})
                                             .repeat(n_UpSample(x[params['weight_column']])))
                ds_train = ds_train.shuffle(20000)
                output_weights = False
                ds_train = batch_data(ds_train, reformat, batch_size)
            elif params['sampling_method'] == 'weighting':
                output_weights = True
                ds_train = batch_data(ds_train, reformat, batch_size)    
            elif params['sampling_method'] == 'stratifying':
                output_weights = False
                ds_train = batch_data(ds_train, reformat, batch_size)

            # Using sample_weighting for Validation and Test Set
            output_weights = True
            ds_val = batch_data(ds_val, reformat, batch_size)
            ds_test = batch_data(ds_test, reformat, batch_size)
    
    else:
        output_weights = False
        ds_train = batch_data(ds_train, reformat, batch_size)    
        ds_val = batch_data(ds_val, reformat, batch_size)
        ds_test = batch_data(ds_test, reformat, batch_size)
  
    return ds_train, ds_val, ds_test

#Function to Read Data into a numpy. Matrix
def make_table(dataset, input_list, label_dict, weight_column=None):
    llx = []
    lly = []
    if weight_column is not None:
        llz = []
    for x in dataset: 
        #X
        lx = [v.numpy() for k,v in x[0].items() if k in input_list]
        lx = [np.expand_dims(x,-1) if len(x.shape)==1 else x for x in lx]
        #Y
        ly = [v.numpy() for k,v in x[1].items() if k in label_dict.keys()]
        ly = [np.expand_dims(x,-1) if len(x.shape)==1 else x for x in ly]
        
        if weight_column is not None:
            #Z
            lz = [v.numpy() for v in x[2]]
            lz = [np.expand_dims(x,-1) if len(x.shape)==1 else x for x in lz]
            llz.append(np.stack(lz))
        
        #Append
        llx.append(np.concatenate(lx, 1))
        lly.append(np.concatenate(ly, 1))
    
    x = np.concatenate(llx, 0)
    y = np.concatenate(lly, 0)
    if weight_column is not None:
        z = np.concatenate(llz, 0)
        
        return x,y,z
    
    return x, y

#Master Function to Make Data
def make_data(input_list, label_dict, **params):
    data, builder = fetch_dataloader(params['task'], params['version'])
    ds_train, ds_val, ds_test = format_dataset(data, builder, label_dict, input_list, **params)
    
    if params['model_type'] != 'nn':
        ds_train = make_table(ds_train, input_list, label_dict, params['weight_column']) 
        ds_val = make_table(ds_val, input_list, label_dict, params['weight_column']) 
        ds_test = make_table(ds_test, input_list, label_dict, params['weight_column']) 
    
    return builder, ds_train, ds_val, ds_test