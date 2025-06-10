import sys, os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

# Makes directories for saving Results + Models
def make_directories(**params):
    #Set Experiment Dir
    base_dir = '.'
    exp_dir = os.path.join(base_dir, params['task'], 'experiments', params['exp_name'])
    
    #Set Model Dir
    run_date = datetime.now().strftime("%Y%m%d")
    run_time = datetime.now().strftime("%H_%M_%S_%f")   
    model_dir = os.path.join(exp_dir, run_date, run_time)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
        
    return exp_dir, model_dir
        
# Saves Results (Pickles raw Data + Appends Results to csv for Experiment)
def save_results(results_dict_val, results_dict_test, exp_dir, model_dir, input_list, label_dict, **params):
    # 1. Pickle
    ##Save Full Results Dictionary
    pickle.dump(results_dict_val, open(os.path.join(model_dir, 'results_val.pkl'), 'wb'))
    pickle.dump(results_dict_test, open(os.path.join(model_dir, 'results_test.pkl'), 'wb'))

    ##Save Full Params Dictionary
    pickle.dump(params, open(os.path.join(model_dir, 'params.pkl'), 'wb'))

    # 2. Save to CSV
    results_path = os.path.join(exp_dir, 'results.csv')

    ## Get Results
    results_header = ['roc_auc', 'prc_auc', 'acc', 'r2', 'mse', 'mae']
    results_header += ['m_auroc', 'U_auroc','L_auroc', 'm_auprc', 'U_auprc', 'L_auprc']
    for P in np.linspace(.5, .9, 5):
        results_header.append('m_sens_' + str(P))
        results_header.append('U_sens_' + str(P))
        results_header.append('L_sens_' + str(P))
        results_header.append('m_spec_' + str(P))
        results_header.append('U_spec_' + str(P))
        results_header.append('L_spec_' + str(P))
    results_header += ['cindex_uw_time', 'cindex_uw_cdf', 'cindex_ipcw_time', 'cindex_ipcw_cdf', 
                       'auc', 'mean_auc', 'brier_scores', 'ibs']
    
    results_val = results_dict_val[params['label_save']]
    for i in set(results_val.keys()).intersection(set(['roc', 'prc', 'density_plot'])):
            results_val.pop(i)
    results_val  = {k:(results_val[k] if k in results_val.keys() else None) for k in results_header}
    
    results_test = results_dict_test[params['label_save']]
    for i in set(results_test.keys()).intersection(set(['roc', 'prc', 'density_plot'])):
            results_test.pop(i)
    results_test  = {k:(results_test[k] if k in results_test.keys() else None) for k in results_header}


    ## Get Params
    params_header = ['exp_name', 'label_save', 
                     'task', 'version', 'model_type', 
                     'sampling_method', 'weight_column',
                     'pretrain', 'pretrain_path', 'tune_layers', 
                     'conv_subsample_lengths', 'conv_filter_length', 'conv_num_filters_start', 
                     'conv_init', 'conv_activation', 'conv_dropout', 
                     'conv_num_skip', 'conv_increase_channels_at', 
                     'learning_rate', 'batch_size', 'epochs', 
                     'is_regular_conv', 'is_by_time', 'is_by_lead', 
                     'ecg_out_size', 'nn_layer_sizes', 'is_multiply_layer', 
                     'l1_ratio', 'c', 'max_iter', 'max_depth', 'n_estimators', 
                     'eta', 'min_child_weight', 'objective',
                     'eval_metric', 'subsample', 'colsample_bytree', 'num_boost_round', 'early_stopping_rounds', 
                     'single_lead'
                    ]
    params  = {k:(params[k] if k in params.keys() else None) for k in params_header}
    
    ## Get Inputs
    inputs = {'inputs': input_list}
    
    #Get Label Info
    labels = label_dict.keys()
    losses = [label_dict[k]['loss_fn'] for k in labels]
    loss_weights = [label_dict[k]['loss_weight'] for k in labels]
    reformats = [label_dict[k]['reformat'] for k in labels]
    labels = {'labels': labels, 'loss_fns': losses,  'loss_weights':loss_weights, 'reformats':reformats}

    ## Make DF
    results_val = {('val_'+k if k in results_header else k):v  for k,v in results_val.items()}
    results_test = {('test_'+k if k in results_header else k):v  for k,v in results_test.items()}
    results = {**results_val, **results_test, **params, **inputs, **labels}
    results = {k:(str(v) if type(v) in [list, np.ndarray] else v)  for k,v in results.items()}
    results['model_dir'] = model_dir
    
    results_header = ['val_' + x for x in results_header] + ['test_' + x for x in results_header]
    header = ['model_dir', 'inputs', 'labels', 'reformats', 'loss_fns', 'loss_weights'] + params_header + results_header
    print(results)
    results_df = pd.DataFrame(results, index=[0])
    results_df = results_df[header]

    ## Save
    if os.path.exists(results_path):
        results_df.to_csv(results_path, mode='a',  header=False)
    else:
        results_df.to_csv(results_path, mode='w',  header=True)