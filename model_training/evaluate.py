import sys, os
import pickle

# Ploting + Computation
import numpy as np
from scipy import interp
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, precision_recall_curve, auc, roc_auc_score,
                             average_precision_score, accuracy_score, 
                             r2_score, mean_squared_error, mean_absolute_error)
from sksurv.metrics import (concordance_index_censored,concordance_index_ipcw,
                            cumulative_dynamic_auc,brier_score,integrated_brier_score)

# Tensorflow
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

import xgboost as xgb

from sklearn.utils import resample

# ROC for Binary Classification
def plot_auc(y_true, y_score, model_dir, label_name, sample_weight = None, split='test'):
    y_true = y_true[:,1]
    y_score = y_score[:,1]
    fpr, tpr, _ = roc_curve(y_true, y_score, sample_weight=sample_weight)
    roc_auc = auc(fpr, tpr)   
    colors = ['darkorange', 'deeppink']
    #Plot ROC
    lw =2
    figure = plt.figure(1, figsize=(10, 10))
    plt.plot(fpr, tpr, color=colors[0], lw=lw,
             label='{0} (area = {1:0.2f})'
             ''.format(1, roc_auc))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="best", fontsize=16)
    plt.savefig(os.path.join(model_dir, label_name + '_' + split + '_binary_roc.png'))
    plt.close(figure)
    
    return roc_auc, figure

# ROC for Multiclass Classification
def plot_auc_multi(y_true, y_score, model_dir, label_name, sample_weight = None, split='test'):
    num_classes = y_true.shape[1]
    #%% Compute PRC, and PRC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        #ROC
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i], sample_weight=sample_weight)
        roc_auc[i] = auc(fpr[i], tpr[i])
    #%% Compute micro-average ROC, PRC curve and ROC, PRC area
    # ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = roc_auc_score(y_true, y_score, average='micro', sample_weight=sample_weight)
   
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    
    # Then interpolate all ROC, PRC curves at this points
    ## ROC
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        
    mean_tpr /= num_classes

    ## ROC
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = roc_auc_score(y_true, y_score, average='macro', sample_weight=sample_weight)
    
    #%% Plot Curves
    lw = 2 
    # Plot all ROC curves
    figure = plt.figure(1, figsize=(10, 10))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    #cycle(['aqua', 'darkorange', 'cornflowerblue'])
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] + ['aqua', 'darkorange', 'cornflowerblue']
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='{0} (area = {1:0.2f}) (n={2})'
                 ''.format(i, roc_auc[i], int(y_true[:, i].sum())))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC')
    plt.legend(loc="best", fontsize=12)
    plt.savefig(os.path.join(model_dir, label_name + '_' + split + '_multi_roc.png'))
    plt.close()
    
    return roc_auc['micro'], figure

# PRC for Binary Classification 
def plot_pr(y_true, y_score, model_dir, label_name, sample_weight = None, split='test'):
    y_true = y_true[:,1]
    y_score = y_score[:,1]
    prn, rec, _ = precision_recall_curve(y_true, y_score, sample_weight=sample_weight)
    prc_auc = average_precision_score(y_true, y_score, sample_weight=sample_weight)
    colors = ['darkorange', 'deeppink']
    #Plot ROC
    lw =2
    #Plot PRC
    figure = plt.figure(2, figsize=(10, 10))
    plt.plot(rec, prn, color=colors[1], lw=lw,
             label='{0} (area = {1:0.2f})'
             ''.format(1, prc_auc))

    plt.plot([0, 1], [np.mean(y_true), np.mean(y_true)], 'k--', lw=lw)
    plt.text(1, np.mean(y_true), str(np.round(np.mean(y_true), 2)), fontsize=16)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Multi-class Precision Recall Curve')
    plt.legend(loc="best", fontsize=16)
    plt.savefig(os.path.join(model_dir, label_name + '_' + split + '_binary_prc.png'))
    plt.close()
    
    return prc_auc, figure    

# PRC for Multiclass Classification
def plot_pr_multi(y_true, y_score, model_dir, label_name, sample_weight = None, split='test'):
    num_classes = y_true.shape[1]
    #%% Compute ROC curve and ROC area for each class
    prn = dict()
    rec = dict()
    prc_auc = dict()
    for i in range(num_classes):
        #PRC
        prn[i], rec[i], _ = precision_recall_curve(y_true[:, i], y_score[:, i], sample_weight=sample_weight)
        prc_auc[i] = average_precision_score(y_true[:, i], y_score[:, i], sample_weight=sample_weight)
    # PRC
    prn["micro"], rec["micro"], _ = precision_recall_curve(y_true.ravel(), y_score.ravel())
    prc_auc["micro"] = average_precision_score(y_true, y_score, average='micro', sample_weight=sample_weight)
                                               
    all_prn = np.unique(np.concatenate([prn[i] for i in range(num_classes)]))     

    ## PRC
    mean_rec = np.zeros_like(all_prn)
    for i in range(num_classes):
        mean_rec += interp(all_prn, prn[i], rec[i])

    mean_rec /= num_classes   

    ## PRC
    prn["macro"] = all_prn
    rec["macro"] = mean_rec
    prc_auc["macro"] = average_precision_score(y_true, y_score, average='macro', sample_weight=sample_weight)

    #%% Plot Curves
    lw = 2 
    #Plot all PRC Curves
    figure = plt.figure(2, figsize=(10, 10))
    plt.plot(rec["micro"], prn["micro"],
             label='micro (area = {0:0.2f})'
                   ''.format(prc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    plt.plot(rec["macro"], prn["macro"],
             label='macro (area = {0:0.2f})'
                   ''.format(prc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] + ['aqua', 'darkorange', 'cornflowerblue']
    for i, color in zip(range(num_classes), colors):
        plt.plot(rec[i], prn[i], color=color, lw=lw,
                 label='{0} (area = {1:0.2f}) (n={2})'
                 ''.format(i, prc_auc[i], int(y_true[:, i].sum())))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.title('Multi-class Precision Recall Curve')
    plt.legend(loc="best", fontsize=12)
    plt.savefig(os.path.join(model_dir, label_name + '_' + split + '_multi_prc.png'))
    plt.close()
                                               
    return prc_auc['micro'], figure                                           

# Plots Regression Results (Density Plot)
def plot_reg(y_test, y_pred, model_dir, label_name, sample_weight = None, split = 'test'):
    y_pred_med = np.median(y_pred, axis = 0)
    figs = []
    r2 = []
    mse = []
    mae = []
    if len(y_test.shape) == 1:
        y_test = np.expand_dims(y_test, -1)
        
    for i in range(y_test.shape[1]):
        u = np.mean(y_test[:,i]) 
        o = np.std(y_test[:,i])
        l_min, l_max = u - 2*o, u + 2*o 
        figs.append(plt.figure(figsize=(8,8), dpi=300))
        plt.plot([l_min,l_max], [l_min,l_max], color='white', lw=0.5)
        r2.append(r2_score(y_test[:,i], y_pred[:,i], sample_weight=sample_weight))
        mse.append(mean_squared_error(y_test[:,i], y_pred[:,i], sample_weight=sample_weight))
        mae.append(mean_absolute_error(y_test[:,i], y_pred[:,i], sample_weight=sample_weight))
        plt.hist2d(y_test[:,i], y_pred[:,i], bins=75, 
                   range=[[l_min,l_max], [l_min,l_max]], vmax=len(y_test[:,i])/1000, 
                   weights = sample_weight)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.colorbar(shrink=0.75)
        plt.xlabel('actual')
        plt.ylabel('predicted')
        plt.title('{}, R2={}, MSE={}, MAE={}'.format(i, r2[i], mse[i], mae[i]))
        plt.savefig(os.path.join(model_dir, '{}_{}_{}_density.png'.format(label_name, split, i)))
        plt.close()
    return figs, r2, mse, mae

# Accuracy Function
def multi_acc(y_test, y_pred, sample_weight = None):
    y_test = np.argmax(y_test, 1)
    y_pred = np.argmax(y_pred, 1)
    acc = accuracy_score(y_test, y_pred, sample_weight=sample_weight)
    return acc

# Plots Time to Event Results (Density Plot)
def plot_tte(y_test, y_pred_time, y_pred_cdf, times, model_dir, label_name, split = 'test'):
    '''
    y_test: Includes times (censoring or failure) --> n x 2
    y_pred_time: Includes predicted failure time --> n x 1
    y_pred_cdf: F(t | X) for each t represented by a bin --> n x n_bin
    '''
    
    U = y_test[:,0]
    delta = y_test[:,1].astype(bool)
    y_train_surv = np.array([(x, y) for x,y in zip(delta, U)], dtype = [('Delta', '?'), ('U', '<f8')])
        
    #Unweighted Concordance
    cindex_uw_cdf, _, _, _, _ = concordance_index_censored(delta, U, y_pred_cdf[:,-2])
    cindex_uw_time, _, _, _, _ = concordance_index_censored(delta, U, -y_pred_time)
    
    mask = U < max(U)-5
    y_test_surv = y_train_surv[mask]
    y_pred_time = y_pred_time[mask]
    y_pred_cdf = y_pred_cdf[mask]
    
    #IPCW KM Concordance
    cindex_ipcw_cdf, _, _, _, _ = concordance_index_ipcw(y_train_surv, y_test_surv, y_pred_cdf[:,-2], tau=max(U))
    cindex_ipcw_time, _, _, _, _ = concordance_index_ipcw(y_train_surv, y_test_surv, -y_pred_time, tau=max(U))
    
    #IPCW KM AUC@t
    auc, mean_auc = cumulative_dynamic_auc(y_train_surv, y_test_surv, y_pred_cdf, times)
    
    #IPCW KM BS@t
    _, brier_scores = brier_score(y_train_surv, y_test_surv, y_pred_cdf, times)
    ibs = integrated_brier_score(y_train_surv, y_test_surv, y_pred_cdf, times)
    
    # Plot AUC and BS Over Time
    fig, axs = plt.subplots(1, 2, figsize=(14*.8, 14*.7))
    ## AUC
    axs[0].plot(times, auc, marker="o", color='blue')
    axs[0].set_xlabel("Days From Drug Initiation")
    axs[0].set_ylabel("time-dependent AUC")
    axs[0].axhline(mean_auc, color='blue', linestyle="--")
    ## Brier Score
    axs[1].plot(times, brier_scores, marker="o", color='red')
    axs[1].set_xlabel("Days From Drug Initiation")
    axs[1].set_ylabel("time-dependent Brier Score")
    axs[1].axhline(ibs, color='red', linestyle="--")
    
    plt.title('iAUC={}, iBS={}, c-index(ipcw)={}, c-index(uw)={}'.format(mean_auc,ibs,cindex_ipcw_cdf,cindex_uw_cdf))
    plt.savefig(os.path.join(model_dir, '{}_{}_tte.png'.format(label_name, split)))
    plt.close()
    
    return fig, cindex_uw_cdf, cindex_uw_time, cindex_ipcw_cdf, cindex_ipcw_time, auc, mean_auc, brier_scores, ibs 

# Generates Results for Neural Network Models
def eval_nn(model, test_data, model_dir, label_dict, split = 'test', **params):
    # Scenarios:
    # Single Label
    # Multiple Labels
    
    # Label Type:
    # Classification
    # Binary Classfication
    # Multi-class Classification
    # Metrics:
    # ROC, PR, ACC
    
    # Regression
    # Single Regression
    # Multiple Regression
    # Metrics:
    # Density Plot
    # R2, 
    
    # Get y_pred 
    y_preds = {}
    for label in label_dict.keys():
        model_label = Model(model.inputs, model.get_layer(label).output)
#         model_label = model
        y_preds[label] = model_label.predict(test_data)
    
    # Get Sample Weights
    # Hao 06/27/22 change to try here if weight_column doesnt exist:
    # Will need to fix a lot of things
    try:
        if params['weight_column'] is not None:
            sample_weight = []
        else:
            sample_weight = None
    except:
        sample_weight = None
    
    # Get y_test
    y_tests = {k:[] for k in label_dict.keys()}
    for x in test_data:
        for k in label_dict.keys():
            y_tests[k].append(x[1][k])
        if params['weight_column'] is not None:
            sample_weight.append(x[2])
        
    y_tests = {k:np.concatenate(v, 0) for k,v in y_tests.items()}
    sample_weight = np.concatenate(sample_weight)
    
    # evaluate by label sets
    out = {}
    for label_name, l_prop in label_dict.items():
        y_pred = y_preds[label_name]
        y_test = y_tests[label_name]
        
        if l_prop['loss_fn'] in ['mean_absolute_error', 'mean_squared_error', 'logcosh']:
            dense, r2, mse, mae = plot_reg(y_test, y_pred, model_dir, label_name, sample_weight, split=split)
            out[label_name] = {'r2':np.mean(r2), 'mse':np.mean(mse), 
                               'mae':np.mean(mae), 'density_plot':dense}
        elif l_prop['loss_fn'] in ['failure', 'failure_only', 'censorship']:
            #Get Times
            times = l_prop['boundaries'][1:-1]
            #Get Prediction Time from Probabilities
            center_times = [(x + y)/2 for x, y in zip(l_prop['boundaries'][0:-1],l_prop['boundaries'][1:])]
            y_pred_bin = y_pred.argmax(1)
            y_pred_time = np.array([center_times[y] for y in y_pred_bin])
            #Get CDF for Each Prediction
            y_pred_cdf = np.cumsum(y_pred, axis=1)[:,:-1]
            
            if l_prop['loss_fn'] == 'censorship':
                y_test[:,1] = np.where(y_test[:,1]==0, 1, 0)
            
            (auc_bs_t,cindex_uw_cdf, cindex_uw_time, cindex_ipcw_cdf, 
             cindex_ipcw_time, auc, mean_auc, brier_scores, ibs ) = plot_tte(y_test, y_pred_time, y_pred_cdf, times, 
                                                                             model_dir, label_name, split)
            out[label_name] = {'cindex_uw_time':cindex_uw_time, 'cindex_uw_cdf':cindex_uw_cdf, 
                               'cindex_ipcw_time':cindex_ipcw_time, 'cindex_ipcw_cdf':cindex_ipcw_cdf, 
                               'auc': auc, 'mean_auc':mean_auc, 
                               'brier_scores':brier_scores, 'ibs':ibs, 
                               'auc_bs_plot':auc_bs_t}
            
        else:
            if y_test.shape[1] > 2:
                roc_auc, roc = plot_auc_multi(y_test, y_pred, model_dir, label_name, sample_weight, split=split)
                prc_auc, prc = plot_pr_multi(y_test, y_pred, model_dir, label_name, sample_weight, split=split)
                
                acc = multi_acc(y_test, y_pred)   
                out[label_name] = {'roc_auc': roc_auc, 'prc_auc':prc_auc, 
                                  'roc':roc, 'prc':prc, 'acc':acc}

            else:
                # Get Curves
                roc_auc, roc = plot_auc(y_test, y_pred, model_dir, label_name, sample_weight, split=split)
                prc_auc, prc = plot_pr(y_test, y_pred, model_dir, label_name, sample_weight, split=split)
                # Get AUCs w/ CI
                m_auroc, U_auroc, L_auroc = bootstrap_ci(y_test[:,1], y_pred[:,1], roc_auc_score, sample_weight=sample_weight)
                m_auprc, U_auprc, L_auprc = bootstrap_ci(y_test[:,1], y_pred[:,1], average_precision_score, sample_weight=sample_weight)
                # Get Sensitivity and Specificity @ PPV
                m_sens, U_sens, L_sens = [], [], []
                m_spec, U_spec, L_spec = [], [], []
                for P in np.linspace(.5, .9, 5):
                    SE, U_SE, L_SE = bootstrap_ci(y_test[:,1], y_pred[:,1], SENS_at_PPV, sample_weight = sample_weight, ppv=P)
                    SP, U_SP, L_SP = bootstrap_ci(y_test[:,1], y_pred[:,1], SPEC_at_PPV, sample_weight = sample_weight, ppv=P)
                    m_sens.append(SE)
                    U_sens.append(U_SE)
                    L_sens.append(L_SE)
                    m_spec.append(SP)
                    U_spec.append(U_SP)
                    L_spec.append(L_SP)
                    
                acc = multi_acc(y_test, y_pred)   
                out[label_name] = {'roc_auc': roc_auc, 'prc_auc':prc_auc, 
                                  'roc':roc, 'prc':prc, 'acc':acc, 
                                  'm_auroc':m_auroc, 'U_auroc':U_auroc, 'L_auroc':L_auroc, 
                                  'm_auprc':m_auprc, 'U_auprc':U_auprc, 'L_auprc':L_auprc}
                
                for i, P in enumerate(np.linspace(.5, .9, 5)):
                    out[label_name]['m_sens_'+ str(P)] = m_sens[i]
                    out[label_name]['U_sens_'+ str(P)] = U_sens[i]
                    out[label_name]['L_sens_'+ str(P)] = L_sens[i]
                    out[label_name]['m_spec_'+ str(P)] = m_spec[i]
                    out[label_name]['U_spec_'+ str(P)] = U_spec[i]
                    out[label_name]['L_spec_'+ str(P)] = L_spec[i]
            
    return out

# Generates Results for Sklearn Learning Models
def eval_ml(model, test_data, model_dir, label_dict, split='test'):
    # Handles only binary classification for now
    y_score = model.predict_proba(test_data[0])
    y_test = test_data[1]
    
    out = {}
    for label_name, _ in label_dict.items():    
        roc_auc, roc = plot_auc(y_test, y_score, model_dir, label_name, split)
        prc_auc, prc = plot_pr(y_test, y_score, model_dir, label_name, split)
        # Get AUCs w/ CI
        m_auroc, U_auroc, L_auroc = bootstrap_ci(y_test[:,1], y_score[:,1], roc_auc_score)
        m_auprc, U_auprc, L_auprc = bootstrap_ci(y_test[:,1], y_score[:,1], average_precision_score)
        # Get Sensitivity and Specificity @ PPV
        m_sens, U_sens, L_sens = [], [], []
        m_spec, U_spec, L_spec = [], [], []
        for P in np.linspace(.5, .9, 5):
            SE, U_SE, L_SE = bootstrap_ci(y_test[:,1], y_score[:,1], SENS_at_PPV, ppv=P)
            SP, U_SP, L_SP = bootstrap_ci(y_test[:,1], y_score[:,1], SPEC_at_PPV, ppv=P)
            m_sens.append(SE)
            U_sens.append(U_SE)
            L_sens.append(L_SE)
            m_spec.append(SP)
            U_spec.append(U_SP)
            L_spec.append(L_SP)

        acc = multi_acc(y_test, y_score)   
        out[label_name] = {'roc_auc': roc_auc, 'prc_auc':prc_auc, 
                          'roc':roc, 'prc':prc, 'acc':acc, 
                          'm_auroc':m_auroc, 'U_auroc':U_auroc, 'L_auroc':L_auroc, 
                          'm_auprc':m_auprc, 'U_auprc':U_auprc, 'L_auprc':L_auprc}

        for i, P in enumerate(np.linspace(.5, .9, 5)):
            out[label_name]['m_sens_'+ str(P)] = m_sens[i]
            out[label_name]['U_sens_'+ str(P)] = U_sens[i]
            out[label_name]['L_sens_'+ str(P)] = L_sens[i]
            out[label_name]['m_spec_'+ str(P)] = m_spec[i]
            out[label_name]['U_spec_'+ str(P)] = U_spec[i]
            out[label_name]['L_spec_'+ str(P)] = L_spec[i]
            
    return out

# Get Bootstrap Confidence Interval
def bootstrap_ci(y_test, y_score, eval_fn, sample_weight, **eval_args):
    results = []
    for i in range(100):
        re_y_test, re_y_score, re_sw = resample(*(y_test, y_score, sample_weight), replace=False, 
                                         n_samples=int(.66*y_test.shape[0]), random_state=i)
        
        results.append(eval_fn(re_y_test, re_y_score, sample_weight = re_sw, **eval_args))
    
    results = np.sort(np.array(results))
    mean = results.mean()
    U, L = results[94], results[4]
    
    return mean, U, L

# Get Sensitivity at x% PPV
def SENS_at_PPV(y_test, y_score, ppv, sample_weight): 
    prn, rec, thresholds = precision_recall_curve(y_test, y_score, sample_weight=sample_weight)
    thresholds = np.append(thresholds, [1])
    #Get Sensitivitiy
    idx = abs(prn-ppv).argmin()
    sens = rec[idx]
    
    return sens
    
    
# Get Specificity at x% PPV
def SPEC_at_PPV(y_test, y_score, ppv, sample_weight): 
    prn, rec, thresholds_prc = precision_recall_curve(y_test, y_score, sample_weight=sample_weight)
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_score, sample_weight=sample_weight)
    thresholds_prc = np.append(thresholds_prc, [1])
    #Get Threshold
    idx = abs(prn-ppv).argmin()
    threshold = thresholds_prc[idx]    
    #Get Specificity
    idx = abs(thresholds_roc-threshold).argmin()
    spec = 1 - fpr[idx] 
   
    return spec
    
    
# Generates Results for XGBoost
def eval_xgb(model, test_data, model_dir, label_dict, split='test', **params):
    # New changes, accomodate patients weights
    #1. Make DMatrix
    y_test = test_data[1]
    
    # Get Sample Weights
    if params['weight_column'] is not None:
        sample_weight = test_data[2]
    else:
        sample_weight = None
        
    test_data = xgb.DMatrix(test_data[0])
 
    
    # Handles only binary classification for now
    #2. Predict
    y_score = model.predict(test_data)
    
    #3. Generate Results
    out = {}
    for label_name, l_prop in label_dict.items(): 
        if l_prop['reformat'] == 'one_hot' or type(l_prop['reformat']) == int or l_prop['reformat'] == 'extreme':
            if y_test.shape[1] > 2:
                roc_auc, roc = plot_auc_multi(y_test, y_score, model_dir, label_name, sample_weight, split)
                prc_auc, prc = plot_pr_multi(y_test, y_score, model_dir, label_name, sample_weight, split)
                acc = multi_acc(y_test, y_score)
                out[label_name] = {'roc_auc': roc_auc, 'prc_auc':prc_auc, 
                                  'roc':roc, 'prc':prc, 'acc':acc}
            else:
                y_score = np.tile(np.expand_dims(y_score, -1), [1,2]) 
                roc_auc, roc = plot_auc(y_test, y_score, model_dir, label_name, sample_weight, split)
                prc_auc, prc = plot_pr(y_test, y_score, model_dir, label_name, sample_weight, split)
                # Get AUCs w/ CI
                m_auroc, U_auroc, L_auroc = bootstrap_ci(y_test[:,1], y_score[:,1], roc_auc_score, sample_weight=sample_weight)
                m_auprc, U_auprc, L_auprc = bootstrap_ci(y_test[:,1], y_score[:,1], average_precision_score, sample_weight=sample_weight)
                # Get Sensitivity and Specificity @ PPV
                m_sens, U_sens, L_sens = [], [], []
                m_spec, U_spec, L_spec = [], [], []
                for P in np.linspace(.5, .9, 5):
                    SE, U_SE, L_SE = bootstrap_ci(y_test[:,1], y_score[:,1], SENS_at_PPV, sample_weight = sample_weight, ppv=P)
                    SP, U_SP, L_SP = bootstrap_ci(y_test[:,1], y_score[:,1], SPEC_at_PPV, sample_weight = sample_weight, ppv=P)
                    m_sens.append(SE)
                    U_sens.append(U_SE)
                    L_sens.append(L_SE)
                    m_spec.append(SP)
                    U_spec.append(U_SP)
                    L_spec.append(L_SP)

                acc = multi_acc(y_test, y_score)   
                out[label_name] = {'roc_auc': roc_auc, 'prc_auc':prc_auc, 
                                  'roc':roc, 'prc':prc, 'acc':acc, 
                                  'm_auroc':m_auroc, 'U_auroc':U_auroc, 'L_auroc':L_auroc, 
                                  'm_auprc':m_auprc, 'U_auprc':U_auprc, 'L_auprc':L_auprc}

                for i, P in enumerate(np.linspace(.5, .9, 5)):
                    out[label_name]['m_sens_'+ str(P)] = m_sens[i]
                    out[label_name]['U_sens_'+ str(P)] = U_sens[i]
                    out[label_name]['L_sens_'+ str(P)] = L_sens[i]
                    out[label_name]['m_spec_'+ str(P)] = m_spec[i]
                    out[label_name]['U_spec_'+ str(P)] = U_spec[i]
                    out[label_name]['L_spec_'+ str(P)] = L_spec[i]

        else:
            y_score = np.tile(np.expand_dims(y_score, -1), [1,2])
            dense, r2, mse, mae = plot_reg(y_test, y_score, model_dir, label_name, split)
            out[label_name] = {'r2':np.mean(r2), 'mse':np.mean(mse), 
                               'mae':np.mean(mae), 'density_plot':dense}
          
    return out

# Master Evaluation Function
def eval_model(model, test_data, model_dir, label_dict, split='test', **params):
    if params['model_type'] == 'nn':
        return eval_nn(model, test_data, model_dir, label_dict, split, **params)
    elif params['model_type'] in ['lr']:
        return eval_ml(model, test_data, model_dir, label_dict, split)
    elif params['model_type'] in ['xgb']:
        return eval_xgb(model, test_data, model_dir, label_dict, split, **params)