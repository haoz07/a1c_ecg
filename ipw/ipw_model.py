#Import Packages
import pandas as pd
import numpy as np
import os 
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from tqdm import tqdm
import pickle
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_consistent_length
from sklearn.preprocessing import label_binarize

data_dir = '.'
data_df = pd.read_csv(os.path.join(data_dir,'cohort.csv'))

train_idx = []
test_idx = []

X = data_df.iloc[:, 11:-2].to_numpy()
y = data_df['label'].to_numpy()

X_train = X[train_idx,:]
y_train = y[train_idx]

X_test = X[test_idx,:]
y_test = y[test_idx]


for i in [X_train, y_train, X_test, y_test]:
    print(i.shape)

parameters = {'n_jobs':[15],
              'objective':['binary:logistic'],
              'learning_rate': [0.05], #so called `eta` value
              'max_depth': [3, 5, 10],
              'subsample': [0.8],
              'colsample_bytree': [0.8],
              'n_estimators': [100, 1000], #number of trees, change it to 1000 for better results
              'seed': [420]}
def ece(y_true, y_prob, *, normalize=False, n_bins=10,
        strategy='uniform'):
    """Compute true and predicted probabilities for a calibration curve.
    The method assumes the inputs come from a binary classifier, and
    discretize the [0, 1] interval into bins.
    Calibration curves may also be referred to as reliability diagrams.
    Read more in the :ref:`User Guide <calibration>`.
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True targets.
    y_prob : array-like of shape (n_samples,)
        Probabilities of the positive class.
    normalize : bool, default=False
        Whether y_prob needs to be normalized into the [0, 1] interval, i.e.
        is not a proper probability. If True, the smallest value in y_prob
        is linearly mapped onto 0 and the largest one onto 1.
    n_bins : int, default=5
        Number of bins to discretize the [0, 1] interval. A bigger number
        requires more data. Bins with no samples (i.e. without
        corresponding values in `y_prob`) will not be returned, thus the
        returned arrays may have less than `n_bins` values.
    strategy : {'uniform', 'quantile'}, default='uniform'
        Strategy used to define the widths of the bins.
        uniform
            The bins have identical widths.
        quantile
            The bins have the same number of samples and depend on `y_prob`.
    Returns
    -------
    prob_true : ndarray of shape (n_bins,) or smaller
        The proportion of samples whose class is the positive class, in each
        bin (fraction of positives).
    prob_pred : ndarray of shape (n_bins,) or smaller
        The mean predicted probability in each bin.
    References
    ----------
    Alexandru Niculescu-Mizil and Rich Caruana (2005) Predicting Good
    Probabilities With Supervised Learning, in Proceedings of the 22nd
    International Conference on Machine Learning (ICML).
    See section 4 (Qualitative Analysis of Predictions).
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.calibration import calibration_curve
    >>> y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    >>> y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9,  1.])
    >>> prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=3)
    >>> prob_true
    array([0. , 0.5, 1. ])
    >>> prob_pred
    array([0.2  , 0.525, 0.85 ])
    """
    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)
    check_consistent_length(y_true, y_prob)

    if normalize:  # Normalize predicted values into interval [0, 1]
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
    elif y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1] and normalize is "
                         "set to False.")

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError("Only binary classification is supported. "
                         "Provided labels %s." % labels)
    y_true = label_binarize(y_true, classes=labels)[:, 0]

    if strategy == 'quantile':  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
        bins[-1] = bins[-1] + 1e-8
    elif strategy == 'uniform':
        bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    else:
        raise ValueError("Invalid entry to 'strategy' input. Strategy "
                         "must be either 'quantile' or 'uniform'.")

    binids = np.digitize(y_prob, bins) - 1

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    
    ece = np.sum(np.abs(prob_true - prob_pred) * (bin_total[nonzero] / len(y_true)))
    
    return ece

ECE_scorer = make_scorer(ece, greater_is_better=False, needs_proba=True)

xgb_model = xgb.XGBClassifier()

scoring = {'auroc': 'roc_auc',
           'acc': 'accuracy', 
           'auprc': 'average_precision',
           'ece': ECE_scorer}

clf = GridSearchCV(xgb_model, parameters, n_jobs=1, 
                   scoring=scoring,
                   verbose=10, refit='auroc')

xgb_model = xgb.XGBClassifier()
best_params = CV_propensity_results[CV_propensity_results['rank_test_ece']==1].params.iloc[0]
xgb_model.set_params(**best_params)

xgb_model.fit(X_train, y_train, 
              eval_set=[(X_train[:200000], y_train[:200000]), (X_test, y_test)],
              eval_metric='logloss',
              verbose=True)


xgb_model.save_model(os.path.join(data_dir, 'xgb_propensity_model_ECG.json'))