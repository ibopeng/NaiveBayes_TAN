"""
Project: Naive Bayes and TAN

10 fold cross validation

@ Author: Bo Peng
@ University of Wisconsin - Madison
"""

import bayes_func as bf
import scipy.io.arff as af
import numpy as np
import scipy.stats as stats

"""Load data"""

# load data
instanceset_cv, meta_data = af.loadarff('lymph_cv.arff')
cv_labels = [ins[-1] for ins in instanceset_cv]

# relative parameters
var_names = meta_data.names()
var_ranges = [meta_data[name][1] for name in var_names]

label_range = var_ranges[-1]
num_instance = len(instanceset_cv)

""" 10 folds cross validation"""
num_fold = 10
unit_sz_fold = num_instance / num_fold
sz_remain = num_instance % num_fold

sz_folds = []  # size of each fold
fold_idxs = []
idx_start = 0
idx_end = 0
for i in range(num_fold):
    if i < sz_remain:
        sz_folds.append(unit_sz_fold + 1)
    else:
        sz_folds.append(unit_sz_fold)
    # set instances indices for each fold
    idx_end = idx_start + sz_folds[i]
    fold_idxs.append(range(idx_start, idx_end))
    idx_start = idx_end

# cross validation
acc_nb = []  # accuracy for Naive Bayes
acc_tan = []  # accuracy for TAN
acc_delta = []  # accuracy difference between Naive Bayes and TAN
for i in range(num_fold):
    # extract training and validation set
    instanceset_trn = []
    instanceset_val = instanceset_cv[fold_idxs[i]]
    val_labels = [cv_labels[idx] for idx in fold_idxs[i]]
    for j in range(num_instance):
        if j not in fold_idxs[i]:
            instanceset_trn.append(instanceset_cv[j])
    # Predict and compute accuracy
    # Naive Bayes
    pred_nb, _ = bf.testset_predict_nb(instanceset_trn, instanceset_val, var_ranges, label_range)

    num_correct_pred_nb = bf.comp_num_correct_predict(val_labels, pred_nb)
    acc_nb.append(1.0 * num_correct_pred_nb / len(fold_idxs[i]))

    # TAN
    # generate the edge weight graph
    weight_graph = bf.edge_weight_graph(instanceset_trn, var_ranges, label_range)
    # compute the new vertex list
    V_new = bf.prim_mst(weight_graph)
    # test set prediction
    pred_tan, _ = bf.testset_prediction_tan(instanceset_trn, instanceset_val, var_ranges, label_range, V_new)
    # number of correct prediction
    num_correct_pred_tan = bf.comp_num_correct_predict(val_labels, pred_tan)
    acc_tan.append(1.0 * num_correct_pred_tan / len(fold_idxs[i]))

    # difference between TAN and Naive Bayes prediction accuracy
    acc_delta.append(acc_tan[i] - acc_nb[i])

    print("Fold {0}".format(i+1))

"""1. Sample Mean"""
acc_delta = np.array(acc_delta)
sample_mean = np.mean(acc_delta)
print('Sample means = ' + str(sample_mean))

"""2. t statistic"""
# freedom: n-1
sample_variance = np.var(acc_delta, ddof=1)
# Null hypothesis is: sample_mean = 0, i.e., no accuracy difference between TAN and NB
t = (sample_mean - 0) / np.sqrt(sample_variance / num_fold)
print('t statistic = ' + str(t))

t_tatistic, p_value = stats.ttest_1samp(acc_delta, 0.0)
print(t_tatistic)
print(p_value)