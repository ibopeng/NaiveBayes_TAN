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
    pred_nb, _= bf.testset_predict_nb(instanceset_trn, instanceset_val, var_ranges, label_range)

    num_correct_pred_nb = bf.comp_num_correct_predict(val_labels, pred_nb)
    acc_nb.append(1.0 * num_correct_pred_nb / len(fold_idxs[i]))

    # TAN
    # generate the edge weight graph
    weight_graph = bf.edge_weight_graph(instanceset_trn, var_ranges, label_range)
    # compute the new vertex list
    V_new = bf.prim_mst(weight_graph)
    # test set prediction
    pred_tan, _, _ = bf.testset_prediction_tan(instanceset_trn, instanceset_val, var_ranges, label_range)
    # number of correct prediction
    num_correct_pred_tan = bf.comp_num_correct_predict(val_labels, pred_tan)
    acc_tan.append(1.0 * num_correct_pred_tan / len(fold_idxs[i]))

    # difference between TAN and Naive Bayes prediction accuracy
    acc_delta.append(acc_tan[i] - acc_nb[i])

    print("Fold {0}".format(i+1))

# save accuracy to file
acc_delta = np.array(acc_delta)
acc_nb = np.array(acc_nb)
acc_tan = np.array(acc_tan)
np.savetxt('acc_delta.txt', acc_delta)
np.savetxt('acc_nb.txt', acc_nb)
np.savetxt('acc_tan.txt', acc_tan)

"""1. Sample Mean"""
sm = 0
for ad in acc_delta:
    sm += ad
sample_mean = sm / num_fold
print('Sample mean = {0:.12f}'.format(sample_mean))

"""2. t statistic"""
dof = num_fold - 1  # degree of freedom: n-1

acc_delta_mean_removal = [ad - sample_mean for ad in acc_delta]
sv = 0
for admr in acc_delta_mean_removal:
    sv += admr * admr
sample_variance = sv / dof

# Null hypothesis is: sample_mean = 0, i.e., no accuracy difference between TAN and NB
mu0 = 0.0
t = (sample_mean - mu0) / np.sqrt(sample_variance / num_fold)
print('t statistic = {0:.12f}'.format(t))
print('Degree of freedom = {0}'.format(dof))