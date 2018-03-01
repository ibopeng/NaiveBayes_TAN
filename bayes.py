"""
Project: Naive Bayes and TAN

Main file to run Naive Bayes or TAN algorithm

@ Author: Bo Peng
@ University of Wisconsin - Madison
"""


import bayes_func as bf
import scipy.io.arff as af

# load training data
instance_data_trn, meta_data = af.loadarff('lymph_train.arff')
var_ranges = [meta_data[name][1] for name in meta_data.names()]
label_range = var_ranges[-1]

prob_per_class = bf.prob_per_class(instance_data_trn, label_range)

aa = bf.test(instance_data_trn, var_ranges, label_range)

print(len(instance_data_trn))