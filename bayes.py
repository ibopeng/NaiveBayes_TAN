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