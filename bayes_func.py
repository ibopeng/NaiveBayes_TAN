"""
Project: Naive Bayes and TAN

Functions to implemetn Naive Bayes and TAN

@ Author: Bo Peng
@ University of Wisconsin - Madison
"""


import numpy as np
import scipy as sp
import sys


def read_cmdln_arg():
    """command line operation"""
    if len(sys.argv) != 4:
        sys.exit("Incorrect arguments...")
    else:
        filename_trn = str(sys.argv[1])
        filename_test = str(sys.argv[2])
        n_t = str(sys.argv[3])
    return filename_trn, filename_test, n_t


def prob_per_class(instance_data, label_range):
    """
    Compute the probability of each class
    :param instance_data: training instances
    :param label_range: possible class values
    :return: a list of probability of each class
    """

    # number of instances
    num_instance = len(instance_data)

    # probability of each class
    prob_per_class = []
    # loop over the label range
    for lb in label_range:
        # compute the number of instances of class lb
        num_ins_lb = sum(np.array([ins[-1] == lb for ins in instance_data]))
        prob_per_class.append(1.0 * num_ins_lb / num_instance)

    return prob_per_class

def split_instance_on_label(instance_data, label_range):

    instance_split = []
    for lb in label_range:
        # extract instances with class value == lb
        instance_with_lb = []
        for ins in instance_data:
            if ins[-1] == lb:
                instance_with_lb.append(ins)
        instance_split.append(instance_with_lb)

    return instance_split


def condition_prob_each_var(var_idx, instance_with_lb, var_range):
    """
    Compute the probability of each possible value of the ith variable given the instance label == lb
    :param var_idx: the index of this variable in the list of var_ranges
    :param instance_with_lb: instances with the same label == lb
    :param var_range: range of the (var_idx)th variable
    :return: conditional probability of each possible value of the (var_idx)th variable given the instance label == lb
    """

    # number of possible values of this variable
    num_var_val = len(var_range)
    num_ins_each_val = np.zeros(num_var_val)
    condition_prob_each_var_val = np.zeros(num_var_val)
    for ins in instance_with_lb:
        for i in range(num_var_val): # for the ith possible value
            if ins[var_idx] == var_range[i]:
                num_ins_each_val[i] += 1
        # compute the probability of each possible value via laplace smoothing
        condition_prob_each_var_val = [(num+1.0) / (len(instance_with_lb)+num_var_val) for num in num_ins_each_val]

    return condition_prob_each_var_val


def condition_prob_all_vars(instance_with_lb, var_ranges):

    condition_prob_all_var = []
    for idx in range(len(var_ranges)):
        condition_prob_all_var.append(condition_prob_each_var(idx, instance_with_lb, var_ranges))

    return condition_prob_all_var


def test(instance_data, var_ranges, label_range):
    instance_split = split_instance_on_label(instance_data, label_range)

    condition_prob_all_var = condition_prob_all_vars(instance_split[0], var_ranges)

    return condition_prob_all_var