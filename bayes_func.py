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


def comp_Pr_perClass(instance_data, label_range):
    """
    Compute the probability of each class
    :param instance_data: training instances
    :param label_range: possible class values
    :return: a list of probability of each class
    """

    # number of instances
    num_instance = len(instance_data)

    # probability of each class
    Pr_perClass = []
    # loop over the label range
    for lb in label_range:
        # compute the number of instances of class lb
        num_ins_lb = sum(np.array([ins[-1] == lb for ins in instance_data]))
        # using laplace estimates to compute probability
        Pr_perClass.append((num_ins_lb + 1.0) / (num_instance + len(label_range)))

    return Pr_perClass

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


def comp_cndPr_oneVar_oneLabel(var_idx, instance_with_lb, var_range):
    """
    Compute the probability of each possible value of the ith variable given the instance label == lb
    :param var_idx: the index of this variable in the list of var_ranges
    :param instance_with_lb: instances with the same label == lb
    :param var_range: range of the (var_idx)th variable
    :return: conditional probability of each possible value of the (var_idx)th variable given the instance label == lb
    """

    num_instance_each_val = []
    for var_val in var_range:
        num_instance_this_val = sum(np.array([ins[var_idx] == var_val for ins in instance_with_lb]))
        num_instance_each_val.append(num_instance_this_val)

        # compute the probability of each possible value via laplace smoothing
    cndPr_eachVar_val = [(num+1.0) / (len(instance_with_lb)+len(var_range)) for num in num_instance_each_val]

    return cndPr_eachVar_val


def comp_cndPr_allVars_oneLabel(instance_with_lb, var_ranges):

    cndPr_allVars_oneLabel = []
    for idx in range(len(var_ranges)-1): # note that the last variable is <class>, which should be excluded
        cndPr_allVars_oneLabel.append(comp_cndPr_oneVar_oneLabel(idx, instance_with_lb, var_ranges[idx]))

    return cndPr_allVars_oneLabel


def comp_cndPr_allVars_allLabels(instance_data, var_ranges, label_range):
    """
    Compute conditional probability for each variable in each subset with different lables
    :param instance_data: training instances
    :param var_ranges: [list] range of all variables
    :param label_range: range of all labels
    :return: a list, each element contains conditional probability of each variable given one class label
    """

    instance_split = split_instance_on_label(instance_data, label_range)
    cndPr_allVars_allLabels = [comp_cndPr_allVars_oneLabel(ins_splt, var_ranges) for ins_splt in instance_split]

    return cndPr_allVars_allLabels

def instance_predict(instance_test, cndPr_allVars_allLabels, Pr_perLabel, var_ranges, label_range):
    """
    Predict the class of the test instance
    :param test_instance: single test instance
    :param cndPr_allVars_allLabels: [list] conditional probability for each variable given different class lables
    :return: the class of the test instance
    """

    # for current instance, loop over all variables to compute the conditional probability give different class labels
    # number of classes
    num_label = len(cndPr_allVars_allLabels)

    # conditional probability for each label
    postPr_allLabels_test = []
    # loop over all subsets
    for i in range(num_label):
        cndPr_allVars_oneLabel = cndPr_allVars_allLabels[i]
        num_var = len(cndPr_allVars_oneLabel) # number of variables

        cndPr_thisLabel_eachVar_test = []

        # loop over all variables
        for j in range(num_var):
            this_var_range = var_ranges[j] # range of this variables
            cndPr_oneVar_oneLabel = cndPr_allVars_oneLabel[j] # conditional probability of each value of this variable
            cndPr_this_val = cndPr_oneVar_oneLabel[this_var_range.index(instance_test[j])]
            cndPr_thisLabel_eachVar_test.append(cndPr_this_val)

        # compute the posteriori probability of this test instance being classified as the ith label
        # posteriori Pr = P(yi) * [P(a1|yi) * P(a2|yi)*... *P(an|yi)]
        postPr = Pr_perLabel[i] * np.product(np.array(cndPr_thisLabel_eachVar_test))
        postPr_allLabels_test.append(postPr)

    # normalize the posteriori probability
    postPr_allLabels_test = np.array(postPr_allLabels_test)
    postPr_sum = sum(postPr_allLabels_test)
    postPr_allLabels_test = [postPr_lb / postPr_sum for postPr_lb in postPr_allLabels_test]

    # get the index of the max conditional probability
    idx_max_postPr = np.argmax(np.array(postPr_allLabels_test))
    # get the label corresponding to this idx
    test_predict = label_range[idx_max_postPr]
    test_max_postPr = postPr_allLabels_test[idx_max_postPr]

    return test_predict, test_max_postPr


def testset_predict(instance_data_trn, instance_data_test, var_ranges, label_range):

    # get conditional probability for each variable given different class lables
    cndPr_allVars_allLabels = comp_cndPr_allVars_allLabels(instance_data_trn, var_ranges, label_range)

    # get probability of each class
    Pr_perLabel = comp_Pr_perClass(instance_data_trn, label_range)

    # prediction and the max posteriori probability of the entire test data set
    testset_pred = []
    testset_max_postPr = []

    for ins_test in instance_data_test:
        ins_test_predict, ins_test_max_postPr = instance_predict(ins_test, cndPr_allVars_allLabels, Pr_perLabel, var_ranges, label_range)
        testset_pred.append(ins_test_predict)
        testset_max_postPr.append(ins_test_max_postPr)

    return testset_pred, testset_max_postPr

def comp_num_correct_predict(instance_data_label, instance_data_prediction):
    """
    Compute the number of correct prediction
    :param instance_data_label:
    :param instance_data_prediction:
    :return: number of correct prediction
    """

    if len(instance_data_label) == len(instance_data_prediction):
        num_instance = len(instance_data_label)
        num_correct_pred = sum(np.array([instance_data_label[i] == instance_data_prediction[i] for i in range(num_instance)]))
        return num_correct_pred
    else:
        return 0