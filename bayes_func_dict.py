"""
Project: Naive Bayes and TAN

Functions to implemetn Naive Bayes and TAN

@ Author: Bo Peng
@ University of Wisconsin - Madison
"""


import numpy as np
import scipy as sp
import sys
import TAN_node as tn


def read_cmdln_arg():
    """command line operation"""
    if len(sys.argv) != 4:
        sys.exit("Incorrect arguments...")
    else:
        filename_trn = str(sys.argv[1])
        filename_test = str(sys.argv[2])
        n_t = str(sys.argv[3])
    return filename_trn, filename_test, n_t


def cpr_Xi_given_Cl(instanceset_Cl, var_range, var_idx):
    """
    P(Xi|Ci): Compute the probability of each possible variable value Xi given the class label Ci
    Note the following function can also be used to compute prior probability as long as the input instanceset is the entire set
    :param instance_set:
    :param var_range:
    :param var_idx:
    :return:
    """

    num_instance_Cl = len(instanceset_Cl) # number of instances with label == Ci
    num_var = len(var_range) # number of possible variable value

    # probability of each possible variable value within the variable range
    _cpr_Xi_Cl = {}  # a dictionary, key = var_value

    for var in var_range:
        # compute the number of instances with the [var_idx]th variable == var
        num_instance_var = sum(np.array([ins[var_idx] == var for ins in instanceset_Cl]))
        _cpr_Xi_Cl[var] = (num_instance_var + 1.0) / (num_instance_Cl + num_var)

    return _cpr_Xi_Cl


def split_instanceset_on_label(instance_data, label_range):
    """
    Split original dataset based on different class labels
    :param instance_data:
    :param label_range:
    :return:
    """

    instanceset_split = {}
    for lb in label_range:
        # extract instances with class value == lb
        instance_with_lb = []
        for ins in instance_data:
            if ins[-1] == lb:
                instance_with_lb.append(ins)

        instanceset_split[lb] = instance_with_lb

    return instanceset_split


def cpr_X_given_Cl(instanceset_Cl, var_name_ranges, var_idxs):
    """
    P(X|Ci): Compute conditional probability of all variables given instance label == Ci
    :param instance_with_lb:
    :param var_ranges:
    :return:
    """

    _cpr_X_Ci = {}
    var_names = var_name_ranges.keys()
    for vn in var_names:
        _cpr_X_Ci[vn] = cpr_Xi_given_Cl(instanceset_Cl, var_name_ranges[vn], var_idxs[vn])

    return _cpr_X_Ci


def cpr_X_given_C(instanceset, var_name_ranges, label_range, var_idxs):
    """
    P(X|C): Compute conditional probability for each variable X given different class labels C
    :param instance_data: training instances
    :param var_ranges: [list] range of all variables
    :param label_range: range of all labels
    :return: a list, each element contains conditional probability of each variable given one class label
    """

    instanceset_split = split_instanceset_on_label(instanceset, label_range)
    _cpr_X_C = {}
    for isp in instanceset_split.keys():
        _cpr_X_C[isp] = cpr_X_given_Cl(instanceset_split[isp], var_name_ranges, var_idxs)

    return _cpr_X_C

def instance_predict_nb(instance, _cpr_X_C, _pripr_C, var_name_ranges, label_range, var_idxs):
    """
    Predict the class of the test instance
    :param instance: single test instance
    :param _cpr_X_C: [list] conditional probability for each variable given different class lables
    :param _pripr_C: prior probability of each class labels
    :return: the class of the test instance
    """

    # for current instance, loop over all variables to compute the conditional probability give different class labels

    # P(C|X): posteriori probability for each label
    _pospr_C_X = {}
    # loop over all subsets
    for Ci in label_range:

        _cpr_X_Ci = _cpr_X_C[Ci] # P(X|Ci): conditional probability of all variables given one class label, Ci
        _ins_cpr_X_Ci = {} # [P(Xi|Ci)]: [dict], conditional probability of the instance value of each variable given one class label Ci

        # loop over all variables for this instance
        for var in var_name_ranges.keys():
            # conditional probability for all values of this variable
            _cpr_Xj_Ci = _cpr_X_Ci[var]

            # get the probability of the instance value for this variable
            idx = var_idxs[var]
            ins_var_val = instance[idx]
            _ins_cpr_Xj_Ci = _cpr_Xj_Ci[ins_var_val]
            _ins_cpr_X_Ci[var] = _ins_cpr_Xj_Ci

        # compute the posteriori probability of this test instance being classified as the ith label
        # posteriori Pr: P(Ci|X) = P(Ci) * (P(X1|Ci) * P(X2|Ci)*... *P(Xn|Ci))
        _pospr_Ci_X = _pripr_C[Ci] * np.product(np.array(_ins_cpr_X_Ci.values()))
        _pospr_C_X[Ci] = _pospr_Ci_X

    # normalize the posteriori probability
    _pospr_C_X_norm = {Ci: _pospr_C_X[Ci] / sum(np.array(_pospr_C_X.values())) for Ci in label_range}

    # get the index of the max conditional probability
    test_predict = max(_pospr_C_X_norm.keys(), key=(lambda k: _pospr_C_X_norm[k]))
    test_max_post_pr = _pospr_C_X_norm[test_predict]

    return test_predict, test_max_post_pr


def testset_predict(instance_data_trn, instance_data_test, var_name_ranges, label_range, var_idxs):

    # get conditional probability for each variable given different class labels
    _cpr_X_C = cpr_X_given_C(instance_data_trn, var_name_ranges, label_range, var_idxs)

    # get probability of each class label
    # note the following function can also be used to compute prior probability as long as the input instanceset is the entire set
    _pripr_C = cpr_Xi_given_Cl(instance_data_trn, label_range, -1)

    # prediction and the max posteriori probability of the entire test data set
    testset_pred = []
    testset_max_pospr = []

    for ins_test in instance_data_test:
        ins_test_predict, ins_test_max_pospr = instance_predict_nb(ins_test, _cpr_X_C, _pripr_C, var_name_ranges, label_range, var_idxs)
        testset_pred.append(ins_test_predict)
        testset_max_pospr.append(ins_test_max_pospr)

    return testset_pred, testset_max_pospr


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


"""
TAN
"""
def range_two_vars(vn1, vn2, var_name_ranges):
    """
    Get the joint range of two variables
    :param vidx1:
    :param vidx2:
    :param var_ranges:
    :return:
    """

    var_range1 = var_name_ranges[vn1]
    var_range2 = var_name_ranges[vn2]

    _range_ = [(vr1, vr2) for vr1 in var_range1 for vr2 in var_range2]

    return _range_


def range_three_vars(vn1, vn2, vn3, var_name_ranges):
    """
    Get the joint range of three variables
    :param vidx1:
    :param vidx2:
    :param vidx3:
    :param var_ranges:
    :return:
    """

    var_range1 = var_name_ranges[vn1]
    var_range2 = var_name_ranges[vn2]
    var_range3 = var_name_ranges[vn3]

    _range_ = [(vr1, vr2, vr3) for vr1 in var_range1 for vr2 in var_range2 for vr3 in var_range3]

    return _range_


def jpr_three_var(vn0, vn1, vn2, instanceset, joint_range_three, var_idxs):
    """
    Joint probability of 3 variables
    :param vidx0:
    :param vidx1:
    :param vidx2:
    :param instance_set:
    :param joint_range:
    :return:
    """

    num_instance = len(instanceset)  # number of instances
    num_jnt_var = len(joint_range_three)  # number of possible value for this joint range

    vidx0 = var_idxs[vn0]
    vidx1 = var_idxs[vn1]
    vidx2 = var_idxs[vn2]
    # joint probability of each possible variable value within the joint range
    _jpr_3v = {}

    for var in joint_range_three:
        # compute the number of instances with corresponding variables == var
        num_instance_var = sum(np.array([ins[vidx0] == var[0] and ins[vidx1] == var[1] and ins[vidx2] == var[2] for ins in instanceset]))
        _jpr_3v[var] = (num_instance_var + 1.0) / (num_instance + num_jnt_var)

    return _jpr_3v


def jpr_two_var(vn0, vn1, instanceset, joint_range_two, var_idxs):
    """
    Joint probability of two variables
    :param vidx0:
    :param vidx1:
    :param instance_set:
    :param joint_range:
    :return:
    """

    num_instance = len(instanceset)  # number of instances
    num_jnt_var = len(joint_range_two)  # number of possible value for this joint range

    vidx0 = var_idxs[vn0]
    vidx1 = var_idxs[vn1]

    # joint probability of each possible variable value within the joint range
    _jpr_2v = {}

    for var in joint_range_two:
        # compute the number of instances with corresponding variables == var
        num_instance_var = sum(np.array([ins[vidx0] == var[0] and ins[vidx1] == var[1] for ins in instanceset]))
        _jpr_2v[var] = (num_instance_var + 1.0) / (num_instance + num_jnt_var)

    return _jpr_2v
