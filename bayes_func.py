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

def pr_var_range(instance_set, var_range, var_idx):
    """
    Compute the probability of each possible variable value within the variable range
    :param instance_set:
    :param var_range:
    :param var_idx:
    :return:
    """

    num_instance = len(instance_set) # number of instances
    num_var = len(var_range) # number of possible variable value

    # probability of each possible variable value within the variable range
    _pr_ = []

    for var in var_range:
        # compute the number of instances with the [var_idx]th variable == var
        num_instance_var = sum(np.array([ins[var_idx] == var for ins in instance_set]))
        _pr_.append((num_instance_var + 1.0) / (num_instance + num_var))

    return _pr_


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


def cnd_pr_one_lb(instance_with_lb, var_ranges):
    """
    Compute conditional probability of all variables given instance label == lb
    :param instance_with_lb:
    :param var_ranges:
    :return:
    """

    _cnd_pr_ = []
    for idx in range(len(var_ranges)-1): # note that the last variable is <class>, which should be excluded
        _cnd_pr_.append(pr_var_range(instance_with_lb, var_ranges[idx], idx))

    return _cnd_pr_


def cnd_pr_all_lb(instance_data, var_ranges, label_range):
    """
    Compute conditional probability for each variable in each subset given different lables
    :param instance_data: training instances
    :param var_ranges: [list] range of all variables
    :param label_range: range of all labels
    :return: a list, each element contains conditional probability of each variable given one class label
    """

    instance_split = split_instance_on_label(instance_data, label_range)
    _cnd_pr_ = [cnd_pr_one_lb(ins_splt, var_ranges) for ins_splt in instance_split]

    return _cnd_pr_

def instance_predict(instance_test, cnd_pr_all_lb, pr_lb, var_ranges, label_range):
    """
    Predict the class of the test instance
    :param test_instance: single test instance
    :param cnd_pr_all_lb: [list] conditional probability for each variable given different class lables
    :return: the class of the test instance
    """

    # for current instance, loop over all variables to compute the conditional probability give different class labels
    # number of class labels
    num_label = len(label_range)
    num_var = len(var_ranges) - 1  # number of variables, the last variable is <class> which should be excluded

    # posteriori probability for each label
    post_pr_all_lb = []
    # loop over all subsets
    for i in range(num_label):
        cnd_pr_one_lb = cnd_pr_all_lb[i] # conditional probability of all variables given one label
        cnd_pr_ins_all_var = [] # conditional probability of the instance value for all variables given one label

        # loop over all variables
        for j in range(num_var):
            this_var_range = var_ranges[j] # range of this variables
            _cnd_pr_var_ = cnd_pr_one_lb[j] # conditional probability for all values of this variable
            # get the probability of the instance value for this variable
            _cnd_pr_ins_this_var_ = _cnd_pr_var_[this_var_range.index(instance_test[j])]
            cnd_pr_ins_all_var.append(_cnd_pr_ins_this_var_)

        # compute the posteriori probability of this test instance being classified as the ith label
        # posteriori Pr = P(yi) * [P(a1|yi) * P(a2|yi)*... *P(an|yi)]
        _post_pr_one_lb_ = pr_lb[i] * np.product(np.array(cnd_pr_ins_all_var))
        post_pr_all_lb.append(_post_pr_one_lb_)

    # normalize the posteriori probability
    post_pr_all_lb_norm = [_post_pr_one_lb_ / sum(np.array(post_pr_all_lb)) for _post_pr_one_lb_ in post_pr_all_lb]

    # get the index of the max conditional probability
    idx_max_post_pr = np.argmax(np.array(post_pr_all_lb_norm))
    # get the label corresponding to this idx
    test_predict = label_range[idx_max_post_pr]
    test_max_post_pr = post_pr_all_lb_norm[idx_max_post_pr]

    return test_predict, test_max_post_pr


def testset_predict(instance_data_trn, instance_data_test, var_ranges, label_range):

    # get conditional probability for each variable given different class labels
    _cnd_pr_ = cnd_pr_all_lb(instance_data_trn, var_ranges, label_range)

    # get probability of each class label
    _pr_lb_ = pr_var_range(instance_data_trn, label_range, -1)

    # prediction and the max posteriori probability of the entire test data set
    testset_pred = []
    testset_max_post_pr = []

    for ins_test in instance_data_test:
        ins_test_predict, ins_test_max_postPr = instance_predict(ins_test, _cnd_pr_, _pr_lb_, var_ranges, label_range)
        testset_pred.append(ins_test_predict)
        testset_max_post_pr.append(ins_test_max_postPr)

    return testset_pred, testset_max_post_pr

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

def range_two_vars(vidx1, vidx2, var_ranges):
    """
    Get the joint range of two variables
    :param vidx1:
    :param vidx2:
    :param var_ranges:
    :return:
    """

    var_range1 = var_ranges[vidx1]
    var_range2 = var_ranges[vidx2]

    _range_ = [[vr1, vr2] for vr1 in var_range1 for vr2 in var_range2]

    return _range_

def range_three_vars(vidx1, vidx2, vidx3, var_ranges):
    """
    Get the joint range of three variables
    :param vidx1:
    :param vidx2:
    :param vidx3:
    :param var_ranges:
    :return:
    """

    var_range1 = var_ranges[vidx1]
    var_range2 = var_ranges[vidx2]
    var_range3 = var_ranges[vidx3]

    _range_ = [[vr1, vr2, vr3] for vr1 in var_range1 for vr2 in var_range2 for vr3 in var_range3]

    return _range_


def joint_pr_three_var(vidx0, vidx1, vidx2, instance_set, joint_range):

    num_instance = len(instance_set)  # number of instances
    num_jnt_var = len(joint_range)  # number of possible value for this joint range

    # probability of each possible variable value within the joint range
    _pr_ = []

    for var in joint_range:
        # compute the number of instances with corresponding variables == var
        num_instance_var = sum(np.array([ins[vidx0] == var[0] and ins[vidx1] == var[1] and ins[vidx2] == var[2] for ins in instance_set]))
        _pr_.append((num_instance_var + 1.0) / (num_instance + num_jnt_var))

    return _pr_


def joint_pr_two_var(vidx0, vidx1, instance_set, joint_range):

    num_instance = len(instance_set)  # number of instances
    num_jnt_var = len(joint_range)  # number of possible value for this joint range

    # probability of each possible variable value within the joint range
    _pr_ = []

    for var in joint_range:
        # compute the number of instances with corresponding variables == var
        num_instance_var = sum(np.array([ins[vidx0] == var[0] and ins[vidx1] == var[1] for ins in instance_set]))
        _pr_.append((num_instance_var + 1.0) / (num_instance + num_jnt_var))

    return _pr_


def mutual_info(pr_xi_xj_y, pr_xi_given_y, pr_xj_given_y, pr_xi_xj_given_y):

    _mut_info_ = pr_xi_xj_y * np.log2(pr_xi_xj_given_y / (pr_xi_given_y * pr_xj_given_y))

    return _mut_info_


def edge_weight(instance_set, instance_set_split, _cnd_pr_, vidx0, vidx1, var_ranges, label_range):

    # set the mutual information between two same variables as -1.0
    if vidx0 == vidx1:
        return -1.0

    # weight of the edge between variable 0 and 1
    weight_v0_v1 = 0

    # get the range of (xi, xj, y)
    range_XXY = range_three_vars(vidx0, vidx1, -1, var_ranges)
    # compute the joint probability list of each P(xi, xj, y) corresponding to the joint range
    _jnt_pr_XXY_ = joint_pr_three_var(vidx0, vidx1, -1, instance_set, range_XXY)

    # get the range of (xi, xj)
    range_XX = range_two_vars(vidx0, vidx1, var_ranges)

    num_label = len(label_range)  # number of labels
    for l in range(num_label):
        # number of possible values for the [vidx_i]th variable
        num_var0 = len(var_ranges[vidx0])

        for i in range(num_var0):
            # number of possible values for the [vidx_j]th variable
            num_var1 = len(var_ranges[vidx1])

            for j in range(num_var1):
                # compute P(xi, xj, y), P(xi, xj|y), P(xi|y), P(xj|y)
                # compute the joint probability for current (xi, xj, y), P(xi, xj, y)
                _idx_ijy = range_XXY.index([var_ranges[vidx0][i], var_ranges[vidx1][j], var_ranges[-1][l]]) # index of current (xi, xj, y)
                jnt_pr_ijy = _jnt_pr_XXY_[_idx_ijy]

                # compute the conditional joint probability list of each (xi,xj|y) given the label y, P(xi, xj|y)
                _cnd_jnt_pr_XX_ = joint_pr_two_var(vidx0, vidx1, instance_set_split[l], range_XX)
                # comupte the conditional joint probability of current (xi,xj|y), P(xi, xj|y)
                _idx_ij = range_XX.index([var_ranges[vidx0][i], var_ranges[vidx1][j]])  # index of current (xi, xj|y)
                cnd_jnt_pr_ij = _cnd_jnt_pr_XX_[_idx_ij]

                # Compute the conditional probability of xi given y, P(xi|y)
                cnd_pr_xi = _cnd_pr_[l][vidx0][i]
                cnd_pr_xj = _cnd_pr_[l][vidx1][j]

                # compute mutual information
                weight_v0_v1 += mutual_info(jnt_pr_ijy, cnd_pr_xi, cnd_pr_xj, cnd_jnt_pr_ij)

    return weight_v0_v1


def edge_weight_graph(instance_set, var_ranges, label_range):
    """
    Compute mutual information / edge weight graph between each pair of variables
    :param instance_set:
    :param var_ranges:
    :param label_range:
    :return:
    """

    instance_set_split = split_instance_on_label(instance_set, label_range)
    _cnd_pr_ = [cnd_pr_one_lb(ins_splt, var_ranges) for ins_splt in instance_set_split]

    edge_weight_graph = []

    for i in range(len(var_ranges)-1):
        _edge_weight_i_ = []
        for j in range(len(var_ranges)-1):
            _edge_weight_i_.append(edge_weight(instance_set, instance_set_split, _cnd_pr_, i, j, var_ranges, label_range))
        edge_weight_graph.append(np.array(_edge_weight_i_))

    return edge_weight_graph


def prim_mst(edge_weight_graph):
    """
    fing the Maximum Spanning Tree via Prim's algorithm
    :param edge_weight_graph:
    :param var_ranges:
    :return:
    """

    # number of variables
    num_var = len(edge_weight_graph)

    # set original vertex list
    V_origin = [tn.TanNode(i) for i in range(num_var)]

    # initialize the new vertex list
    V_new = []
    v_root = V_origin[0]
    v_root.is_new = True
    v_root.is_root = True  # the 1st variable is set to be the root
    V_new.append(v_root)  # add root node to the V_new
    del V_origin[0]  # delete root node from V_origin


    # select nodes from V_new and V_origin separately to get the maximum edge

    while len(V_origin) > 0:

        idx_max_v_origin = []
        weight_max = []
        for vn in V_new:
            idx_vn = vn.var_idx

            # find node in V_origin which produecs maximum weight edge, i.e., v
            _idx_ = np.argmax(edge_weight_graph[idx_vn])
            idx_max_v_origin.append(_idx_)
            weight_max.append(edge_weight_graph[idx_vn][_idx_])

        # find node in V_new which produecs maximum weight edge, i.e., u
        idx_u = np.argmax(np.array(weight_max))

        # extract the corresponding node v in V_origin
        # note that the index found here is not the index in V_origin because the size of V_origin is decreasing
        # the index here is the index of the node's original index in "var_ranges"
        var_idx_v =  idx_max_v_origin[idx_u]

        # find node in V_origin whose var_idx equals var_idx_v
        var_idx_origin = [node.var_idx for node in V_origin]
        idx_v = var_idx_origin.index(var_idx_v)  # index of node v in "V_origin"

        # add v to V_new
        vo = V_origin[idx_v]
        vo.is_new = True
        vo.parents.append(V_new[idx_u].var_idx)
        V_new[idx_u].children.append(vo.var_idx)
        V_new.append(vo)
        del V_origin[idx_v]

        # set the edge weight between nodes in V_new to be -1.0
        for v1 in V_new:
            var_idx1 = v1.var_idx
            for v2 in V_new:
                var_idx2 = v2.var_idx
                edge_weight_graph[var_idx1][var_idx2] = -1.0

    return V_new