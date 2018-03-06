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


def cpr_Xi_given_Ci(instanceset_Ci, var_range, var_idx):
    """
    P(Xi|Ci): Compute the probability of each possible variable value Xi given the class label Ci
    :param instance_set:
    :param var_range:
    :param var_idx:
    :return:
    """

    num_instance_Ci = len(instanceset_Ci) # number of instances with label == Ci
    num_var = len(var_range) # number of possible variable value

    # probability of each possible variable value within the variable range
    _cpr_Xi_Ci = []

    for var in var_range:
        # compute the number of instances with the [var_idx]th variable == var
        num_instance_var = sum(np.array([ins[var_idx] == var for ins in instanceset_Ci]))
        _cpr_Xi_Ci.append((num_instance_var + 1.0) / (num_instance_Ci + num_var))

    return _cpr_Xi_Ci


def split_instanceset_on_label(instance_data, label_range):
    """
    Split original dataset based on different class labels
    :param instance_data:
    :param label_range:
    :return:
    """

    instanceset_split = []
    for lb in label_range:
        # extract instances with class value == lb
        instance_with_lb = []
        for ins in instance_data:
            if ins[-1] == lb:
                instance_with_lb.append(ins)

        instanceset_split.append(instance_with_lb)

    return instanceset_split


def cpr_X_given_Ci(instanceset_lb, var_ranges):
    """
    P(X|Ci): Compute conditional probability of all variables given instance label == Ci
    :param instance_with_lb:
    :param var_ranges:
    :return:
    """

    _cpr_X_Ci = []
    for idx in range(len(var_ranges)-1): # note that the last variable is <class>, which should be excluded
        _cpr_X_Ci.append(cpr_Xi_given_Ci(instanceset_lb, var_ranges[idx], idx))

    return _cpr_X_Ci


def cpr_X_given_C(instanceset, var_ranges, label_range):
    """
    P(X|C): Compute conditional probability for each variable X given different class labels C
    :param instance_data: training instances
    :param var_ranges: [list] range of all variables
    :param label_range: range of all labels
    :return: a list, each element contains conditional probability of each variable given one class label
    """

    instanceset_split = split_instanceset_on_label(instanceset, label_range)
    _cpr_X_C = [cpr_X_given_Ci(isp, var_ranges) for isp in instanceset_split]

    return _cpr_X_C

def instance_predict_nb(instance_test, cnd_pr_all_lb, pr_lb, var_ranges, label_range):
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
    _cnd_pr_ = cpr_X_given_C(instance_data_trn, var_ranges, label_range)

    # get probability of each class label
    _pr_lb_ = cpr_Xi_given_Ci(instance_data_trn, label_range, -1)

    # prediction and the max posteriori probability of the entire test data set
    testset_pred = []
    testset_max_post_pr = []

    for ins_test in instance_data_test:
        ins_test_predict, ins_test_max_postPr = instance_predict_nb(ins_test, _cnd_pr_, _pr_lb_, var_ranges, label_range)
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
    """
    Joint probability of 3 variables
    :param vidx0:
    :param vidx1:
    :param vidx2:
    :param instance_set:
    :param joint_range:
    :return:
    """

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
    """
    Joint probability of two variables
    :param vidx0:
    :param vidx1:
    :param instance_set:
    :param joint_range:
    :return:
    """

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
    """
    Compute mutual information between two nodes/variables
    :param pr_xi_xj_y:
    :param pr_xi_given_y:
    :param pr_xj_given_y:
    :param pr_xi_xj_given_y:
    :return:
    """

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

    instance_set_split = split_instanceset_on_label(instance_set, label_range)
    _cnd_pr_ = [cpr_X_given_Ci(isp, var_ranges) for isp in instance_set_split]

    edge_weight_graph = []

    for i in range(len(var_ranges)-1):
        _edge_weight_i_ = []
        for j in range(0, i):
            _edge_weight_i_.append(edge_weight_graph[j][i])  # the edge weight graph is symmetric [i,j] = [j,i]
        for j in range(i, len(var_ranges)-1):
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

def cnd_pr_root(var_ranges, label_range, instance_set_split):
    """
    Compute the conditional probability of root given each label
    :param var_ranges:
    :param label_range:
    :param instance_set_split:
    :return:
    """

    root_val = var_ranges[0]  # the range of the root node: 1st variable
    # for each label/class, compute the conditional root probability
    _cpr_lb_ = []  # conditional probability of root for each class/label
    for i in range(len(label_range)):
        _cpr_ = []  # conditional probability of each root value
        for rv in root_val:
            num_rv = sum(np.array([rv==ins[0] for ins in instance_set_split[i]]))  # number of instances with root value = rv
            _cpr_.append(1.0*num_rv / len(instance_set_split[i]))

        _cpr_lb_.append(_cpr_)

    return _cpr_lb_


def cnd_pr_lb_parent(V_new_node, instance_lb_split, instance_test, var_ranges):
    """
    Compute the conditional probability given <class> and <parent>, P(Xi | C, Xparent)
    :param V_new_node:
    :param instance_lb_split:
    :param instance_test:
    :param var_ranges:
    :return:
    """

    # get the parent of this node
    parent_var_idx = V_new_node.parents[1]
    parent_value = instance_test[parent_var_idx]
    # number of instances = (C, Xparent)
    num_pv = sum(np.array([parent_value==ins[parent_var_idx] for ins in instance_lb_split]))

    self_var_idx = V_new_node.var_idx
    self_val = instance_test[self_var_idx]
    # number of instances = (Xi, C, Xparent)
    num_pv_sv = sum(np.array([parent_value==ins[parent_var_idx] and self_val==ins[self_var_idx] for ins in instance_lb_split]))

    num_self_var_range = len(var_ranges[self_var_idx])  # number of possible values for current node
    _cnd_pr_ = (num_pv_sv + 1.0) / (num_pv + num_self_var_range)  # laplace estimate: # P(Xi | C, Xparent)

    return _cnd_pr_


def instance_pred_tan(instance_test, _prior_pr_lb_, _cnd_pr_all_lb_, var_ranges, label_range, V_new, instance_set_split):
    """
    prediction of a single instance
    :param instance_test:
    :param _prior_pr_lb_:
    :param _cnd_pr_all_lb_:
    :param var_ranges:
    :param label_range:
    :param V_new:
    :param instance_set_split:
    :return:
    """

    # root value index
    root_value = instance_test[0]  # the 1st node/variable is root
    idx_rv = var_ranges[0].index(root_value)

    # for each class/label, compute the posteriori probability
    _post_pr_lb_ = []
    for i in range(len(label_range)):
        pri_pr_i = _prior_pr_lb_[i]  # P(C)
        cpr_root = _cnd_pr_all_lb_[i][0][idx_rv]  # P(Xroot | C)

        # for each node/variable, compute the conditional probabilty of Xi given root and class/label
        # the 1st node is root, no need to compute, start with the 2nd
        _cnd_pr_xi_given_parents_ = []
        for j in range(1, len(V_new)):
            # compute the conditional probability of Xj given its parents, P(Xi | C, Xparent)
            _cnd_pr_xi_given_parents_.append(cnd_pr_lb_parent(V_new[j], instance_set_split[i], instance_test, var_ranges))

        # compute the posterori probability of C given Xi
        _post_pr_lb_i_ = pri_pr_i * cpr_root * np.product(np.array(_cnd_pr_xi_given_parents_))  # post probability for current class/label
        _post_pr_lb_.append(_post_pr_lb_i_)

    # posterori probability normalization
    _post_pr_lb_norm_ = [_pr_ / sum(np.array(_post_pr_lb_)) for _pr_ in _post_pr_lb_]
    test_pred_idx = np.argmax(np.array(_post_pr_lb_norm_))
    test_pred = label_range[test_pred_idx]
    test_max_post_pr = _post_pr_lb_norm_[test_pred_idx]

    return test_pred, test_max_post_pr


def testset_prediction_tan(instance_data_trn, instance_data_test, var_ranges, label_range, V_new):
    """
    Prediction of a instance set with multiple instances
    :param instance_data_trn:
    :param instance_data_test:
    :param var_ranges:
    :param label_range:
    :param V_new:
    :return:
    """

    # get conditional probability for each variable given different class labels
    _cnd_pr_ = cpr_X_given_C(instance_data_trn, var_ranges, label_range)

    # get probability of each class label
    _pr_lb_ = cpr_Xi_given_Ci(instance_data_trn, label_range, -1)

    instance_set_split = split_instanceset_on_label(instance_data_trn, label_range)

    # prediction and the max posteriori probability of the entire test data set
    testset_pred = []
    testset_max_post_pr = []

    for ins_test in instance_data_test:
        ins_test_predict, ins_test_max_postPr = instance_pred_tan(ins_test, _pr_lb_, _cnd_pr_, var_ranges, label_range, V_new, instance_set_split)

        testset_pred.append(ins_test_predict)
        testset_max_post_pr.append(ins_test_max_postPr)

    return testset_pred, testset_max_post_pr