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


"""
Naive Bayes
"""

def cpr_Xi_given_Ci(instanceset_Ci, var_range, var_idx):
    """
    P(Xi|Ci): Compute the probability of each possible variable value Xi given the class label Ci
    Note the following function can also be used to compute prior probability as long as the input instanceset is the entire set
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

def instance_predict_nb(instance, _cpr_X_C, _pripr_C, var_ranges, label_range):
    """
    Predict the class of the test instance
    :param instance: single test instance
    :param _cpr_X_C: [list] conditional probability for each variable given different class lables
    :param _pripr_C: prior probability of each class labels
    :return: the class of the test instance
    """

    # for current instance, loop over all variables to compute the conditional probability give different class labels
    # number of class labels
    num_class = len(label_range)
    num_var = len(var_ranges) - 1  # number of variables, the last variable is <class> which should be excluded

    # P(C|X): posteriori probability for each label
    _pospr_C_X = []
    # loop over all subsets
    for i in range(num_class):
        _cpr_X_Ci = _cpr_X_C[i] # P(X|Ci): conditional probability of all variables given one class label, Ci
        _ins_cpr_X_Ci = [] # [P(Xi|Ci)]: [list], conditional probability of the instance value of each variable given one class label Ci

        # loop over all variables for this instance
        for j in range(num_var):
            this_var_range = var_ranges[j] # range of this variables
            _cpr_Xj_Ci = _cpr_X_Ci[j] # conditional probability for all values of this variable
            # get the probability of the instance value for this variable
            _ins_cpr_Xj_Ci = _cpr_Xj_Ci[this_var_range.index(instance[j])]
            _ins_cpr_X_Ci.append(_ins_cpr_Xj_Ci)

        # compute the posteriori probability of this test instance being classified as the ith label
        # posteriori Pr: P(Ci|X) = P(Ci) * (P(X1|Ci) * P(X2|Ci)*... *P(Xn|Ci))
        _pospr_Ci_X = _pripr_C[i] * np.product(np.array(_ins_cpr_X_Ci))
        _pospr_C_X.append(_pospr_Ci_X)

    # normalize the posteriori probability
    _pospr_C_X_norm = [_pospr_Ci_X / sum(np.array(_pospr_C_X)) for _pospr_Ci_X in _pospr_C_X]

    # get the index of the max conditional probability
    idx_max_pospr = np.argmax(np.array(_pospr_C_X_norm))
    # get the label corresponding to this idx
    test_predict = label_range[idx_max_pospr]
    test_max_post_pr = _pospr_C_X_norm[idx_max_pospr]

    return test_predict, test_max_post_pr


def testset_predict_nb(instanceset_trn, instanceset_test, var_ranges, label_range):

    # get conditional probability for each variable given different class labels
    _cpr_X_C = cpr_X_given_C(instanceset_trn, var_ranges, label_range)

    # get probability of each class label
    # note the following function can also be used to compute prior probability as long as the input instanceset is the entire set
    _pripr_C = cpr_Xi_given_Ci(instanceset_trn, label_range, -1)

    # prediction and the max posteriori probability of the entire test data set
    testset_pred = []
    testset_max_pospr = []

    for ins_test in instanceset_test:
        ins_test_predict, ins_test_max_pospr = instance_predict_nb(ins_test, _cpr_X_C, _pripr_C, var_ranges, label_range)
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


def jpr_three_var(vidx0, vidx1, vidx2, instanceset, joint_range_three):
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

    # joint probability of each possible variable value within the joint range
    _jpr_3v = []

    for var in joint_range_three:
        # compute the number of instances with corresponding variables == var
        num_instance_var = sum(np.array([ins[vidx0] == var[0] and ins[vidx1] == var[1] and ins[vidx2] == var[2] for ins in instanceset]))
        _jpr_3v.append((num_instance_var + 1.0) / (num_instance + num_jnt_var))

    return _jpr_3v


def jpr_two_var(vidx0, vidx1, instanceset, joint_range_two):
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

    # joint probability of each possible variable value within the joint range
    _jpr_2v = []

    for var in joint_range_two:
        # compute the number of instances with corresponding variables == var
        num_instance_var = sum(np.array([ins[vidx0] == var[0] and ins[vidx1] == var[1] for ins in instanceset]))
        _jpr_2v.append((num_instance_var + 1.0) / (num_instance + num_jnt_var))

    return _jpr_2v


def mutual_info(_jpr_X0iX1jCl, _cpr_X0i_Cl, _cpr_X1j_Cl, _cjpr_X0iX1j_Cl):
    """
    Compute mutual information between two nodes/variables
    :param _jpr_X0iX1jCl: joint probability of (X0i, X1j, Cl)
    :param _cpr_X0i_Cl: conditional probability of X0i given Cl, where 0 mean one variable, i means the ith value of this variable range
    :param _cpr_X1j_Cl: conditional probability of X1j given Cl, where 1 mean the other variable, j means the jth value of this variable range
    :param _cjpr_X0iX1j_Cl: conditional joint probability of (X0i, X1j) given Cl
    :return:
    """

    _mut_info_ = _jpr_X0iX1jCl * np.log2(_cjpr_X0iX1j_Cl / (_cpr_X0i_Cl * _cpr_X1j_Cl))

    return _mut_info_


def edge_weight(instanceset, instanceset_split, _cpr_X_C, vidx0, vidx1, var_ranges, label_range):

    # set the mutual information between two same variables as -1.0
    if vidx0 == vidx1:
        return -1.0

    # weight of the edge between variable 0 and 1
    weight_v0_v1 = 0

    # get the range of (X0, X1, C),
    # note 0, 1 refer to two variables in var_ranges, not necessarily the 1st and 2nd variables
    range_X0X1C = range_three_vars(vidx0, vidx1, -1, var_ranges)
    # compute the joint probability list of each P(xi, xj, y) corresponding to the joint range
    _jpr_X0X1C = jpr_three_var(vidx0, vidx1, -1, instanceset, range_X0X1C)

    # get the range of (X0, X1)
    range_X0X1 = range_two_vars(vidx0, vidx1, var_ranges)

    num_class = len(label_range)  # number of class labels
    for l in range(num_class):
        # number of possible values for the [vidx_i]th variable
        num_var0 = len(var_ranges[vidx0])

        for i in range(num_var0):
            # number of possible values for the [vidx_j]th variable
            num_var1 = len(var_ranges[vidx1])

            for j in range(num_var1):
                # compute P(X0i, X1j, Cl), P(X0i, X1j|Cl), P(X0i|Cl), P(X1j|Cl)
                # compute the joint probability for current (X0i, X1j, Cl), P(X0i, X1j, Cl)
                # index of current (X0i, X1j, Cl) in the joint range
                _idx_X0iX1jCl = range_X0X1C.index([var_ranges[vidx0][i], var_ranges[vidx1][j], var_ranges[-1][l]])
                _jpr_X0iX1jCl = _jpr_X0X1C[_idx_X0iX1jCl]

                # conditional joint probability of all possible values in joint range: P(X0,X1|Cl), a list including cpr for each joint value
                _cjpr_X0X1_Cl = jpr_two_var(vidx0, vidx1, instanceset_split[l], range_X0X1)

                # index of current (X0i,X1j)
                _idx_X0iX1j = range_X0X1.index([var_ranges[vidx0][i], var_ranges[vidx1][j]])
                # conditional joint probability of current value (X0i,X1j|Cl), P(X0i,X1j|Cl)
                _cjpr_X0iX1j_Cl = _cjpr_X0X1_Cl[_idx_X0iX1j]

                # Compute the conditional probability of X0i and X1j given the class label Cl, P(X0i|Cl), P(X1j|Cl)
                _cpr_X0i_Cl = _cpr_X_C[l][vidx0][i]
                _cpr_X1j_Cl = _cpr_X_C[l][vidx1][j]

                # compute mutual information
                weight_v0_v1 += mutual_info(_jpr_X0iX1jCl, _cpr_X0i_Cl, _cpr_X1j_Cl, _cjpr_X0iX1j_Cl)

    return weight_v0_v1


def edge_weight_graph(instanceset, var_ranges, label_range):
    """
    Compute mutual information / edge weight graph between each pair of variables
    :param instance_set:
    :param var_ranges:
    :param label_range:
    :return:
    """

    instanceset_split = split_instanceset_on_label(instanceset, label_range)
    _cpr_X_C = [cpr_X_given_Ci(isp, var_ranges) for isp in instanceset_split]


    edge_weight_graph = []

    for i in range(len(var_ranges)-1):
        _edge_weight_i_ = []
        for j in range(0, i):
            _edge_weight_i_.append(edge_weight_graph[j][i])  # the edge weight graph is symmetric [i,j] = [j,i]
        for j in range(i, len(var_ranges)-1):
            _edge_weight_i_.append(edge_weight(instanceset, instanceset_split, _cpr_X_C, i, j, var_ranges, label_range))
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


def cpr_Xroot_Ci(var_ranges, label_range, instance_set_split):
    """
    P(Xroot|Ci)Compute the conditional probability of root given each label
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


def cpr_Xi_given_Cl_Xparent(V_new_node, instanceset_split_Cl, instance_test, var_ranges):
    """
    P(Xi | Cl, Xparent): conditional probability of Xi given <class> and <parent>
    Xi refers to the input node: V_new_node
    :param V_new_node:
    :param instance_lb_split:
    :param instance_test:
    :param var_ranges:
    :return:
    """

    # get the Xparent of this node
    Xparent_var_idx = V_new_node.parents[1]
    parent_value = instance_test[Xparent_var_idx]
    # number of instances = (C, Xparent)
    num_CiXparent = sum(np.array([parent_value==ins[Xparent_var_idx] for ins in instanceset_split_Cl]))

    Xi_var_idx = V_new_node.var_idx
    Xi_val = instance_test[Xi_var_idx]
    # number of instances = (Xi, C, Xparent)
    num_XiClXparent = sum(np.array([parent_value==ins[Xparent_var_idx] and Xi_val==ins[Xi_var_idx] for ins in instanceset_split_Cl]))

    num_Xi_var_range = len(var_ranges[Xi_var_idx])  # number of possible values for current node
    _cpr_Xi_ClXparent = (num_XiClXparent + 1.0) / (num_CiXparent + num_Xi_var_range)  # laplace estimate: # P(Xi | Cl, Xparent)

    return _cpr_Xi_ClXparent


def cpr_X_given_C_Xparent(instanceset, V_new, var_ranges):

    _cpr_X_CXp = []  # conditional joint probability P(X|C, Xparent)
    # loop over V_new
    for i in range(1, len(V_new)):
        Xi_var_idx = V_new[i].var_idx  # variable index of Xi
        Xp_var_idx = V_new[i].parents[1]  # variable index of the parent of Xi
        range_Xi = var_ranges[Xi_var_idx]

        # joint range of three variables [Xi, class, Xparent]
        range_Xi_C_Xp = range_three_vars(Xi_var_idx, -1, Xp_var_idx, var_ranges)

        # joint range of [class, Xparent]
        range_C_Xp = range_two_vars(-1, Xp_var_idx, var_ranges)

        _cjpr_Xi_CXp = []  # conditional joint probability  P(Xi|C,Xparent)
        for rcxp in range_C_Xp:
            # number of instances corresponding to (C, Xparent)
            _num_ins_CXparent = sum(np.array([ins[-1] == rcxp[0] and ins[Xp_var_idx] == rcxp[1] for ins in instanceset]))

            _cjpr_Xi_CXpj = []  # P(Xi|(C,Xp)j)
            for rxi in range_Xi:
                # number of instances corresponding to (Xi, C, Xparent)
                _num_ins_Xi_C_Xp = sum(np.array([ins[Xi_var_idx] == rxi and ins[-1] == rcxp[0] and ins[Xp_var_idx] == rcxp[1] for ins in instanceset]))
                _cjpr_Xi_CXpj.append((_num_ins_Xi_C_Xp + 1.0) / (_num_ins_CXparent + len(range_Xi)))

            _cjpr_Xi_CXp.append(_cjpr_Xi_CXpj)

        _cpr_X_CXp.append(_cjpr_Xi_CXp)  # add each node Xi

    return _cpr_X_CXp


def ins_cpr_Xi_given_Cl_Xparent(_cpr_X_CXp, V_new, vn_idx, C_idx, instance_test, var_ranges, label_range):

    i = C_idx  # index of class/label in label_range
    j = vn_idx  # index of node in V_new

    Xj_var_idx = V_new[j].var_idx
    Xp_var_idx = V_new[j].parents[1]

    range_CXp = range_two_vars(-1, Xp_var_idx, var_ranges)  # joint range of [class, Xparent]
    range_Xj = var_ranges[Xj_var_idx]  # node self range

    Xj_val = instance_test[Xj_var_idx]  # node self value
    Ci_val = label_range[i]  # possible class value
    Xp_val = instance_test[Xp_var_idx]  # parent value

    idx_CiXp = range_CXp.index([Ci_val, Xp_val])  # index within joint range [class, Xparent]
    idx_Xj = range_Xj.index(Xj_val)  # index within self range

    # IMPORTANT: _cpr_X_CXp does not include the root node, so the length of _cpr_X_CXp is len(V_new) - 1
    # in that case, the index of jth node in _cpr_X_CXp should be [j-1]
    _cpr_Xj_CiXp = _cpr_X_CXp[j - 1][idx_CiXp][idx_Xj]

    return _cpr_Xj_CiXp


def instance_pred_tan(instance_test, _prior_pr_C, _cpr_X_C, _cpr_X_CXp, var_ranges, label_range, V_new):
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
    _post_pr_C = []
    for i in range(len(label_range)):
        pri_pr_Ci = _prior_pr_C[i]  # prior probability P(C)
        cpr_Xroot_Ci = _cpr_X_C[i][0][idx_rv]  # P(Xroot | C)

        # for each node/variable, compute the conditional probabilty of Xi given root and class/label
        # the 1st node is root, no need to compute, start with the 2nd
        _cpr_X_CiXp = []  # P(Xi|Ci,Xp)
        for j in range(1, len(V_new)):
            # compute the conditional probability of Xj given its parents, P(Xi | C, Xparent)
            _cpr_Xj_CiXp_ = ins_cpr_Xi_given_Cl_Xparent(_cpr_X_CXp, V_new, j, i, instance_test, var_ranges, label_range)
            _cpr_X_CiXp.append(_cpr_Xj_CiXp_)

        # compute the posterori probability of C given Xi
        _post_pr_Ci_ = pri_pr_Ci * cpr_Xroot_Ci * np.product(np.array(_cpr_X_CiXp))  # post probability for current class/label
        _post_pr_C.append(_post_pr_Ci_)

    # posterori probability normalization
    _post_pr_C_norm = [_ppc_ / sum(np.array(_post_pr_C)) for _ppc_ in _post_pr_C]
    test_pred_idx = np.argmax(np.array(_post_pr_C_norm))
    test_pred = label_range[test_pred_idx]
    test_max_post_pr = _post_pr_C_norm[test_pred_idx]

    return test_pred, test_max_post_pr


def testset_prediction_tan(instanceset_trn, instanceset_test, var_ranges, label_range):
    """
    Prediction of a instance set with multiple instances
    :param instance_data_trn:
    :param instance_data_test:
    :param var_ranges:
    :param label_range:
    :param V_new:
    :return:
    """

    # generate the edge weight graph
    weight_graph = edge_weight_graph(instanceset_trn, var_ranges, label_range)

    # compute the new vertex list
    V_new = prim_mst(weight_graph)

    # get conditional probability for each variable given different class labels
    _cpr_X_C = cpr_X_given_C(instanceset_trn, var_ranges, label_range)

    # get probability of each class label
    # Note the this function can also be used to compute prior probability as long as the input instanceset is the entire set
    _pri_pr_C = cpr_Xi_given_Ci(instanceset_trn, label_range, -1)

    # P(Xi | C, Xparent)
    _cpr_X_CXp = cpr_X_given_C_Xparent(instanceset_trn, V_new, var_ranges)

    # prediction and the max posteriori probability of the entire test data set
    testset_pred = []
    testset_max_post_pr = []

    for ins_test in instanceset_test:
        ins_test_predict, ins_test_max_postPr = instance_pred_tan(ins_test, _pri_pr_C, _cpr_X_C, _cpr_X_CXp, var_ranges, label_range, V_new)

        testset_pred.append(ins_test_predict)
        testset_max_post_pr.append(ins_test_max_postPr)

    return testset_pred, testset_max_post_pr, V_new