"""
Project: Naive Bayes and TAN

Main file to run Naive Bayes or TAN algorithm

@ Author: Bo Peng
@ University of Wisconsin - Madison
"""


import bayes_func_v_error as bf
import scipy.io.arff as af

"""
Step 1: read parameters from command line
"""
#filename_trn, filename_test, n_t = bf.read_cmdln_arg()
filename_trn = 'lymph_train.arff'
filename_test = 'lymph_test.arff'
n_t = 't'
"""
Step 2: load training and testing data
"""
# load data
instance_data_trn, meta_data = af.loadarff(filename_trn)
instance_data_test, meta_data = af.loadarff(filename_test)
test_labels = [ins[-1] for ins in instance_data_test]

# relative parameters
var_ranges = [meta_data[name][1] for name in meta_data.names()]
var_names = meta_data.names()
label_range = var_ranges[-1]

if None in var_ranges:
    print("Error: data contain non-discrete attribute, this program only process discrete data")
    exit(0)

"""
Step 3: Training and Tesing using Naive Bayes or TAN
"""
if n_t == 't':  # TAN
    # generate the edge weight graph
    weight_graph = bf.edge_weight_graph(instance_data_trn, var_ranges, label_range)
    # compute the new vertex list
    V_new = bf.prim_mst(weight_graph)
    # test set prediction
    test_pred, test_postPr = bf.testset_prediction_tan(instance_data_trn, instance_data_test, var_ranges, label_range, V_new)
    # number of correct prediction
    num_correct_pred = bf.comp_num_correct_predict(test_labels, test_pred)

    # print out the TAN
    # the 1st part, node, parents
    var_idx_self = [node.var_idx for node in V_new]
    for i in range(len(var_names) - 1):  # the last variable in var_names is <class> should be excluded
        if i == 0:  # the 1st node/variable is root, only one parent <class>
            print(var_names[0] + ' ' + var_names[-1])
        else:
            var_name_self = var_names[i]
            idx_self = var_idx_self.index(i)  # index of the ith variable in V_new
            var_idx_parent = V_new[idx_self].parents[1]  # variable index of parent
            var_name_parent = var_names[var_idx_parent]
            print(var_name_self + ' ' + var_name_parent + ' ' + var_names[-1])

    print('')

elif n_t == 'n':  # Naive Bayes
    test_pred, test_postPr = bf.testset_predict(instance_data_trn, instance_data_test, var_ranges, label_range)
    num_correct_pred = bf.comp_num_correct_predict(test_labels, test_pred)

    # print out the results
    # the 1st part, node and its parents
    for i in range(len(var_names) - 1):
        print('{0} {1}'.format(var_names[i], var_names[-1]))

    print('')

else:
    print("Incorrect arguments...")
    exit(0)

# print out the prediction results
# the 2nd part, prediction, label, posteriori probability
for i in range(len(test_pred)):
    print('{0} {1} {2:.12f}'.format(test_pred[i], test_labels[i], test_postPr[i]))
print('')

# the 3rd part, number of correct prediction
print(str(num_correct_pred) + '\n')