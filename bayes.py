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
var_names = meta_data.names()
label_range = var_ranges[-1]

"""
TAN
"""
weight_graph = bf.edge_weight_graph(instance_data_trn, var_ranges, label_range)
V_new = bf.prim_mst(weight_graph)

#print out the TAN

var_idx_self = [node.var_idx for node in V_new]
for i in range(len(var_names)-1): # the last variable in var_names is <class> should be excluded
    if i == 0: # the 1st node/variable is root, only one parent <class>
        print(var_names[0] + ' ' + var_names[-1])
    else:
        var_name_self = var_names[i]
        idx_self = var_idx_self.index(i) # index of the ith variable in V_new
        var_idx_parent = V_new[idx_self].parents[1] # variable index of parent
        var_name_parent = var_names[var_idx_parent]
        print(var_name_self + ' ' + var_name_parent + ' ' + var_names[-1])


"""
NB
"""
instance_data_test, meta_data = af.loadarff('lymph_test.arff')
test_labels = [ins[-1] for ins in instance_data_test]

test_pred, test_postPr = bf.testset_predict(instance_data_trn, instance_data_test, var_ranges, label_range)

num_correct_pred = bf.comp_num_correct_predict(test_labels, test_pred)

# print out the results

# the 1st part, node and its parents

for i in range(len(var_names)-1):
    print('{0} {1}'.format(var_names[i], var_names[-1]))

print('')

# the 2nd part, prediction, label, posteriori probability
for i in range(len(test_pred)):
    print('{0} {1} {2:.12f}'.format(test_pred[i], instance_data_test[i][-1], test_postPr[i]))

print('')

# the 3rd part, number of correct prediction
print(str(num_correct_pred) + '\n')

