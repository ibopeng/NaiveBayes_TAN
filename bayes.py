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

instance_data_test, meta_data = af.loadarff('lymph_test.arff')
test_labels = [ins[-1] for ins in instance_data_test]

test_pred, test_postPr = bf.testset_predict(instance_data_trn, instance_data_test, var_ranges, label_range)

num_correct_pred = bf.comp_num_correct_predict(test_labels, test_pred)

# print out the results

# the 1st part, node and its parents
var_names = meta_data.names()
for i in range(len(var_names)-1):
    print('{0} {1}'.format(var_names[i], var_names[-1]))

print('')

# the 2nd part, prediction, label, posteriori probability
for i in range(len(test_pred)):
    print('{0} {1} {2:.12f}'.format(test_pred[i], instance_data_test[i][-1], test_postPr[i]))

print('')

# the 3rd part, number of correct prediction
print(str(num_correct_pred) + '\n')

