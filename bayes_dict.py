"""
Project: Naive Bayes and TAN

Main file to run Naive Bayes or TAN algorithm

@ Author: Bo Peng
@ University of Wisconsin - Madison
"""


import bayes_func_dict as bf
import scipy.io.arff as af

"""
Step 1: read parameters from command line
"""
#filename_trn, filename_test, n_t = bf.read_cmdln_arg()
filename_trn = 'lymph_train.arff'
filename_test = 'lymph_test.arff'
n_t = 'n'
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
var_name_ranges = {var_names[i]: var_ranges[i] for i in range(len(var_names)-1)}
var_idxs = {var_names[i]: i for i in range(len(var_names)-1)}
label_name_range = {var_names[-1]: var_ranges[-1]}
label_range = var_ranges[-1]

if None in var_ranges:
    print("Error: data contain non-discrete attribute, this program only process discrete data")
    exit(0)

"""
Step 3: Training and Tesing using Naive Bayes or TAN
"""

test_pred, test_postPr = bf.testset_predict(instance_data_trn, instance_data_test, var_name_ranges, label_range, var_idxs)
num_correct_pred = bf.comp_num_correct_predict(test_labels, test_pred)

# print out the results
# the 1st part, node and its parents
for i in range(len(var_names) - 1):
    print('{0} {1}'.format(var_names[i], var_names[-1]))

print('')


#print("Incorrect arguments...")
#exit(0)

# print out the prediction results
# the 2nd part, prediction, label, posteriori probability
for i in range(len(test_pred)):
    print('{0} {1} {2:.12f}'.format(test_pred[i], test_labels[i], test_postPr[i]))
print('')

# the 3rd part, number of correct prediction
print(str(num_correct_pred) + '\n')