"""
Project: Naive Bayes and TAN

Plot precision recall curve

@ Author: Bo Peng
@ University of Wisconsin - Madison
"""

import bayes_func as bf
import scipy.io.arff as af
import numpy as np
import matplotlib.pyplot as plt

# precision_recall curve
def PRC_ROC(test_confidence, test_labels, label_range):
    """
    Compute Precision_Recall Curve and ROC
    :param test_confidence:
    :param test_labels:
    :param label_range:
    :return:
    """

    test_confidence = np.array(test_confidence)

    # compute actual number of positive and negative instances
    num_instance = len(test_confidence)
    num_true_pos = sum(np.array([label_range[0] == lb for lb in test_labels]))
    num_true_neg = num_instance - num_true_pos

    # sort the confidence and label pair
    zipped = zip(test_confidence, test_labels)
    zipped.sort(key = lambda t: t[0]) # sort confidence and label based on confidence, ascending order
    zipped.reverse() # sort the confidence from high to low, descending order
    [test_confidence, test_labels] = zip(*zipped)

    # compute cutoff
    cutoff = []
#    cutoff.append(1)
    for i in range(num_instance):
        if i == 0:
            cutoff.append(test_confidence[0])
            current_state = test_labels[0]
        else:
            if current_state == test_labels[i]:
                continue
            else:
                current_state = test_labels[i]
                cutoff.append(test_confidence[i-1])
                cutoff.append(test_confidence[i])
    cutoff.append(0)

    # compute TP at each cutoff
    PRC_array = []  # recall, precision
    ROC_array = []  # ROC False positive, True positive
    for cf in cutoff:
        TP = 0
        FP = 0
        for i in range(num_instance):
            if test_confidence[i] < cf:
                break
            else:
                if test_labels[i] == label_range[0]:
                    TP += 1
                elif test_labels[i] != label_range[0]:
                    FP += 1
        TP = TP * 1.0
        FP = FP * 1.0
        PRC_array.append([TP/num_true_pos, TP/(TP+FP)]) #(recall, precision)
        ROC_array.append([FP/num_true_neg, TP/num_true_pos])  #(false positive, true positive)

    return PRC_array, ROC_array


"""Load data"""

# load data
instance_data_trn, meta_data = af.loadarff('lymph_train.arff')
instance_data_test, meta_data = af.loadarff('lymph_test.arff')
test_labels = [ins[-1] for ins in instance_data_test]

# relative parameters
var_ranges = [meta_data[name][1] for name in meta_data.names()]
var_names = meta_data.names()
var_name_ranges = {var_names[i]: var_ranges[i] for i in range(len(var_names))}

label_range = var_ranges[-1]

"""TAN prediction"""
weight_graph = bf.edge_weight_graph(instance_data_trn, var_ranges, label_range)
# compute the new vertex list
V_new = bf.prim_mst(weight_graph)
# test set prediction
test_pred_tan, test_postPr_tan = bf.testset_prediction_tan(instance_data_trn, instance_data_test, var_ranges, label_range, V_new)

"""Naive Bayes prediction"""
test_pred_nb, test_postPr_nb = bf.testset_predict_nb(instance_data_trn, instance_data_test, var_ranges, label_range)

"""Compute confidence"""
# tan
positive_tan = [pred_t==label_range[0] for pred_t in test_pred_tan]
confidence_tan = []
for i in range(len(instance_data_test)):
    if positive_tan[i] is True:
        confidence_tan.append(test_postPr_tan[i])
    else:
        confidence_tan.append(1-test_postPr_tan[i])

# nb
positive_nb = [pred_n==label_range[0] for pred_n in test_pred_nb]
confidence_nb = []
for i in range(len(instance_data_test)):
    if positive_nb[i] is True:
        confidence_nb.append(test_postPr_nb[i])
    else:
        confidence_nb.append(1-test_postPr_nb[i])

"""Plot PRC"""
PRC_tan, ROC_tan = PRC_ROC(confidence_tan, test_labels, label_range)
PRC_nb, ROC_nb = PRC_ROC(confidence_nb, test_labels, label_range)
[Recall_tan, Precision_tan] = zip(*PRC_tan)
[Recall_nb, Precision_nb] = zip(*PRC_nb)
plt.plot(Recall_tan, Precision_tan, label='TAN')
plt.plot(Recall_nb, Precision_nb, label='Naive Bayes')
plt.legend()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PRC for TAN and Naive Bayes')
plt.grid()
plt.show()


"""Plot ROC"""
[FP_tan, TP_tan] = zip(*ROC_tan)
[FP_nb, TP_nb] = zip(*ROC_nb)
plt.plot(FP_tan, TP_tan, label='TAN')
plt.plot(FP_nb, TP_nb, label='Naive Bayes')
plt.legend()
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC for TAN and Naive Bayes')
plt.grid()
plt.show()
