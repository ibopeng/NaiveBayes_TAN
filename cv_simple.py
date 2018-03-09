import numpy as np
import scipy.stats as stats

acc_delta = np.loadtxt('acc_delta.txt')
num_fold = len(acc_delta)

"""1. Sample Mean"""
sm = 0
for ad in acc_delta:
    sm += ad
sample_mean = sm / num_fold
print('Sample means = {0:.12f}'.format(sample_mean))

"""2. t statistic"""
dof = num_fold - 1  # degree of freedom: n-1
acc_delta_mean_removal = [ad - sample_mean for ad in acc_delta]
sv = 0
for admr in acc_delta_mean_removal:
    sv += admr * admr
sample_variance = sv / dof
# Null hypothesis is: sample_mean = 0, i.e., no accuracy difference between TAN and NB
t = (sample_mean - 0) / np.sqrt(sample_variance / num_fold)
print('t statistic = {0:.12f}'.format(t))

t_tatistic, p_value = stats.ttest_1samp(acc_delta, 0.0)
print(t_tatistic)
print(p_value)