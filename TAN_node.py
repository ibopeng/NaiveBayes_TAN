"""
Project: Naive Bayes and TAN

A TAN node class

@ Author: Bo Peng
@ University of Wisconsin - Madison
"""


class TanNode:

    def __init__(self, var_idx):
        self.parents = [-1] # each variable <class> is one of the parents
        self.children = []
        self.var_idx = var_idx
        self.is_root = False
        self.is_new = False
