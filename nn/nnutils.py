# Utility functions specific to NN

from __future__ import division

import numpy as np
from keras import backend as K

# The labels are '1' and '2'
# This changes all '2' labels to '-1'
def standardize_labels(array):
    for y in np.nditer(array, op_flags=['readwrite']):
        if y == 2:
            y[...] = 0


# Performs the reverse of standardize_labels
def unstandardize_labels(array):
    for y in np.nditer(array, op_flags=['readwrite']):
        if y == 0:
            y[...] = 2


# From LeCun's paper 'Efficient BackProp'
# Should be Keras-compatible
def lecun_tanh(x):
    return K.dot(1.7159, K.tanh(K.dot(2/3, x)))
