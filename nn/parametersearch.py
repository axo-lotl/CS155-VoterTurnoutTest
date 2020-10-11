from __future__ import division

import tensorflow as tf
import utils
import nn.nnutils as nnutils
import numpy as np
import keras
from keras.layers.core import Dense, Dropout

tf.python.control_flow_ops = tf

log_file = "trials.txt"
training_points = np.load(file='../dataprocessing/s1trainingpoints.npy')
training_labels = np.load(file='../dataprocessing/s1traininglabels.npy')

nnutils.standardize_labels(training_labels) # change labels to +1, -1

print("Training points: " + str(training_points.shape))
print("Training labels: " + str(training_labels.shape))

# Split 1/8 of the data for a validation set
(X_train, y_train), (X_valid, y_valid) = \
    utils.split_validation_set(training_points, training_labels, 0.125)

# Try these parameters, and log the results in the 'log_file' specified above.
def try_parameters(hidden_layers,
                   X_train,
                   y_train,
                   X_valid,
                   y_valid,
                   n_epochs=10,
                   loss='mean_squared_error',
                   optimizer='adam'):
    assert len(hidden_layers) >= 1
    model = keras.models.Sequential()

    # Connect first hidden layer
    model.add(Dense(hidden_layers[0][0],
                    input_dim=X_train.shape[1],
                    activation=hidden_layers[0][1]))
    if hidden_layers[0][2] != 0:
        model.add(Dropout(hidden_layers[0][2]))

    for i in range(1, len(hidden_layers)):
        hidden_layer = hidden_layers[i]
        model.add(Dense(hidden_layer[0], activation=hidden_layer[1]))
        if hidden_layer[2] != 0:
            model.add(Dropout(hidden_layer[2]))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    model.fit(X_train, y_train, nb_epoch=n_epochs)

    with open(log_file, 'a') as file:
        config = model.get_config()
        for element in config:
            file.write(str(element) + '\n')
        # file.write(str(hidden_layers) + '\n')
        # file.write('epochs=' + str(n_epochs))
        # file.write(', loss: ' + str(loss))
        # file.write(', optimizer' + str(optimizer) + '\n')

        file.write(str(model.optimizer) + '\n')

        metrics = model.evaluate(X_valid, y_valid, verbose=0)
        for i in range(len(metrics)):
            file.write(model.metrics_names[i] + '=' + str(metrics[i]) + ' // ')
        file.write('\n\n')


# Only dense layers are used.
# (num neurons, activation, dropout) for represent each layer
hidden_layers = [(300, 'relu', 0.3), (200, 'relu', 0), (100, 'relu', 0)]

try_parameters(hidden_layers,
               X_train,
               y_train,
               X_valid,
               y_valid,
               optimizer='adam',
               n_epochs=2)

