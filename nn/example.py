import utils
import nn.nnutils as nnutils
import numpy as np
import keras
from keras.layers.core import Dense

training_points, training_labels = \
    utils.get_training_data("../data/train_2008.csv")

nnutils.standardize_labels(training_labels)

assert len(training_points) == len(training_labels)
validation_set_size = len(training_points) / 6
validation_indices = np.random.choice(a=len(training_points),
                                      size=int(validation_set_size),
                                      replace=False)

print("Separating into training and validation sets...")
X_valid = training_points[validation_indices]
y_valid = training_labels[validation_indices]
X_train = np.delete(training_points, obj=validation_indices, axis=0)
y_train = np.delete(training_labels, obj=validation_indices, axis=0)

print("Creating Keras model...")
model = keras.models.Sequential()
model.add(Dense(30, input_dim=X_train.shape[1], activation='tanh'))
model.add(Dense(100, activation='tanh'))
model.add(Dense(1, activation='softmax'))
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, nb_epoch=10)

model.summary()

print("validation/hillclimb set performance:")
validation_metrics = model.evaluate(X_valid, y_valid, verbose=0)
for i in range(len(validation_metrics)):
    print(model.metrics_names[i] + ": " + str(validation_metrics[i]))


