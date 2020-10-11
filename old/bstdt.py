from sklearn import ensemble

import utils

# this is pure shit.

training_points, training_labels = utils.get_training_data('../train_2008.csv')
test_points = utils.get_test_points('../test_2008.csv')

clf = ensemble.AdaBoostClassifier()  # uses DT by default
clf.fit(training_points, training_labels)

utils.prepare_submission_sklearn(clf.predict, test_points)
