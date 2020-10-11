from sklearn import ensemble

import utils

training_points, training_labels = utils.get_training_data('../train_2008.csv')
test_points = utils.get_test_points('../test_2008.csv')


forest = ensemble.RandomForestClassifier(n_estimators=30,
                                         max_features='sqrt',
                                         verbose=1)
print("Starting training...")
forest.fit(training_points, training_labels)

utils.prepare_submission_sklearn(forest.predict, test_points)
