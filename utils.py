import csv
import numpy as np

index_id = 0
index_PES1 = 382  # fuck it
training_set_filename = 'data/train_2008.csv'
test_set_filename = 'data/test_2008.csv'

# Works specifically with 'train_2008.csv'
def get_training_data(filename=training_set_filename):
    with open(filename, 'r') as data_file:
        reader = csv.reader(data_file, delimiter=',', quotechar='"')
        raw_data = [row for row in reader]
    training_data = np.asarray(raw_data[1:], dtype='f')  # cut header
    training_labels = training_data[:, index_PES1]
    training_points = np.delete(arr=training_data, obj=index_PES1, axis=1)
    return training_points, training_labels


# Works specifically with 'test_2008.csv'
def get_test_points(filename=test_set_filename):
    with open(filename, 'r') as data_file:
        reader = csv.reader(data_file, delimiter=',', quotechar='"')
        raw_data = [row for row in reader]
    return np.asarray(raw_data[1:], dtype='f')  # cut header


# Works for a general csv without headers.
# Returns a numpy ndarray with the data. Data type is inferred.
def extract_csv(filename):
    with open(filename, 'r') as data_file:
        reader = csv.reader(data_file, delimiter=',', quotechar='"')
        raw_data = [row for row in reader]
    return np.asarray(raw_data)


# Format a set of test labels into the desired submission format.
# The submission file is specified by 'out_file.'
def format_submission(test_labels,
                      out_file='submission.csv'):
    size = len(test_labels)
    # Create ids
    id_col = np.empty(shape=[size, 1], dtype=np.int)
    for i in range(id_col.shape[0]):
        id_col[i][0] = i

    labels_col = np.reshape(test_labels, newshape=[size, 1])

    # horizontal concatenation
    submission_array = np.concatenate(id_col, labels_col, axis=1)

    # Write the csv file
    np.savetxt(fname=out_file,
               X=submission_array,
               fmt='%i',
               delimiter=',',
               header='id,PES1',
               comments='')


# 'X' is training points, 'y' is training labels
# 'proportion' is a float < 1 that indicates what proportion of examples should
#   be used for validation
def split_validation_set(X, y, proportion):
    assert len(X) == len(y)
    v_size = int(len(X) * proportion)
    v_indices = np.random.choice(a=len(X), size=v_size, replace=False)

    X_valid = X[v_indices]
    y_valid = y[v_indices]
    X_train = np.delete(X, obj=v_indices, axis=0)
    y_train = np.delete(y, obj=v_indices, axis=0)
    return (X_train, y_train), (X_valid, y_valid)