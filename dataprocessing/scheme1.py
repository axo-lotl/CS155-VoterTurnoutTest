# Convert categorical features (with 20 or less categories) to 1-hot vectors.
# Normalize each feature to have mean 0 and std/variance 1.

import utils
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Maximum number of unique values a feature can attain and still be considered
#   categorical
max_categories = 20

X_train, y_train = \
    utils.get_training_data('../data/train_2008.csv')
X_test = utils.get_test_points('../data/test_2008.csv')

print("X_train shape" + str(X_train.shape))
print("y_train shape" + str(y_train.shape))
print("X_test shape" + str(X_test.shape))

training_set_size = X_train.shape[0]

# Concatenate X_train and X_test (vertically) to apply operations together.
X_all = np.concatenate([X_train, X_test], axis=0)

# Count the number of unique values assumed by each feature.
cat_cols = []  # columns deemed categorical
cat_unique_values = []  # unique values for each categorical feature
noncat_cols = []  # non-categorical columns
for col_num in range(X_all.shape[1]):
    column = X_all[:, col_num:col_num+1]  # (N,1) ndarray
    unique_values = np.unique(column)
    if len(unique_values) <= max_categories:
        cat_cols.append(column)
        cat_unique_values.append(unique_values)
    else:
        noncat_cols.append(column)

assert len(cat_cols) == len(cat_unique_values)



# To process this for OneHotEncoder, we need to make sure that if there are
# k unique values of features in the column, they they take values 0,1,...,k-1

# First, for each feature we create a dictionary from unique_values to its
# indices
unique_dicts = [0] * len(cat_unique_values)
for ii in range(len(unique_dicts)):
    unique_dict = {}
    unique_values = cat_unique_values[ii]
    for jj in range(len(unique_values)):
        unique_dict[unique_values[jj]] = jj
    unique_dicts[ii] = unique_dict


for cat_feature_num in range(len(cat_cols)):
    column = cat_cols[cat_feature_num]
    unique_values = cat_unique_values[cat_feature_num]
    unique_dict = unique_dicts[cat_feature_num]
    for row_index in range(column.shape[0]):
        # Find the match within unique_values; replace the feature value with
        # the index of the value within unique_values
        column[row_index][0] = unique_dict.get(column[row_index][0])

cat_matrix = np.concatenate(cat_cols, axis=1)
encoder = OneHotEncoder(dtype=np.float, sparse=False)

enc_cat_matrix = encoder.fit_transform(cat_matrix)
noncat_matrix = np.concatenate(noncat_cols, axis=1)
enc_matrix = np.concatenate([enc_cat_matrix, noncat_matrix], axis=1)

print("1-hot encoded matrix (train and test): " + str(enc_matrix.shape))

# Scaling / normalization

feature_means = np.mean(a=enc_matrix, axis=0)
feature_stds = np.std(a=enc_matrix, axis=0)


for col in range(enc_matrix.shape[1]):
    if feature_stds[col] != 0:
        for row in range(enc_matrix.shape[0]):
            enc_matrix[row][col] = (enc_matrix[row][col] - feature_means[col]) \
                                   / feature_stds[col]
    else:
        for row in range(enc_matrix.shape[0]):
            enc_matrix[row][col] = enc_matrix[row][col] - feature_means[col]

np.save(file="s1trainingpoints",
        fix_imports=True,
        arr=enc_matrix[0:training_set_size, :])
np.save(file="s1testpoints",
        fix_imports=True,
        arr=enc_matrix[training_set_size:, :])
np.save(file="s1traininglabels",
        fix_imports=True,
        arr=y_train)