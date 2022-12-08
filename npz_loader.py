import numpy as np

SAMPLES_PER_GROUP = 44
COLUMNS_PER_GROUP = 116

def load_X_data(filepath: str):
    X = np.load(filepath)
    X = X.f.arr_0
    X = X.reshape((X.shape[0], -1, SAMPLES_PER_GROUP, COLUMNS_PER_GROUP))

    return X

def load_y_data(filepath: str):
    y = np.load(filepath)
    y = y.f.arr_0

    return y

def merge_X_y_data(X, y):
    data = [[X[i], y[i]] for i in range(X.shape[0])]
    return data
