from sklearn.model_selection import TimeSeriesSplit

def walk_forward_split(X, y, n_splits=5):
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        splits.append((X_train, X_test, y_train, y_test))
    return splits
