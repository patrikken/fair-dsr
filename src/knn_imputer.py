from sklearn.impute import KNNImputer 
import numpy as np

def knn_impute(X_train, s_train, X_test):
    imputer = KNNImputer(n_neighbors=3)
    #print(X_train.shape, X_test.shape) 
    if s_train.ndim == 1:
        X = np.hstack((X_train, s_train[:, np.newaxis]))
    else:
        X = np.hstack((X_train, s_train))
    #print(X.shape) 
    imputer.fit(X)

    missing_val = np.empty([X_test.shape[0], 1])
    missing_val[:] = np.nan 

    X_t = np.hstack((X_test, missing_val))

    imputed = imputer.transform(X_t)
    
    return imputed[:, -1].round()