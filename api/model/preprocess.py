import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Même stratégie que ton notebook
imputer = SimpleImputer(strategy="median")

def preprocess_X(X):
    """
    Applique l'imputation sur la matrice X (shape: [1, n_features])
    """
    X = np.array(X, dtype=float)
    return imputer.fit_transform(X) if not hasattr(imputer, "statistics_") else imputer.transform(X)
