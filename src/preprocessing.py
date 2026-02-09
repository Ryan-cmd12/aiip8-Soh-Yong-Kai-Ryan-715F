from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from .config import (Numerical_Imputer_Strategy, Categorical_Imputer_Strategy, Target, Numerical_cols, Categorical_cols)

def preprocessor():
    numerical_pipe = Pipeline([
        ("scaler", RobustScaler(with_centering =True, with_scaling =True)),
        ("imputer", Numerical_Imputer_Strategy),
    ])

    categorical_pipe = Pipeline([
        ("imputer", Categorical_Imputer_Strategy),
        ("onehot", OneHotEncoder(handle_unknown= "ignore")),
    ])

    return ColumnTransformer([
        ("num", numerical_pipe, Numerical_cols),
        ("cate", categorical_pipe, Categorical_cols)
    ])

def preprocessor_dense():
        numerical_pipe = Pipeline([
        ("scaler", StandardScaler(with_mean = False)),
        ("imputer", Numerical_Imputer_Strategy),
    ])

        categorical_pipe = Pipeline([
            ("imputer", Categorical_Imputer_Strategy),
            ("onehot", OneHotEncoder(handle_unknown = "ignore", sparse_output =False)),
        ])

        return ColumnTransformer([
            ("num", numerical_pipe, Numerical_cols),
            ("cate", categorical_pipe, Categorical_cols)
        ])
