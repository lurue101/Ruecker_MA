import numpy as np

from sklearn.linear_model import LassoLarsCV, LinearRegression


def train_lr_model(X, y, model_type: str = "linear"):
    if model_type == "linear":
        reg = LinearRegression()
    elif model_type == "lasso_lars_cv":
        reg = LassoLarsCV(normalize=False, max_iter=500)
    else:
        raise ValueError(f"{model_type} is not implemented")
    reg.fit(X, y)
    return reg


def get_random_intercept_coef(nr_of_features):
    rnd_intercept = np.random.normal(0, 2)
    rnd_coefs = np.random.normal(0, 2, nr_of_features)
    return (rnd_intercept, rnd_coefs)


def get_lr_model_random_params(nr_of_features, model_type="linear"):
    if model_type == "linear":
        reg = LinearRegression()
    elif model_type == "lasso_lars_cv":
        reg = LassoLarsCV(normalize=False, max_iter=50)
    else:
        raise ValueError("we don't have this model")
    rnd_intercept, rnd_coefs = get_random_intercept_coef(nr_of_features)
    reg.intercept_ = rnd_intercept
    reg.coef_ = rnd_coefs
    return reg
