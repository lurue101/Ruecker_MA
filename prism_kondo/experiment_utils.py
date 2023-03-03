import json
import os
from datetime import datetime, timezone
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from prism_kondo._constants import SELECTOR_DICT
from prism_kondo.instance_selection._params_dict import PARAMS_DICT
from prism_kondo.model import train_lr_model
from prism_kondo.utils import transform_selector_output_into_mask


def remove_twentyeight_days(
    X_to_remove_from: np.ndarray,
    y_to_remove_from: np.ndarray,
    recorded_at_to_remove_from: np.ndarray,
    target_recorded_at: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Takes out all samples that have a recorded_at date between 1 and 27 days before the target.
    Parameters
    ----------
    X_to_remove_from
        feature array from which to remove samples that are within 28 day range older than the target
    y_to_remove_from
        target array from which to remove samples that are within 28 day range older than the target
    recorded_at_to_remove_from
        time array from which to remove samples that are within 28 day range older than the target
    target_recorded_at
        target date

    Returns
    -------
    arrays with the samples removed
    """
    due_date = target_recorded_at - np.timedelta64(28, "D")
    subset_idx = np.argwhere(recorded_at_to_remove_from.flatten() < due_date).flatten()
    return (
        X_to_remove_from[subset_idx, :],
        y_to_remove_from[subset_idx],
        recorded_at_to_remove_from[subset_idx],
    )


def remove_twentyeight_days_df(
    df_train: pd.DataFrame,
    target_recorded_at: np.ndarray,
) -> pd.DataFrame:
    """
    Takes out all samples that have a recorded_at date between 1 and 27 days before the target.
    Parameters
    ----------
    df_train
    target_recorded_at
        target date

    Returns
    -------
    """
    target_recorded_at = pd.to_datetime(target_recorded_at)
    due_date = target_recorded_at - np.timedelta64(28, "D")
    return df_train[pd.to_datetime(df_train.recorded_at) < due_date]


def split_df_into_arrays(
    df: pd.DataFrame, target_column: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the dataframe into feature, target and time arrays for further use in algorithms

    Parameters
    ----------
    df
        dataframe containing features, target and time columns
    target_column
        name of the target column

    Returns
    -------
    tuple of the arrays
    """
    return (
        df.drop(columns=[target_column, "recorded_at"], inplace=False).to_numpy(
            dtype="float32"
        ),
        df[target_column].to_numpy(dtype="float32"),
        df["recorded_at"].to_numpy().reshape(-1, 1),
    )


def run_selector(
    X_train: np.ndarray, y_train: np.ndarray, selector_name: str, params_dict: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs the specified IS algorithm, calculating the labels for each input sample
    Parameters
    ----------
    X_train
        features of the input samples
    y_train
        target array of the input samples
    selector_name
        name of the IS algorithm to use
    params_dict
        dict containing the hyper parameter values for the IS algorithm

    Returns
    -------
    Array containing boolean labels to indicate which samples the algorithm selected
    """
    # Gets the class of the IS algorithm from the name
    selector_class = SELECTOR_DICT[selector_name]
    # Creates and instance of the IS class with the specified parameters
    selector_obj = selector_class(**params_dict)
    # runs the IS algorithm and transforms the integer labels into boolean labels
    boolean_labels = transform_selector_output_into_mask(
        selector_obj.fit_predict(X_train, y_train)
    )
    if selector_name in [
        "reg_enn",
        "reg_enn_time",
        "reg_cnn",
        "drop_two_re",
        "drop_two_rt",
        "drop_three_re",
        "drop_three_rt",
        "shapley",
        "lof",
        "fixed_window",
        "selcon",
        "mutual_information",
        "fish1",
    ]:
        return boolean_labels, selector_obj.scores
    else:
        return boolean_labels, np.ndarray([])


def calc_model_errors(
    model: LinearRegression, X_test: np.ndarray, y_test: np.ndarray
) -> dict[str, float]:
    """
    Calculates the mean absolute error, mean squared error, mean absolute percentage error and the R^2 coefficient,
    which technically is not an error, for the given model and samples
    Parameters
    ----------
    model
        trained model to predict target variable y from X
    X_test
        features to predict values in y_test
    y_test
        true values for the given features in X_test

    Returns
    -------
    dict containing all three errors
    """
    mae = mean_absolute_error(y_test, model.predict(X_test))
    mse = mean_squared_error(y_test, model.predict(X_test))
    mape = mean_absolute_percentage_error(y_test, model.predict(X_test))
    r2 = r2_score(y_test, model.predict(X_test))
    return {"mae": float(mae), "mse": float(mse), "mape": float(mape), "r2": float(r2)}


def check_and_fix_np_dtypes(np_type_dict: dict) -> dict:
    """
    Converts numpy specific datatypes into native python types, so that they can be saved into a json file
    Parameters
    ----------
    np_type_dict
        dict containing potential numpy datatypes

    Returns
    -------
    dict with only native python data types
    """
    python_type_dict = {}
    for key in np_type_dict.keys():
        if "numpy.ndarray" in str(type(np_type_dict[key])):
            python_type_dict[key] = np_type_dict[key].tolist()
        elif "numpy" in str(type(np_type_dict[key])):
            python_type_dict[key] = np_type_dict[key].item()
        else:
            python_type_dict[key] = np_type_dict[key]
    return python_type_dict


def save_hyperopt_result(
    selector_name: str,
    selector_params: dict,
    error_dict: dict,
    company_slug: str,
    suffix: str = "",
    directory_path: str = "hyperopt_results",
):
    hyperopt_dict = {
        "selector_name": selector_name,
    }
    # check if any values are numpy dtypes and convert them as they can't be written to json
    selector_params = check_and_fix_np_dtypes(selector_params)
    # add all available information into one dict
    hyperopt_dict.update(selector_params)
    hyperopt_dict.update(error_dict)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S.%f")
    directory_path = os.path.join(directory_path, selector_name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    filename = f"{company_slug}_hyperopt_{timestamp}__{suffix}"
    full_path = os.path.join(directory_path, filename)
    with open(full_path, "w") as outfile:
        json.dump(hyperopt_dict, outfile)


def save_hyperopt_log(
    selector_name: str, param_dict_ranges: dict, company_slug: str, extra_info: str = ""
):
    """
    Saves a json with all the important information about which hyperparamter optimization has been run

    Parameters
    ----------
    selector_name
        name of the IS algorithm
    param_dict_ranges
        dict containing all values for each parameter
    company_slug
        slug indicating the company

    Returns
    -------

    """
    hyperopt_dict = {
        "selector_name": selector_name,
    }
    # transform numpy dtypes to native python types so they can be saved into a json
    param_dict_ranges = check_and_fix_np_dtypes(param_dict_ranges)
    hyperopt_dict.update(param_dict_ranges)
    hyperopt_dict["extra_info"] = extra_info
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S.%f")
    directory_path = "hyperopt_log"
    directory_path = os.path.join(directory_path, selector_name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    filename = f"{company_slug}_hyperopt_log_{timestamp}"
    full_path = os.path.join(directory_path, filename)
    with open(full_path, "w") as outfile:
        json.dump(hyperopt_dict, outfile)


def save_experiment_results(
    selector_name: str,
    company_slug: str,
    boolean_labels: np.ndarray,
    scores: np.ndarray,
    error_dict: dict[str, float],
    data_date: str,
    feature_info: str,
    params: dict,
    directory_path: str = "results",
):
    """
    Saves the result of a experiment in a json. A experiment consists of running a IS algorithm, training a model with
    the chosen samples and then calculating the error on the test set. The results_no_one_hot are saved twice, once in directories
    sorted by algorithms, once sorted by companies

    Parameters
    ----------
    selector_name
        name of the IS algorithm to use
    company_slug
        name of the company to work with
    boolean_labels
        result of the IS algorithm
    scores
        score that rates if an instance should be included, the higher the better
    error_dict
        dict containing the mae, mse and mape error on the test set
    data_date
        which version of company data to use
    feature_info
        which features were used
    params
        dict containing the hyperparameteres of the IS that were used
    directory_path
        directory in which to save the results_no_one_hot

    Returns
    -------

    """
    directory_path_selector = os.path.join(directory_path, selector_name)
    directory_path_company = os.path.join(directory_path, company_slug)
    experiment_dict = {
        "selector_name": selector_name,
        "labels": boolean_labels.tolist(),
        "scores": scores.tolist(),
        "features": feature_info,
        "params": params,
    }
    experiment_dict.update(error_dict)
    if not os.path.exists(directory_path_selector):
        os.makedirs(directory_path_selector)
    if not os.path.exists(directory_path_company):
        os.makedirs(directory_path_company)
    filename = f"{company_slug}_{data_date}"
    full_path_selector = os.path.join(directory_path_selector, filename)
    full_path_company = os.path.join(directory_path_company, filename)
    with open(full_path_selector, "w") as outfile:
        json.dump(experiment_dict, outfile)
    with open(full_path_company, "w") as outfile:
        json.dump(experiment_dict, outfile)


def create_results_df(
    base_directory: str, selector_name: str, file_name_must_include: str = ""
):
    """
    Combines the json files for the given selector in the directory into a dataframe

    Parameters
    ----------
    base_directory
        directory which contains folders for each selector containing jsons
    selector_name
        name of the IS algorithm
    file_name_must_include
        str that must be included in the filename to be put into the df

    Returns
    -------
    df that contains the information from the relevant jsons
    """
    # create path, where the json lie
    selector_path = os.path.join(base_directory, selector_name)
    dict_list = []
    merged_hyperopt_dict = {}
    for file_name in os.listdir(selector_path):
        if file_name_must_include not in file_name:
            continue
        company_slug = file_name.split("_")[0]
        full_path = os.path.join(selector_path, file_name)
        with open(full_path, "r") as file:
            hyperopt_dict = json.load(file)
            hyperopt_dict["company_slug"] = company_slug
            if "__" in file_name:
                hyperopt_dict["suffix_info"] = file_name.split("__")[1]
            dict_list.append(hyperopt_dict)
    for key in dict_list[0].keys():
        merged_hyperopt_dict[key] = [
            single_hyperopt[key] for single_hyperopt in dict_list
        ]
    df = pd.DataFrame.from_dict(merged_hyperopt_dict)
    return df


def get_features_from_keywords(columns: pd.Index, keywords: list[str]) -> list[str]:
    """
    Returns all features in the given that include one of passed keywords
    Parameters
    ----------
    columns
        names of the columns of a dataframe
    keywords
        keywords describing a single column of a measurement method (ie xrd, xrf)
    Returns
    -------
    list of columns names
    """
    features = []
    for keyword in keywords:
        features = features + [col for col in columns if keyword in col]
    return features


def read_featureopt_results_into_df():
    dict_list = []
    merged_featureopt_dict = {}
    directory_path = "/Users/rueck/alcemy/prism_kondo/prism_kondo/featureopt"
    for file_name in os.listdir(
        "/Users/rueck/alcemy/prism_kondo/prism_kondo/featureopt"
    ):
        full_path = os.path.join(directory_path, file_name)
        with open(full_path, "r") as file:
            featureopt_dict = json.load(file)
            dict_list.append(featureopt_dict)
    for key in dict_list[0].keys():
        merged_featureopt_dict[key] = [
            single_hyperopt[key] for single_hyperopt in dict_list
        ]
    df = pd.DataFrame.from_dict(merged_featureopt_dict)
    return df


def run_featureopt(df, features, company_slug, info):
    df_opt = df.loc[:, features + ["recorded_at", "lab__press__cs_28d"]].copy()
    (
        X_train,
        y_train,
        recorded_at_train,
    ) = split_df_into_arrays(df_opt, "lab__press__cs_28d")
    (
        X_train,
        X_val,
        y_train,
        y_val,
        recorded_at_train,
        recorded_at_val,
    ) = train_test_split(
        X_train, y_train, recorded_at_train, test_size=0.15, shuffle=False
    )
    X_train, y_train, recorded_at_train = remove_twentyeight_days(
        X_train, y_train, recorded_at_train, recorded_at_val[0]
    )
    for model_type in ["linear", "lasso_lars_cv"]:
        model = train_lr_model(X_train, y_train, model_type)
        error_dict = calc_model_errors(model, X_val, y_val)
        featureopt_dict = {
            "company": company_slug,
            "features": info,
            "model_type": model_type,
        }
        featureopt_dict.update(error_dict)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S.%f")
        filename = f"{company_slug}_featureopt_{timestamp}__{info}__{model_type}"
        full_path = os.path.join("featureopt/", filename)
        with open(full_path, "w") as outfile:
            json.dump(featureopt_dict, outfile)


def get_param_names(selector_name: str):
    """
    Retrieves the names of all parameter the chosen selector uses

    Parameters
    ----------
    selector_name
        name of the selector

    Returns
    -------
    list of parameter names
    """
    params_dict = PARAMS_DICT[selector_name]
    return list(params_dict.keys())


def get_best_params_from_hyperopt(selector_name, dataset) -> dict:
    """
    Automatically chooses the best set of parameters from the results of the hyperopts

    Parameters
    ----------
    selector_name
        name of selector
    dataset
        name of the keel dataset
    Returns
    -------
    dict containing parameter name and value pairs
    """
    df_results = create_results_df(
        "/Users/rueck/alcemy/prism_kondo/prism_kondo/hyperopt_keel", selector_name
    )
    param_names = get_param_names(selector_name)
    best_params = (
        df_results.groupby("suffix_info").get_group(dataset).sort_values("mape").iloc[0]
    )
    params_dict = {}
    for param in param_names:
        params_dict[param] = best_params[param]
    return params_dict


def add_random_noise_arnaiz(
    y: np.ndarray, noise_frac: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    As described in paper "Instance Selection for regression" by Arnaiz-Gonzalez under 4.5, to add random noise
    for some % of the samples, we simply exchange the target values. Thus neither the feature nor target distribution
    is changed

    Parameters
    ----------
    y
        array containing target values
    noise_frac
        % of samples that are affected. Thus noise_pct/2 is the number of affected sample pairs

    Returns
    -------
    array with the swapped values
    """
    if noise_frac == 0:
        return y, []
    else:
        y_noisy = y.copy()
        possible_idx = np.arange(len(y_noisy))
        nr_swapping_pairs = int(len(y_noisy) * noise_frac / 2)
        swapping_pairs = np.random.choice(
            possible_idx, (nr_swapping_pairs, 2), replace=False
        )
        first_half = swapping_pairs[:, 0]
        second_half = swapping_pairs[:, 1]
        y_noisy[first_half], y_noisy[second_half] = (
            y_noisy[second_half],
            y_noisy[first_half],
        )
        noisy_indices = swapping_pairs.flatten()
    return y_noisy, noisy_indices
