import json
import os

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

from prism_kondo.experiment_utils import (
    create_results_df,
    get_features_from_keywords,
    remove_twentyeight_days_df,
)
from prism_kondo.experiments import ExperimentRunner
from prism_kondo.instance_selection._params_dict import PARAMS_DICT_COMPANIES
from prism_kondo.utils import calc_pct_increase


def get_results_filepath(company_slug, selector_name, directory="results"):
    file_directory = os.path.join(directory, selector_name)
    file_list = os.listdir(file_directory)
    filename = [file for file in file_list if company_slug in file]
    if len(filename) > 1:
        raise ValueError(
            f"result are not unique. There is >1 file for company {company_slug} and selector {selector_name}"
        )
    return os.path.join(file_directory, filename[0])


def get_results_dict_from_filepath(file_path):
    with open(file_path, "r") as f:
        results_dict = json.load(f)
    return results_dict


def create_labels_df(
    company_slug: str, selectors: list[str], results_dir: str = "results"
):
    """
    Creates a dataframe, where each column contains the labels that are given by an IS algorithm, indicating which
    samples should be used for training
    Parameters
    ----------
    company_slug
        name of the company
    selectors
        list of selectors to use
    results_dir
        rel path to folder with results_no_one_hot

    Returns
    -------
    dataframe with labels as columns
    """
    labels = {}
    for selector_name in selectors:
        filepath = get_results_filepath(company_slug, selector_name, results_dir)
        results_dict = get_results_dict_from_filepath(filepath)
        labels[selector_name] = results_dict["labels"]
        labels[f"{selector_name}_scores"] = results_dict["scores"]
    labels_df = pd.DataFrame.from_dict(labels)
    return labels_df


def get_train_df(company_slug, data_date="2022_12_23", directory_data="data"):
    file_path = os.path.join(directory_data, f"{company_slug}_train_{data_date}.csv")
    df = pd.read_csv(file_path)
    return df


def get_sample_which_selector_kicked_out(
    company_slug, selector_name, data_date="2022_12_23", directory_data="data"
):
    samples = get_train_df(company_slug, data_date, directory_data)
    labels = create_labels_df(
        company_slug,
        [selector_name],
        "/Users/rueck/alcemy/prism_kondo/prism_kondo/results/",
    )
    # to remove last 28 days we just take the index that correspond to length of labels
    samples = samples.iloc[: len(labels), :]
    return samples.loc[~labels[selector_name].values]


def df_frac_remaining_after_IS(companies, selectors, dir_results="results"):
    frac_kept = {}
    for company_slug in companies:
        frac_kept[company_slug] = {}
        for selector_name in selectors:
            filepath = get_results_filepath(company_slug, selector_name, dir_results)
            with open(
                filepath,
                "r",
            ) as f:
                results_dict = json.load(f)
                frac_kept[company_slug][selector_name] = np.round(
                    sum(results_dict["labels"]) / len(results_dict["labels"]), 2
                )
    df = pd.DataFrame.from_dict(frac_kept, orient="index")
    return df


def remove_outlier_error(company_slug, results_dict, file_name_date):
    exp = ExperimentRunner(company_slug=company_slug, file_name_date=file_name_date)
    errors = exp.get_pred_errors_from_labels(results_dict["labels"])
    rm_idx = np.argmax(errors > 50)

    print(np.mean(errors), results_dict["mae"])
    print(np.mean(np.delete(errors, rm_idx)))


def create_test_error_df(
    companies: list[str],
    selectors: list[str],
    error_type: str = "mape",
    pct_increase: bool = True,
    results_dir="results",
):
    final_result = {}
    for company_slug in companies:
        final_result[company_slug] = {}
        filepath = get_results_filepath(company_slug, "full", results_dir)
        with open(
            filepath,
            "r",
        ) as f:
            results_dict_full = json.load(f)
            full_set_error = results_dict_full[error_type]
        for selector_name in selectors:
            filepath = get_results_filepath(company_slug, selector_name, results_dir)
            with open(
                filepath,
                "r",
            ) as f:
                results_dict = json.load(f)
                """if company_slug == "amoeneburg":
                    remove_outlier_error(company_slug,results_dict, "2023_02_12")"""
                if pct_increase:
                    final_result[company_slug][selector_name] = np.round(
                        calc_pct_increase(full_set_error, results_dict[error_type]), 2
                    )
                else:
                    final_result[company_slug][selector_name] = np.round(
                        results_dict[error_type] - full_set_error, 4
                    )
    results_df = pd.DataFrame.from_dict(final_result, orient="index")
    return results_df


def raw_performance_df(
    companies: list[str],
    selectors: list[str],
    metric: str = "mape",
    results_dir="results",
):
    final_result = {}
    for company_slug in companies:
        final_result[company_slug] = {}
        filepath = get_results_filepath(company_slug, "full", results_dir)
        with open(
            filepath,
            "r",
        ) as f:
            results_dict_full = json.load(f)
            full_set_error = results_dict_full[metric]
            final_result[company_slug]["full"] = full_set_error
        for selector_name in selectors:
            filepath = get_results_filepath(company_slug, selector_name, results_dir)
            with open(
                filepath,
                "r",
            ) as f:
                results_dict = json.load(f)
                final_result[company_slug][selector_name] = results_dict[metric]
    results_df = pd.DataFrame.from_dict(final_result, orient="index")
    return results_df


def calc_frac_kept_by_cement_slug(
    company_slug: str, selectors: list[str], data_date: str, results_dir="results"
):
    df = pd.read_csv(
        f"/Users/rueck/alcemy/prism_kondo/prism_kondo/data_not_normalized/{company_slug}_train_{data_date}.csv"
    )
    df_labels = create_labels_df(
        company_slug,
        selectors,
        results_dir=results_dir,
    )
    # labels already has 28d removed, df not
    df = df.iloc[: len(df_labels), :]
    df = pd.concat([df, df_labels], axis=1)
    df_frac_total = pd.DataFrame(
        df_labels.sum(axis=0) / len(df_labels), columns=["overall"]
    ).T
    df_cements = pd.DataFrame()
    for selector in selectors:
        df_cements_frac = (
            df[df[selector] == True].groupby("cement_slug").count()[selector]
            / df.groupby("cement_slug").count()[selector]
        )
        df_cements = pd.concat([df_cements, df_cements_frac], axis=1)
    df_cements = pd.concat([df_frac_total, df_cements], axis=0)
    df_cements = df_cements.round(decimals=2)
    return df_cements


def get_pca(
    company_slug: str,
    data_date: str,
    n_components=2,
    feature_keywords: list[str] = ["xrd", "xrf", "perc_63", "slope"],
    unnormalized_data_dir: str = "data_not_normalized",
):
    df = pd.read_csv(
        os.path.join(unnormalized_data_dir, f"{company_slug}_train_{data_date}.csv")
    )
    df_test = pd.read_csv(
        os.path.join(unnormalized_data_dir, f"{company_slug}_test_{data_date}.csv")
    )
    df = remove_twentyeight_days_df(df, df_test.recorded_at.iloc[0])
    features = get_features_from_keywords(df.columns, feature_keywords)
    pca = PCA(n_components=n_components)
    column_names = [f"pc{i+1}" for i in range(n_components)]
    pca.fit(df.loc[:, features])
    df_pca = pd.DataFrame(pca.transform(df.loc[:, features]), columns=column_names)
    df_pca = pd.concat(
        [df_pca, df.loc[:, ["cement_slug", "recorded_at", "lab__press__cs_28d"]]],
        axis=1,
    )
    df_pca_test = pd.DataFrame(
        pca.transform(df_test.loc[:, features]), columns=column_names
    )
    df_pca_test = pd.concat(
        [
            df_pca_test,
            df_test.loc[:, ["cement_slug", "recorded_at", "lab__press__cs_28d"]],
        ],
        axis=1,
    )
    return df_pca, df_pca_test, pca


def compare_val_and_test_errors(
    companies,
    selectors,
    error_type: str = "mae",
    hyperopt_result_dir: str = "hyperopt_results",
    results_dir: str = "results",
):
    # Get hyperparameter settings
    # get results from hyperparameter results df for the parameters used in test
    # compare with test results
    dict_val = {}
    dict_test = {}
    for selector_name in selectors:
        dict_val[selector_name] = {}
        dict_test[selector_name] = {}
        hyperopt_results = create_results_df(hyperopt_result_dir, selector_name)
        results = create_results_df(results_dir, selector_name)
        for company_slug in companies:
            params = PARAMS_DICT_COMPANIES[selector_name][company_slug]
            mask = hyperopt_results.company_slug == company_slug
            for key in list(params.keys()):
                mask = mask * (hyperopt_results[key] == params[key])
            error_val = hyperopt_results[mask][error_type].values
            error_test = results[results.company_slug == company_slug][
                error_type
            ].values
            if len(error_test) >= 1:
                error_test = error_test[0]
            else:
                error_test = np.nan
            if len(error_val) >= 1:
                error_val = error_val[0]
            else:
                error_val = np.nan
            dict_val[selector_name][company_slug] = error_val
            dict_test[selector_name][company_slug] = error_test
    return dict_val, dict_test


def build_synthetic_df(
    selectors, noise_frac, nr_features, synthetic_results_dir="arnaiz_synthetic"
):
    syn_df = pd.DataFrame()
    for selector in selectors:
        df = create_results_df(synthetic_results_dir, selector)
        syn_df = pd.concat([syn_df, df], axis=0)
    syn_df["clean_frac_kept"] = 1 - syn_df["clean_frac_kicked_out"]
    syn_df["noisy_frac_kept"] = 1 - syn_df["noisy_frac_kicked_out"]
    if noise_frac:
        syn_df = syn_df[syn_df.noise_frac == noise_frac]
    if nr_features:
        syn_df = syn_df[syn_df.nr_features == nr_features]
    return syn_df


def build_drift_df(selectors, drift_results="results_drift"):
    drift_df = pd.DataFrame()
    for selector in selectors:
        df = create_results_df(drift_results, selector)
        drift_df = pd.concat([drift_df, df], axis=0)
    return drift_df
