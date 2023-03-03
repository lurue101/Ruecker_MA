import datetime
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import prism_raymond.config as config
import prism_raymond.database.controller.sample as csamp
from prism_kondo._constants import (
    COMPANY_FEATURES,
    MAX_DATES,
    MIN_DATES,
    PCA_FEATURE_KEYWORDS,
    STANDARD_COLUMNS,
    MAX_NAN_ROW_RATIO_28d_PROD,
)
from prism_raymond.database.db import configure_engine
from prism_raymond.database.model.company import Company
from prism_raymond.database.model.sample import Cement, CementSample, SampleType


def active_shipment_samples_to_df(company_slug):
    engine, session = configure_engine(config.POSTGRES_DATABASE_URI)
    where = (
        (Company.slug == company_slug)
        & (SampleType.slug == "shipment")
        & (CementSample.active == True)
    )
    if MIN_DATES[company_slug]:
        where = where & (CementSample.recorded_at > MIN_DATES[company_slug])
    if MAX_DATES[company_slug]:
        where = where & (CementSample.recorded_at < MAX_DATES[company_slug])
    samples = (
        session.query(CementSample)
        .join(Company)
        .join(SampleType, CementSample.sample_type_id == SampleType.id)
        .join(Cement, CementSample.cement_id == Cement.id)
        .filter(where)
        .all()
    )
    df = csamp.cement_samples_to_df(samples)
    return df


def remove_samples_with_nan_target(
    df: pd.DataFrame, target_column: str
) -> pd.DataFrame:
    mask_nan_target = np.isnan(df[target_column])
    return df.loc[~mask_nan_target, :]


def filter_cements_with_too_little_samples(
    df: pd.DataFrame, pct_threshold: float = 0.05
) -> pd.DataFrame:
    cement_value_counts = df["cement_slug"].value_counts()
    cements_above_threshold = cement_value_counts[
        cement_value_counts / len(df) > pct_threshold
    ].index.values
    return df.loc[df["cement_slug"].isin(cements_above_threshold), :]


def remove_samples_above_nan_ratio(df: pd.DataFrame, nan_ratio: float) -> pd.DataFrame:
    return df.dropna(axis=0, thresh=np.round(df.shape[1] * nan_ratio))


def remove_samples_that_miss_a_whole_method(df):
    for method in ["psd", "xrd", "xrf"]:
        cols = [col for col in df.columns if method in col]
        df = df[~(pd.isnull(df[cols]).sum(axis=1) == len(cols))]
    return df


def remove_columns_above_nan_ratio(
    df: pd.DataFrame, nan_ratio: float
) -> Tuple[pd.DataFrame, list[str]]:
    before_columns = set(df.columns)
    df_drop = df.dropna(axis=1, thresh=np.round(df.shape[0] * nan_ratio))
    return df_drop, list(set(before_columns).difference(df_drop.columns))


def filter_by_recorded_at(df, min_recorded_at=None):
    if min_recorded_at is not None:
        return df[df["recorded_at"] > min_recorded_at]
    else:
        return df


def transform_pca(df_train, df_test, features, n_components) -> pd.DataFrame:
    X_train = df_train.loc[:, features].to_numpy(dtype="float32")
    X_test = df_test.loc[:, features].to_numpy(dtype="float32")
    pca = PCA(n_components)
    pca.fit(X_train)
    X_pca_train = pca.transform(X_train)
    X_pca_test = pca.transform(X_test)
    new_feature_names = [f"pca_{i}" for i in range(X_pca_train.shape[1])]
    df_train = df_train.drop(columns=features)
    df_train.loc[:, new_feature_names] = X_pca_train
    df_test = df_test.drop(columns=features)
    df_test.loc[:, new_feature_names] = X_pca_test
    return df_train, df_test


def impute_median_by_cement(df_train, df_test, company_feature_columns):
    median_imputer = SimpleImputer(strategy="median")
    for cem_slug in df_train.cement_slug.unique():
        cem_slug_mask_train = df_train.cement_slug == cem_slug
        cem_slug_mask_test = df_test.cement_slug == cem_slug
        # If a cement doesn't exist in test set, kick it out of train set as well
        if cem_slug_mask_test.sum() == 0:
            df_train = df_train.loc[~cem_slug_mask_train, :]
            continue
        median_imputer.fit(df_train.loc[cem_slug_mask_train, company_feature_columns])
        df_train.loc[
            cem_slug_mask_train, company_feature_columns
        ] = median_imputer.transform(
            df_train.loc[cem_slug_mask_train, company_feature_columns]
        )
        df_test.loc[
            cem_slug_mask_test, company_feature_columns
        ] = median_imputer.transform(
            df_test.loc[cem_slug_mask_test, company_feature_columns]
        )
    return df_train, df_test


def set_measure_zero_if_irrelevant_for_cement(df):
    for cem_slug in df.cement_slug.unique():
        cem_slug_mask = df.cement_slug == cem_slug
        for column in [col for col in df.columns if "lab" in col]:
            # if less than 5% of samples have values in this column set all values to 0 for this cement, as the
            # measurement doesn't apply to this cement
            if pd.notnull(df.loc[cem_slug_mask, column]).sum() < 0.05 * len(
                df.loc[cem_slug_mask, column]
            ):
                df.loc[cem_slug_mask, column] = 0
    return df


def get_pca_features(df, company_slug):
    keyword = PCA_FEATURE_KEYWORDS[company_slug]
    if len(keyword) == 0:
        return None
    elif len(keyword) == 1:
        return [col for col in df.columns if keyword[0] in col]


def preprocess_data_and_save_as_csv(
    company_slug: str, target_column="lab__press__cs_28d", test_set_pct=0.15
):
    # load data
    df = active_shipment_samples_to_df(company_slug=company_slug)
    df.loc[:, "recorded_at"] = pd.to_datetime(df["recorded_at"])

    # copy to update feature last after finding NaN features
    company_feature_columns = COMPANY_FEATURES[company_slug].copy()
    # only take meta-info, feature columns, and target
    df = df.loc[:, STANDARD_COLUMNS + company_feature_columns + [target_column]]
    df = filter_by_recorded_at(df, MIN_DATES[company_slug])
    df = remove_samples_with_nan_target(df, target_column)
    # Only keep cements that make up more than 5% of total samples
    df = filter_cements_with_too_little_samples(df, 0.05)
    # remove all samples where either psd,xrd or xrf is missing entirely
    df = remove_samples_that_miss_a_whole_method(df)
    # set values of a measurement zero, if less than 5% of the samples for a specific sample have that measurement
    df = set_measure_zero_if_irrelevant_for_cement(df)
    # remove all samples that have more than a certain percentage of nan values
    df = remove_samples_above_nan_ratio(df, MAX_NAN_ROW_RATIO_28d_PROD[company_slug])
    # remove all measurements that have more than 50% nan
    df, dropped_columns = remove_columns_above_nan_ratio(df, 0.5)
    # remove those features that were dropped due to too many nan values
    company_feature_columns = [
        x for x in company_feature_columns if x not in dropped_columns
    ]
    # one-hot encode cement-slug
    cem_one_hot = pd.get_dummies(df["cement_slug"], prefix="cem", prefix_sep="__")
    # kick out meta-info
    df = df.loc[
        :, company_feature_columns + [target_column, "recorded_at", "cement_slug"]
    ]

    df = pd.concat([df, cem_one_hot], axis=1)

    # sort such that split is by date
    df = df.sort_values(by="recorded_at")
    df_train, df_test = train_test_split(df, test_size=test_set_pct, shuffle=False)

    # Median impute within each cement
    df_train, df_test = impute_median_by_cement(
        df_train, df_test, company_feature_columns
    )
    df_train_not_normalized = df_train.copy()
    df_test_not_normalized = df_test.copy()
    # fit on train data
    normalizer = StandardScaler()
    normalizer.fit(df_train.loc[:, company_feature_columns])
    # transform train data
    df_train.loc[:, company_feature_columns] = normalizer.transform(
        df_train.loc[:, company_feature_columns]
    )
    # transform test data
    df_test.loc[:, company_feature_columns] = normalizer.transform(
        df_test.loc[:, company_feature_columns]
    )
    # save processed data
    date_format = "%Y_%m_%d"
    df_train = df_train.drop(columns="cement_slug")
    df_train.to_csv(
        f"data/{company_slug}_train_{datetime.date.today().strftime(date_format)}.csv",
        index=False,
    )
    df_train_not_normalized.to_csv(
        f"data_not_normalized/{company_slug}_train_{datetime.date.today().strftime(date_format)}.csv",
        index=False,
    )
    df_test = df_test.drop(columns="cement_slug")
    df_test.to_csv(
        f"data/{company_slug}_test_{datetime.date.today().strftime(date_format)}.csv",
        index=False,
    )
    df_test_not_normalized.to_csv(
        f"data_not_normalized/{company_slug}_test_{datetime.date.today().strftime(date_format)}.csv",
        index=False,
    )


if __name__ == "__main__":
    companies = [
        "spenner",
        "rohrdorfer",
        "maerker",
        "woessingen",
        "amoeneburg",
    ]
    for company_slug in companies:
        print("#########", company_slug, "#########")
        df = preprocess_data_and_save_as_csv(company_slug)
