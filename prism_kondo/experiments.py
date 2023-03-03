import json
import os
from datetime import datetime, timezone
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    KFold,
    ParameterGrid,
    TimeSeriesSplit,
    train_test_split,
)

from prism_kondo.experiment_utils import (
    calc_model_errors,
    create_results_df,
    get_features_from_keywords,
    remove_twentyeight_days,
    run_selector,
    save_experiment_results,
    save_hyperopt_log,
    save_hyperopt_result,
    split_df_into_arrays,
)
from prism_kondo.instance_selection._params_dict import PARAMS_DICT_COMPANIES
from prism_kondo.model import train_lr_model


class ExperimentRunner:
    def __init__(
        self,
        company_slug: str,
        file_name_date: str,
        feature_keywords: list = ["cem__", "xrd", "xrf", "perc_63", "slope"],
    ):
        self.company_slug = company_slug
        self.file_name_date = file_name_date
        self.feature_keywords = feature_keywords

    def prepare_data(
        self, set_type: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Loads the data set specified by the set_type, picks the relevant features and splits the set into features,
        target und time arrays

        Parameters
        ----------
        set_type:
            Either train or test

        Returns
        -------
        X, y, recorded_at
        """
        df = pd.read_csv(
            f"data/{self.company_slug}_{set_type}_{self.file_name_date}.csv"
        )
        df["recorded_at"] = pd.to_datetime(df["recorded_at"])
        df = df.sort_values(by="recorded_at")
        features = get_features_from_keywords(df.columns, self.feature_keywords)
        df = df.loc[:, features + ["recorded_at", "lab__press__cs_28d"]]
        (
            X,
            y,
            recorded_at,
        ) = split_df_into_arrays(df, "lab__press__cs_28d")
        one_hot_columns_idx = [
            df.columns.get_loc(col) for col in df.columns if "cem__" in col
        ]
        return X, y, recorded_at, one_hot_columns_idx

    def run_hyperopt(
        self,
        selector_name: str,
        param_dict_ranges: dict[str, np.ndarray],
        suffix: str = "",
        val_pct: float = 0.15,
        IS_without_one_hot: bool = True,
    ):
        """
        Runs a hyperopt with all possible combinations from the arrays given in param_dict_ranges. Then saves the result
        of each parameter combination as a json file. Also saves a json file containing the parameter ranges
        Parameters
        ----------
        selector_name
            name of the IS algorithm
        param_dict_ranges
            dict containing an array of options for each parameter
        suffix
            suffix that's appended to the json file name to specify any changes in the procedure
        val_pct
            float number indicating the fraction of train set, that's used for the validation set
        IS_without_one_hot
            If true the IS algorithm exludes the one-hot encoded cem slugs, If false IS uses them as well. In any case
            the trained model uses the encoded features

        Returns
        -------

        """
        X_train, y_train, recorded_at_train, one_hot_columns_idx = self.prepare_data(
            "train"
        )
        (
            X_train,
            X_val,
            y_train,
            y_val,
            recorded_at_train,
            recorded_at_val,
        ) = train_test_split(
            X_train, y_train, recorded_at_train, test_size=val_pct, shuffle=False
        )
        X_train, y_train, recorded_at_train = remove_twentyeight_days(
            X_train, y_train, recorded_at_train, recorded_at_val[0]
        )
        save_hyperopt_log(selector_name, param_dict_ranges, self.company_slug)
        all_param_combinations = list(ParameterGrid(param_dict_ranges))
        iteration = 0
        print(f"starting company {self.company_slug}")
        for param_dict in all_param_combinations:
            iteration += 1
            print(f"trying param combination {iteration}/{len(all_param_combinations)}")
            X_selector = X_train.copy()
            if IS_without_one_hot:
                X_selector = np.delete(X_train, one_hot_columns_idx, axis=1)
            if selector_name == "reg_enn_time":
                X_selector = np.hstack(
                    [X_selector, recorded_at_train.astype(np.float32)]
                )
            boolean_labels, scores = run_selector(
                X_selector, y_train, selector_name, param_dict
            )
            model = train_lr_model(X_train[boolean_labels, :], y_train[boolean_labels])
            error_dict = calc_model_errors(model, X_val, y_val)
            error_dict["features"] = self.feature_keywords
            save_hyperopt_result(
                selector_name, param_dict, error_dict, self.company_slug, suffix
            )

    def run_hyperopt_cv(
        self,
        selector_name: str,
        param_dict_ranges: dict[str, np.ndarray],
        n_splits: int = 5,
        suffix: str = "",
        IS_without_one_hot: bool = True,
    ):
        """
        Runs a hyperopt with all possible combinations from the arrays given in param_dict_ranges. Then saves the result
        of each parameter combination as a json file. Also saves a json file containing the parameter ranges
        Parameters
        ----------
        selector_name
            name of the IS algorithm
        param_dict_ranges
            dict containing an array of options for each parameter
        suffix
            suffix that's appended to the json file name to specify any changes in the procedure
        val_pct
            float number indicating the fraction of train set, that's used for the validation set
        IS_without_one_hot
            If true the IS algorithm exludes the one-hot encoded cem slugs, If false IS uses them as well. In any case
            the trained model uses the encoded features

        Returns
        -------

        """
        X, y, recorded_at_train, one_hot_columns_idx = self.prepare_data("train")
        save_hyperopt_log(
            selector_name,
            param_dict_ranges,
            self.company_slug,
            extra_info=f"cv_{n_splits}",
        )

        all_param_combinations = list(ParameterGrid(param_dict_ranges))
        iteration = 0
        for param_dict in all_param_combinations:
            iteration += 1
            print(f"trying param combination {iteration}/{len(all_param_combinations)}")
            kf = KFold(n_splits=n_splits, shuffle=False)
            cv_val_scores = []
            cv_dict = {}
            for i, (train_index, val_index) in enumerate(kf.split(X)):
                X_train = X[train_index, :]
                y_train = y[train_index]
                X_val = X[val_index, :]
                y_val = y[val_index]
                X_selector = X_train.copy()
                if IS_without_one_hot:
                    X_selector = np.delete(X_train, one_hot_columns_idx, axis=1)
                if selector_name in ["fixed_window", "reg_enn_time"]:
                    X_selector = np.hstack(
                        [X_selector, recorded_at_train[train_index].astype(np.float32)]
                    )
                if selector_name in ["fish1", "fish2"]:
                    X_selector = np.hstack(
                        [X_selector, recorded_at_train[train_index].astype(np.float32)]
                    )
                    if IS_without_one_hot:
                        x_target = np.hstack(
                            [
                                np.delete(X_val, one_hot_columns_idx, axis=1)[0, :],
                                recorded_at_train[val_index][0].astype(np.float32),
                            ]
                        )
                    else:
                        x_target = np.hstack(
                            [
                                X_val[0, :],
                                recorded_at_train[val_index][0].astype(np.float32),
                            ]
                        )
                    X_selector = np.vstack([X_selector, x_target])
                boolean_labels, scores = run_selector(
                    X_selector, y_train, selector_name, param_dict
                )
                model = train_lr_model(
                    X_train[boolean_labels, :], y_train[boolean_labels]
                )
                error_dict = calc_model_errors(model, X_val, y_val)
                cv_val_scores.append(error_dict["mae"])
                error_dict["features"] = self.feature_keywords
            cv_dict["raw_scores"] = cv_val_scores
            cv_dict["mean_score"] = np.mean(cv_val_scores)
            cv_dict["std_scores"] = np.std(cv_val_scores)
            cv_dict["n_splits"] = n_splits
            save_hyperopt_result(
                selector_name,
                param_dict,
                cv_dict,
                self.company_slug,
                suffix,
                "hyperopt_cv",
            )

    def run_hyperopt_tscv(
        self,
        selector_name: str,
        param_dict_ranges: dict[str, np.ndarray],
        n_splits: int = 5,
        suffix: str = "",
        IS_without_one_hot: bool = True,
    ):
        X, y, recorded_at_train, one_hot_columns_idx = self.prepare_data("train")
        save_hyperopt_log(
            selector_name,
            param_dict_ranges,
            self.company_slug,
            extra_info=f"cv_{n_splits}",
        )

        all_param_combinations = list(ParameterGrid(param_dict_ranges))
        iteration = 0
        for param_dict in all_param_combinations:
            iteration += 1
            print(f"trying param combination {iteration}/{len(all_param_combinations)}")
            tscv = TimeSeriesSplit(n_splits=n_splits)
            cv_val_scores = []
            cv_dict = {}
            for i, (train_index, val_index) in enumerate(tscv.split(X)):
                X_train = X[train_index, :]
                y_train = y[train_index]
                X_val = X[val_index, :]
                y_val = y[val_index]
                X_selector = X_train.copy()
                if IS_without_one_hot:
                    X_selector = np.delete(X_train, one_hot_columns_idx, axis=1)
                if selector_name in ["fixed_window", "reg_enn_time"]:
                    X_selector = np.hstack(
                        [X_selector, recorded_at_train[train_index].astype(np.float32)]
                    )
                if selector_name in ["fish1", "fish2"]:
                    X_selector = np.hstack(
                        [X_selector, recorded_at_train[train_index].astype(np.float32)]
                    )
                    if IS_without_one_hot:
                        x_target = np.hstack(
                            [
                                np.delete(X_val, one_hot_columns_idx, axis=1)[0, :],
                                recorded_at_train[val_index][0].astype(np.float32),
                            ]
                        )
                    else:
                        x_target = np.hstack(
                            [
                                X_val[0, :],
                                recorded_at_train[val_index][0].astype(np.float32),
                            ]
                        )
                    X_selector = np.vstack([X_selector, x_target])
                boolean_labels, scores = run_selector(
                    X_selector, y_train, selector_name, param_dict
                )
                model = train_lr_model(
                    X_train[boolean_labels, :], y_train[boolean_labels]
                )
                error_dict = calc_model_errors(model, X_val, y_val)
                cv_val_scores.append(error_dict["mae"])
                error_dict["features"] = self.feature_keywords
            cv_dict["raw_scores"] = cv_val_scores
            cv_dict["mean_score"] = np.mean(cv_val_scores)
            cv_dict["std_scores"] = np.std(cv_val_scores)
            cv_dict["n_splits"] = n_splits
            save_hyperopt_result(
                selector_name,
                param_dict,
                cv_dict,
                self.company_slug,
                suffix,
                "hyperopt_tscv",
            )

    def run_experiment(
        self,
        selector_name: str,
        IS_without_one_hot: bool = True,
        param_dict: dict = None,
        output_dir="results",
    ):
        """
        Runs the specified IS algorithm, trains a model with the chosen samples and tests the results. The result is
        saved in a json file

        Parameters
        ----------
        selector_name
            name of the IS algorithm to be used
        IS_without_one_hot
            If true the IS algorithm exludes the one-hot encoded cem slugs, If false IS uses them as well. In any case
            the trained model uses the encoded features

        Returns
        -------

        """
        # load train set
        X_train, y_train, recorded_at_train, one_hot_columns_idx = self.prepare_data(
            "train"
        )
        # load test set
        X_test, y_test, recorded_at_test, one_hot_columns_idx = self.prepare_data(
            "test"
        )
        X_train, y_train, recorded_at_train = remove_twentyeight_days(
            X_train, y_train, recorded_at_train, recorded_at_test[0]
        )
        if param_dict == None:
            # load best param configuration
            param_dict = PARAMS_DICT_COMPANIES[selector_name][self.company_slug]
        print("running selector:", selector_name, "\nfor company:", self.company_slug)
        # compute labels, that indicate which samples to use
        X_selector = X_train.copy()
        if selector_name in ["reg_enn_time", "fixed_window"]:
            X_selector = np.hstack([X_selector, recorded_at_train.astype(np.float32)])
        if selector_name in ["fish1", "fish"]:
            X_selector = np.hstack([X_selector, recorded_at_train.astype(np.float32)])
            x_target = np.hstack([X_test[0, :], recorded_at_test[0].astype(np.float32)])
            X_selector = np.vstack([X_selector, x_target])
        if IS_without_one_hot:
            X_selector = np.delete(X_selector, one_hot_columns_idx, axis=1)
        boolean_labels, scores = run_selector(
            X_selector, y_train, selector_name, param_dict
        )
        # train model with chosen samples
        model = train_lr_model(X_train[boolean_labels, :], y_train[boolean_labels])
        # calculate errors with trained model
        error_dict = calc_model_errors(model, X_test, y_test)
        # remove those params that are samples, as they shouldn't be saved in the json
        # save results of experiments as json
        save_experiment_results(
            selector_name,
            self.company_slug,
            boolean_labels,
            scores,
            error_dict,
            self.file_name_date,
            "_".join(self.feature_keywords),
            param_dict,
            output_dir,
        )

    def run_random_experiments(self, subsize_fractions: list):
        # load train set
        X_train, y_train, recorded_at_train, _ = self.prepare_data("train")
        # load test set
        X_test, y_test, recorded_at_test, _ = self.prepare_data("test")
        X_train, y_train, recorded_at_train = remove_twentyeight_days(
            X_train, y_train, recorded_at_train, recorded_at_test[0]
        )
        for frac in subsize_fractions:
            param_dict = {"subsize_frac": frac}
            # compute labels, that indicate which samples to use
            boolean_labels, scores = run_selector(
                X_train, y_train, "random", param_dict
            )
            # train model with chosen samples
            model = train_lr_model(X_train[boolean_labels, :], y_train[boolean_labels])
            # calculate errors with trained model
            error_dict = calc_model_errors(model, X_test, y_test)
            save_experiment_results(
                "random" + f"{frac*100}",
                self.company_slug,
                boolean_labels,
                scores,
                error_dict,
                self.file_name_date,
                "_".join(self.feature_keywords),
                param_dict,
            )

    def run_featureopt(self, val_pct=0.15):
        X_train, y_train, recorded_at_train, one_hot_columns_idx = self.prepare_data(
            "train"
        )
        (
            X_train,
            X_val,
            y_train,
            y_val,
            recorded_at_train,
            recorded_at_val,
        ) = train_test_split(
            X_train, y_train, recorded_at_train, test_size=val_pct, shuffle=False
        )
        X_train, y_train, recorded_at_train = remove_twentyeight_days(
            X_train, y_train, recorded_at_train, recorded_at_val[0]
        )
        for model_type in ["linear", "lasso_lars_cv"]:
            model = train_lr_model(X_train, y_train, model_type)
            error_dict = calc_model_errors(model, X_val, y_val)
            featureopt_dict = {
                "company": self.company_slug,
                "features": self.feature_keywords,
                "model_type": model_type,
            }
            featureopt_dict.update(error_dict)
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S.%f")
            filename = f"{self.company_slug}_featureopt_{timestamp}__{model_type}"
            full_path = os.path.join("featureopt/", filename)
            with open(full_path, "w") as outfile:
                json.dump(featureopt_dict, outfile)

    def get_pred_errors_from_labels(
        self,
        labels,
    ):
        # load train set
        X_train, y_train, recorded_at_train, one_hot_columns_idx = self.prepare_data(
            "train"
        )
        # load test set
        X_test, y_test, recorded_at_test, one_hot_columns_idx = self.prepare_data(
            "test"
        )
        X_train, y_train, recorded_at_train = remove_twentyeight_days(
            X_train, y_train, recorded_at_train, recorded_at_test[0]
        )
        model = train_lr_model(X_train[labels, :], y_train[labels])
        y_pred = model.predict(X_test)
        error = np.abs(y_pred - y_test)
        return error

    def run_experiment_fixed_subsize_frac_with_scores(
        self,
        selector_names: list[str],
        subsize_fractions: list,
        output_dir="results_fixed_subsize",
        remove_highest_score_first: bool = False,
        results_dir="results",
    ):
        # load train set
        X_train, y_train, recorded_at_train, one_hot_columns_idx = self.prepare_data(
            "train"
        )
        # load test set
        X_test, y_test, recorded_at_test, one_hot_columns_idx = self.prepare_data(
            "test"
        )
        X_train, y_train, recorded_at_train = remove_twentyeight_days(
            X_train, y_train, recorded_at_train, recorded_at_test[0]
        )
        for selector_name in selector_names:
            results_df = create_results_df(results_dir, selector_name)
            results_df = results_df[results_df.company_slug == self.company_slug]
            if len(results_df) > 1:
                raise ValueError("there should be only 1 results here")
            scores = results_df.scores.iloc[0]
            param_dict = results_df.params.iloc[0]
            """param_dict = PARAMS_DICT_CV[selector_name][self.company_slug]
            labels, scores = run_selector(X_train, y_train, selector_name, param_dict)"""
            sorted_score_idx = np.argsort(scores)
            if remove_highest_score_first:
                sorted_score_idx = np.flip(sorted_score_idx)
            for subsize_frac in subsize_fractions:
                train_idx = sorted_score_idx[
                    -int(np.round(subsize_frac * len(scores))) :
                ]
                model = train_lr_model(X_train[train_idx, :], y_train[train_idx])
                errors = calc_model_errors(model, X_test, y_test)
                save_dict = param_dict.copy()
                save_dict["subsize_frac"] = subsize_frac
                save_hyperopt_result(
                    selector_name,
                    save_dict,
                    errors,
                    self.company_slug,
                    suffix="",
                    directory_path=output_dir,
                )

    def run_experiments_fixed_subsize_selcon(
        self,
        subsize_fractions: list,
        output_dir="results_fixed_subsize",
    ):
        # load train set
        X_train, y_train, recorded_at_train, one_hot_columns_idx = self.prepare_data(
            "train"
        )
        # load test set
        X_test, y_test, recorded_at_test, one_hot_columns_idx = self.prepare_data(
            "test"
        )
        X_train, y_train, recorded_at_train = remove_twentyeight_days(
            X_train, y_train, recorded_at_train, recorded_at_test[0]
        )
        for subsize in subsize_fractions:
            param_dict = {"subsize_frac": subsize, "val_frac": 0.01}
            labels, scores = run_selector(X_train, y_train, "selcon", param_dict)
            model = train_lr_model(X_train[labels, :], y_train[labels])
            errors = calc_model_errors(model, X_test, y_test)
            save_hyperopt_result(
                "selcon",
                param_dict,
                errors,
                self.company_slug,
                suffix="",
                directory_path=output_dir,
            )

    def prepare_data_random_split(self, random_state):
        df_train = pd.read_csv(
            f"/Users/rueck/alcemy/prism_kondo/prism_kondo/data_not_normalized/{self.company_slug}_train_{self.file_name_date}.csv"
        )
        df_test = pd.read_csv(
            f"/Users/rueck/alcemy/prism_kondo/prism_kondo/data_not_normalized/{self.company_slug}_test_{self.file_name_date}.csv"
        )
        df = pd.concat([df_train, df_test], axis=0)
        df["recorded_at"] = pd.to_datetime(df["recorded_at"])
        features = get_features_from_keywords(df.columns, self.feature_keywords)
        df = df.loc[:, features + ["recorded_at", "lab__press__cs_28d"]]
        (
            X,
            y,
            recorded_at,
        ) = split_df_into_arrays(df, "lab__press__cs_28d")
        one_hot_columns_idx = [
            df.columns.get_loc(col) for col in df.columns if "cem__" in col
        ]
        (
            X_train,
            X_test,
            y_train,
            y_test,
            recorded_at_train,
            recorded_at_test,
        ) = train_test_split(
            X, y, recorded_at, test_size=0.15, shuffle=True, random_state=random_state
        )
        return (
            X_train,
            X_test,
            y_train,
            y_test,
            recorded_at_train,
            recorded_at_test,
            one_hot_columns_idx,
        )

    def run_experiment_random_split(
        self,
        selector_name: str,
        random_state=42,
        IS_without_one_hot: bool = True,
        param_dict: dict = None,
        output_dir="results_random_split",
    ):
        (
            X_train,
            X_test,
            y_train,
            y_test,
            recorded_at_train,
            recorded_at_test,
            one_hot_columns_idx,
        ) = self.prepare_data_random_split(random_state)
        if param_dict == None:
            # load best param configuration
            param_dict = PARAMS_DICT_COMPANIES[selector_name][self.company_slug]
        print("running selector:", selector_name, "\nfor company:", self.company_slug)
        # compute labels, that indicate which samples to use
        X_selector = X_train.copy()
        if selector_name in ["reg_enn_time", "fixed_window"]:
            X_selector = np.hstack([X_selector, recorded_at_train.astype(np.float32)])
        if selector_name in ["fish1", "fish"]:
            X_selector = np.hstack([X_selector, recorded_at_train.astype(np.float32)])
            x_target = np.hstack([X_test[0, :], recorded_at_test[0].astype(np.float32)])
            X_selector = np.vstack([X_selector, x_target])
        if IS_without_one_hot:
            X_selector = np.delete(X_selector, one_hot_columns_idx, axis=1)
        boolean_labels, scores = run_selector(
            X_selector, y_train, selector_name, param_dict
        )
        # train model with chosen samples
        model = train_lr_model(X_train[boolean_labels, :], y_train[boolean_labels])
        # calculate errors with trained model
        error_dict = calc_model_errors(model, X_test, y_test)
        error_dict["random_state"] = random_state
        # remove those params that are samples, as they shouldn't be saved in the json
        # save results of experiments as json
        save_experiment_results(
            selector_name,
            self.company_slug,
            boolean_labels,
            scores,
            error_dict,
            self.file_name_date,
            "_".join(self.feature_keywords),
            param_dict,
            output_dir,
        )
