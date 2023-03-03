import datetime
import json
import os
from datetime import timezone

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, ParameterGrid, train_test_split

from prism_kondo.experiment_utils import (
    add_random_noise_arnaiz,
    calc_model_errors,
    run_selector,
    save_hyperopt_result,
    train_lr_model,
)
from prism_kondo.instance_selection._params_dict import (
    NOISE_DEPENDENT_PARAMS,
    PARAMS_DICTS_NOISE,
)


class NoiseExperimenter:
    def generate_gaussian_linear_data(
        self,
        nr_samples: int,
        nr_features: int,
        mean: float,
        std: float,
        random_state=None,
    ):
        rs = np.random.RandomState(random_state)
        X = rs.normal(mean, std, size=(nr_samples, nr_features))
        y = np.zeros(nr_samples)
        coefs = np.round(rs.uniform(-10, 10, nr_features), 2)
        for i in range(nr_features):
            y += coefs[i] * X[:, i]
        y += rs.normal(0, 1, size=nr_samples)
        return X, y

    def run_experiments_arnaiz(
        self,
        selectors,
        nr_datasets,
        nr_samples,
        nr_features,
        mean,
        std,
        noise_frac: float,
        output_dir="arnaiz_synthetic",
    ):
        for i in range(nr_datasets):
            if i % 20 == 0:
                print(f"generated {i}/{nr_datasets} datasets")
            X, y = self.generate_gaussian_linear_data(
                nr_samples, nr_features, mean, std
            )
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
            X_val, X_test = X_test[:200], X_test[200:]
            y_val, y_test = y_test[:200], y_test[200:]
            y_train_noisy, noisy_idx = add_random_noise_arnaiz(
                y_train, noise_frac=noise_frac
            )
            for selector_name in selectors:
                if selector_name in ["selcon", "shapley", "fish1"]:
                    params = NOISE_DEPENDENT_PARAMS[noise_frac][selector_name]
                elif selector_name == "ground_truth":
                    params = {}
                else:
                    params = PARAMS_DICTS_NOISE[selector_name]
                model_clean = train_lr_model(X_train, y_train)
                errors_clean = calc_model_errors(model_clean, X_test, y_test)
                model_noisy = train_lr_model(X_train, y_train_noisy)
                errors_noisy = calc_model_errors(model_noisy, X_test, y_test)
                if selector_name == "ground_truth":
                    labels_clean = np.ones(len(X_train), dtype="bool")
                    labels_noisy = np.ones(len(X_train), dtype="bool")
                    labels_noisy[noisy_idx] = False
                elif selector_name in ["selcon"]:
                    X_train_and_val = np.vstack([X_train, X_val])
                    y_train_and_val = np.concatenate([y_train, y_val])
                    y_train_noisy_and_val = np.concatenate([y_train_noisy, y_val])
                    labels_clean, _ = run_selector(
                        X_train_and_val, y_train_and_val, selector_name, params
                    )
                    labels_clean = labels_clean[:600]
                    labels_noisy, _ = run_selector(
                        X_train_and_val, y_train_noisy_and_val, selector_name, params
                    )
                    labels_noisy = labels_noisy[:600]
                elif selector_name in ["reg_enn_time", "fixed_window", "fish1"]:
                    base_time = datetime.datetime(2000, 1, 1)
                    time_train = np.array(
                        [
                            base_time + datetime.timedelta(days=i)
                            for i in range(X_train.shape[0])
                        ],
                        dtype="datetime64[ns]",
                    ).astype(np.float32)
                    X_time = np.hstack([X_train, time_train.reshape(-1, 1)])
                    if selector_name in ["reg_enn_time", "fixed_window"]:
                        labels_clean, _ = run_selector(
                            X_time, y_train, selector_name, params
                        )
                        labels_noisy, _ = run_selector(
                            X_time, y_train_noisy, selector_name, params
                        )
                    elif selector_name in ["fish1"]:
                        x_target = np.hstack(
                            [X_test[0, :], time_train[-1].astype(np.float32)]
                        )
                        X_fish = np.vstack([X_time, x_target])
                        labels_clean, _ = run_selector(
                            X_fish, y_train, selector_name, params
                        )
                        labels_noisy, _ = run_selector(
                            X_fish, y_train_noisy, selector_name, params
                        )

                else:
                    labels_clean, _ = run_selector(
                        X_train, y_train, selector_name, params
                    )
                    labels_noisy, _ = run_selector(
                        X_train, y_train_noisy, selector_name, params
                    )
                model_labels_clean = train_lr_model(
                    X_train[labels_clean, :], y_train[labels_clean]
                )
                model_labels_noisy = train_lr_model(
                    X_train[labels_noisy, :], y_train_noisy[labels_noisy]
                )
                errors_clean_selector = calc_model_errors(
                    model_labels_clean, X_test, y_test
                )
                errors_noisy_selector = calc_model_errors(
                    model_labels_noisy, X_test, y_test
                )
                errors_clean.update(
                    {f"clean_selector_{k}": v for k, v in errors_clean_selector.items()}
                )
                errors_clean.update({f"noisy_{k}": v for k, v in errors_noisy.items()})
                errors_clean.update(
                    {f"noisy_selector_{k}": v for k, v in errors_noisy_selector.items()}
                )
                errors_clean["selector"] = selector_name
                if selector_name == "ground_truth":
                    errors_clean["params"] = {}
                else:
                    errors_clean["params"] = PARAMS_DICT[selector_name]
                errors_clean["noise_frac"] = noise_frac
                errors_clean["mean"] = mean
                errors_clean["std"] = std
                errors_clean["nr_samples"] = nr_samples
                errors_clean["nr_features"] = nr_features
                errors_clean["std_y"] = float(np.std(y_test))
                (
                    correctly_kicked_out,
                    falsely_kicked_out,
                ) = self.calc_correctly_identified_noise_samples(
                    noisy_idx, labels_noisy
                )
                errors_clean["clean_frac_kicked_out"] = len(
                    np.argwhere(labels_clean == False).flatten()
                ) / len(labels_clean)
                errors_clean["noisy_frac_kicked_out"] = len(
                    np.argwhere(labels_noisy == False).flatten()
                ) / len(labels_noisy)
                errors_clean["frac_correctly_kicked_out"] = correctly_kicked_out / len(
                    noisy_idx
                )
                errors_clean["frac_falsely_kicked_out"] = falsely_kicked_out / len(
                    y_train
                )
                self.save_json_file(
                    errors_clean,
                    output_dir,
                    selector_name,
                )

    def save_json_file(self, info_dict, output_dir, selector_name):
        directory_path = os.path.join(output_dir, selector_name)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        timestamp = datetime.datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S.%f")
        filename = f"{timestamp}"
        filepath = os.path.join(directory_path, filename)
        with open(filepath, "w") as outfile:
            json.dump(info_dict, outfile)

    def run_hyperopt(
        self,
        selector_name,
        param_dict_ranges,
        nr_samples,
        nr_features,
        mean,
        std,
        noise_frac,
        n_splits=10,
    ):
        X, y = self.generate_gaussian_linear_data(nr_samples, nr_features, mean, std)
        X, _, y, _ = train_test_split(X, y, test_size=0.25)
        y_train_noisy, noisy_idx = add_random_noise_arnaiz(y, noise_frac=noise_frac)
        base_time = datetime.datetime(2000, 1, 1)
        time_train = np.array(
            [base_time + datetime.timedelta(days=i) for i in range(X.shape[0])],
            dtype="datetime64[ns]",
        ).astype(np.float32)
        X_time = np.hstack([X, time_train.reshape(-1, 1)])
        all_param_combinations = list(ParameterGrid(param_dict_ranges))
        iteration = 1
        for param_dict in all_param_combinations:
            print("trying combination", iteration, "/", len(all_param_combinations))
            iteration += 1
            kf = KFold(n_splits=n_splits, shuffle=False)
            cv_val_scores = []
            cv_dict = {}

            for i, (train_index, val_index) in enumerate(kf.split(X)):
                X_train = X[train_index, :]
                y_train = y_train_noisy[train_index]
                X_val = X[val_index, :]
                y_val = y_train_noisy[val_index]
                if selector_name == "reg_enn_time":
                    boolean_labels, scores = run_selector(
                        X_time[train_index, :], y_train, selector_name, param_dict
                    )
                elif selector_name == "fish1":
                    x_target = np.hstack(
                        [X_val[0, :], time_train[-1].astype(np.float32)]
                    )
                    X_fish = np.vstack([X_time[train_index, :], x_target])
                    boolean_labels, scores = run_selector(
                        X_fish, y_train, selector_name, param_dict
                    )
                else:
                    boolean_labels, scores = run_selector(
                        X_train, y_train, selector_name, param_dict
                    )
                model = train_lr_model(
                    X_train[boolean_labels, :], y_train[boolean_labels]
                )
                error_dict = calc_model_errors(model, X_val, y_val)
                cv_val_scores.append(error_dict["r2"])
            cv_dict["raw_scores"] = cv_val_scores
            cv_dict["mean_score"] = np.mean(cv_val_scores)
            cv_dict["std_scores"] = np.std(cv_val_scores)
            cv_dict["n_splits"] = n_splits
            cv_dict["noise_frac"] = noise_frac
            cv_dict["mean"] = mean
            cv_dict["std"] = std
            cv_dict["nr_samples"] = nr_samples
            cv_dict["nr_features"] = nr_features
            save_hyperopt_result(
                selector_name,
                param_dict,
                cv_dict,
                f"noise{noise_frac}",
                "",
                "hyperopt_synthetic",
            )

    def calc_correctly_identified_noise_samples(self, noisy_idx, labels):
        idx_by_selector = np.argwhere(labels == False).flatten()
        correctly_kicked_out = len(
            set(noisy_idx).intersection(set(np.argwhere(labels == False).flatten()))
        )
        falsely_kicked_out = len(idx_by_selector) - correctly_kicked_out
        return correctly_kicked_out, falsely_kicked_out

    def create_pca_plot(self, noise_frac=0.3, filename=None, random_state=None):
        X, y = self.generate_gaussian_linear_data(
            1000, 5, 0, 1, random_state=random_state
        )
        y_noisy, noisy_idx = add_random_noise_arnaiz(y, noise_frac=noise_frac)
        pca = PCA(2)
        X = pca.fit_transform(X)
        # fig, axes = plt.subplots(1, 2, figsize=(15, 10), sharey=True)
        gridspec = {"width_ratios": [1, 1, 0.05]}
        fig, axes = plt.subplots(1, 3, figsize=(15, 10), gridspec_kw=gridspec)
        axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap="seismic")
        axes[0].set_title("a)     Clean Data Set")
        size = np.ones(len(y)) * 20
        size[noisy_idx] = 60
        axes[1].scatter(X[:, 0], X[:, 1], c=y_noisy, s=size, cmap="seismic")
        axes[1].set_title(f"b)     Noisy Data Set ( {int(noise_frac*100)}% )")

        fig.supxlabel("Principal Component 1", fontsize=16, y=0.0)
        fig.supylabel("Principal Component 2", fontsize=16, x=0.08)
        cmap = mpl.cm.get_cmap("seismic")
        norm = mpl.colors.Normalize(vmin=y.min(), vmax=y.max())
        cbar = plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=axes[2],
        )
        cbar.set_label(label="y Value", size="large", weight="bold")
        if filename:
            fig.savefig(filename)
