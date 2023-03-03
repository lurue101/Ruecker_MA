import json
import os

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import numpy as np
import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split

from prism_kondo._constants import COMPANIES, COMPANY_ANONYMIZER, SELECTOR_NAMES_DISPLAY
from prism_kondo.analysis_utils import (
    build_drift_df,
    build_synthetic_df,
    create_labels_df,
    create_test_error_df,
    df_frac_remaining_after_IS,
    get_pca,
    get_results_dict_from_filepath,
    get_results_filepath,
)
from prism_kondo.drift_experiments import DriftExperimenter
from prism_kondo.experiment_utils import (
    calc_model_errors,
    create_results_df,
    remove_twentyeight_days,
)
from prism_kondo.experiments import ExperimentRunner
from prism_kondo.model import train_lr_model


def rename_selector_xtick_labels(ax):
    old_labels = ax.get_xticklabels()
    new_labels = [
        SELECTOR_NAMES_DISPLAY[old_label.get_text()] for old_label in old_labels
    ]
    ax.set_xticklabels(new_labels, rotation=45, ha="right", rotation_mode="anchor")


def create_heatmap_array(labels_df, selectors, pct_selector, samples_kicked_out):
    heatmap_array = np.zeros((len(selectors), len(selectors)), dtype="float64")
    idx = 0
    for selector_name in selectors:

        if samples_kicked_out == True:
            if pct_selector:
                ref_size = np.invert(labels_df[selector_name]).sum()
            else:
                ref_size = len(labels_df)
            heatmap_array[idx, :] = np.array(
                [
                    np.round(
                        np.logical_and(
                            np.invert(labels_df[selector_name]),
                            np.invert(labels_df[other_selector]),
                        ).sum()
                        / ref_size,
                        2,
                    )
                    for other_selector in selectors
                ]
            )
        else:
            if pct_selector:
                ref_size = labels_df[selector_name].sum()
            else:
                ref_size = len(labels_df)
            heatmap_array[idx, :] = np.array(
                [
                    np.round(
                        np.logical_and(
                            labels_df[selector_name], labels_df[other_selector]
                        ).sum()
                        / ref_size,
                        2,
                    )
                    for other_selector in selectors
                ]
            )
        idx += 1
    return heatmap_array


def create_heatmap(
    company_slug: str,
    selectors: list[str],
    pct_selector=True,
    samples_kicked_out: bool = True,
    results_dir: str = "results",
):
    """
    Plots a heatmap indicating what percentage of samples chosen by the algorithm on the y-axis, are also chosen by the
    algorithm on the x-axis

    Parameters
    ----------
    company_slug
        name of the company
    selectors
        list of selectors to use
    pct_selector
        True if pct in reference to the size of the set chosen by the selector on y axis, False if in reference to size
        of the whole train set
    samples_kicked_out
        True if in reference to the samples which were kicked out rather than those kept in the subset
    results_dir
        rel path to folder with results_no_one_hot

    Returns
    -------

    """
    labels_df = create_labels_df(company_slug, selectors, results_dir)
    heatmap_array = create_heatmap_array(
        labels_df, selectors, pct_selector, samples_kicked_out
    )
    selectors_display = [SELECTOR_NAMES_DISPLAY[selector] for selector in selectors]
    fig, ax = plt.subplots(figsize=(10, 7))
    heat = sns.heatmap(
        heatmap_array,
        ax=ax,
        vmin=0,
        vmax=1,
        annot=True,
        xticklabels=selectors_display,
        yticklabels=selectors_display,
        cmap="coolwarm",
        cbar_kws={"label": "Fraction of kicked out samples shared"},
    )
    heat.xaxis.tick_top()
    heat.set(title=COMPANY_ANONYMIZER[company_slug])
    plt.xticks(rotation=90)
    plt.show()


def get_metrics_succesively_taking_out_samples(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    labels: np.ndarray,
    pct_of_train_set: bool = True,
    shuffle=True,
    steps=np.linspace(0, 1, 21),
):
    """
    Returns lists of errors, indicating how big the error is after taking out a certain % of samples.
    The samples taken out are chosen from those, indicated by the labels to not be used.

    Parameters
    ----------
    X_train
        train set features
    y_train
        train set target
    X_test
        train set features
    y_test
        train set target
    labels
        boolean labels indicating which samples (index) should be kept or removed
    pct_of_train_set
        True if pct is in relation to the size of the train set, False if in relation of the samples taken out by the
        algorithm
    shuffle
        if True randomly removes samples from those, instead of using the order of the labels array ( old to new)
    steps
        list of floats indicating for which pct errors should be calculated

    Returns
    -------
    lists of errors, indicating how big the error is after taking out a certain % of samples of the train set
    """
    mae = []
    mape = []
    mse = []
    idx_to_take_out = np.argwhere(labels == False)
    if shuffle:
        idx_to_take_out = sklearn.utils.shuffle(idx_to_take_out, random_state=42)
    for pct_to_take_out in steps:
        if pct_of_train_set:
            amount_to_take_out = int(np.round(pct_to_take_out * len(labels)))
        else:
            amount_to_take_out = int(
                np.round(pct_to_take_out * np.sum(labels == False))
            )
        if amount_to_take_out > len(idx_to_take_out):
            break
        idx_to_use = np.ones_like(labels, dtype="bool")
        kicked_out_idx = idx_to_take_out[:amount_to_take_out]
        idx_to_use[kicked_out_idx] = False
        model = train_lr_model(X_train[idx_to_use], y_train[idx_to_use])
        error_dict = calc_model_errors(model, X_test, y_test)
        mae.append(error_dict["mae"])
        mape.append(error_dict["mape"])
        mse.append(error_dict["mse"])
    return mae, mse, mape


def table_full_results(results_dir):
    df = create_results_df(results_dir, "full").loc[:, ["company_slug", "r2"]]
    df["company_slug"] = df["company_slug"].apply(lambda x: COMPANY_ANONYMIZER[x])
    df = df.rename(columns={"company_slug": "Company", "r2": "$R^2"})
    return df


def get_metrics_randomly_removing_samples(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    steps: np.ndarray = np.linspace(0, 1, 21),
    random_state: int = 42,
):
    """
     Returns lists of errors, indicating how big the error is after randomly taking out a certain % of sample.

     Parameters
     ----------
    X_train
         train set features
     y_train
         train set target
     X_test
         train set features
     y_test
         train set target
     labels
         boolean labels indicating which samples (index) should be kept or removed
     steps
         list of floats indicating for which pct errors should be calculated

     Returns
     -------
    lists of error, for each step of taking out samples
    """
    mae = []
    mape = []
    mse = []
    r2 = []
    idx_to_take_out = np.arange(0, X_train.shape[0])
    idx_to_take_out = sklearn.utils.shuffle(idx_to_take_out, random_state=random_state)
    for pct_to_take_out in steps:
        if pct_to_take_out == 1:
            continue
        amount_to_take_out = int(np.round(pct_to_take_out * X_train.shape[0]))
        idx_to_use = np.ones(X_train.shape[0], dtype="bool")
        kicked_out_idx = idx_to_take_out[:amount_to_take_out]
        idx_to_use[kicked_out_idx] = False
        model = train_lr_model(X_train[idx_to_use], y_train[idx_to_use])
        error_dict = calc_model_errors(model, X_test, y_test)
        mae.append(error_dict["mae"])
        mape.append(error_dict["mape"])
        mse.append(error_dict["mse"])
        r2.append(error_dict["r2"])
    return mae, mse, mape, r2


def plot_metric_against_samples_taken_out_per_algorithm(
    company_slug: str,
    data_date: str,
    selectors: list[str],
    pct_of_train_set: bool,
    shuffle=True,
    xticks: np.ndarray = np.linspace(0, 1, 21),
    dir_results="results",
):
    """
    Creates a plot containing errors for each of the selectors, when successively taking out more samples.

    Parameters
    ----------
    company_slug
        name of the company
    data_date
        date of the prepared data set
    selectors
        list of selectors to include in the plot
    pct_of_train_set
        True if 100% should refer to all samples in the train set, False if 100 % refers to all samples marked to be
        taken out by each algorithm
    shuffle
        True if samples are taken out in random order, False if samples are taken out oldest to newest
    xticks
        steps in which the errors should be calculated and displayed
    dir_results
        dir in which results for each selector are stored

    Returns
    -------

    """
    # load data sets
    experimenter = ExperimentRunner(company_slug, data_date)
    X_train, y_train, recorded_at_train, _ = experimenter.prepare_data("train")
    X_test, y_test, recorded_at_test, _ = experimenter.prepare_data("test")
    X_train, y_train, recorded_at_train = remove_twentyeight_days(
        X_train, y_train, recorded_at_train, recorded_at_test[0]
    )
    # prepare plot figure
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xticks(xticks)
    for selector_name in selectors:
        if selector_name == "full":
            continue
        # get json for chosen company and algorithm
        filepath = get_results_filepath(company_slug, selector_name, dir_results)
        results_dict = get_results_dict_from_filepath(filepath)
        # get errors for training a model, when taking out different amount of samples
        mae, mse, mape = get_metrics_succesively_taking_out_samples(
            X_train,
            y_train,
            X_test,
            y_test,
            np.array(results_dict["labels"]),
            pct_of_train_set,
            shuffle,
            xticks,
        )
        plt.plot(xticks[: len(mape)], mape, "--o", label=selector_name, markersize=4)
    plt.plot(xticks, np.ones(len(xticks)) * mape[0], "--k", label="full_set")
    # get baseline errors when taking out random samples
    mae_rnd, mse_rnd, mape_rnd = get_metrics_randomly_removing_samples(
        X_train, y_train, X_test, y_test, xticks
    )
    plt.plot(
        xticks[:-1],
        mape_rnd,
        "--o",
        color="#808080",
        label="random selection",
        markersize=4,
    )
    plt.xticks(rotation=45)
    # TODO
    # clearer labels
    if pct_of_train_set:
        plt.xlabel("% of samples in train set taken out")
    else:
        plt.xlabel("% of samples taken out of those indicated by algorithm")
    plt.ylabel("mean absolute percentage error")
    plt.legend()


def plot_metric_against_samples_taken_out_by_scores(
    company_slug: str,
    selectors: list[str],
    ax=plt.axes,
    title: str = None,
    with_legend=False,
    cutoff_tick=0.95,
    ylim=None,
    metric="r2",
    results_dir="results_fixed_subsize",
):
    df_all = pd.DataFrame()
    for selector in selectors:
        df = create_results_df(results_dir, selector)
        df_all = pd.concat([df_all, df], axis=0)
    df_all = df_all[df_all.company_slug == company_slug]
    df_all["frac_taken_out"] = 1 - df_all["subsize_frac"]
    for selector in selectors:
        df_plot = df_all[df_all["selector_name"] == selector].sort_values(
            by="frac_taken_out"
        )
        df_plot = df_plot[df_plot["frac_taken_out"] <= cutoff_tick]
        ax.plot(df_plot["frac_taken_out"], df_plot[metric], label=selector, marker=".")
    ax.plot(
        df_plot["frac_taken_out"],
        np.ones(len(df_plot["frac_taken_out"]))
        * df_plot[df_plot["frac_taken_out"] == 0][metric].values,
        "--",
        color="black",
        label="full",
    )
    ax.set_xticks(np.linspace(0, cutoff_tick, int(np.round(cutoff_tick / 0.05, 0) + 1)))
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    if (
        metric == "r2"
        and df_all[df_all["frac_taken_out"] <= cutoff_tick][metric].min() < 0
    ):
        ax.set_ylim(0, 1)
    if ylim:
        low, high = ylim
        ax.set_ylim(low, high)
    if with_legend:
        plt.legend(bbox_to_anchor=(1.2, 1), fontsize=16)


def setup_5_plot_layout(width=15, height=10):
    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    gs = fig.add_gridspec(2, 6)
    # gs = gridspec.GridSpec(2, 6)
    gs.update(wspace=1)
    ax1 = plt.subplot(
        gs[0, :2],
    )
    ax2 = plt.subplot(gs[0, 2:4])
    ax3 = plt.subplot(gs[0, 4:6])
    ax4 = plt.subplot(gs[1, 1:3])
    ax5 = plt.subplot(gs[1, 3:5])
    axes = [ax1, ax2, ax3, ax4, ax5]
    return fig, axes


def plot_metrics_against_samples_taken_out_by_scores_all_customers(
    selectors: list[str],
    cutoff_tick=0.95,
    add_random_lines_nr=None,
    ylims=[None, None, None, None, None],
    metric="r2",
    results_dir="results_fixed_subsize",
):
    fig, axes = setup_5_plot_layout(30, 20)
    letters = ["a", "b", "c", "d", "e"]
    for i in range(len(COMPANIES)):
        with_legend = False
        if i == len(COMPANIES) - 1:
            with_legend = True
        plot_metric_against_samples_taken_out_by_scores(
            COMPANIES[i],
            selectors,
            axes[i],
            with_legend=with_legend,
            cutoff_tick=cutoff_tick,
            ylim=ylims[i],
            metric=metric,
            title=f"{letters[i]}) {COMPANY_ANONYMIZER[COMPANIES[i]]}",
            results_dir=results_dir,
        )
        if add_random_lines_nr:
            error_array = error_array_taking_out_random_samples(
                COMPANIES[i], "2023_02_12", add_random_lines_nr, metric=metric
            )
            xticks = axes[i].get_xticks()
            number_values = len(axes[i].get_xticklabels())
            axes[i].plot(
                xticks,
                error_array[0, :number_values].T,
                alpha=0.1,
                color="k",
                label="Random",
            )
            plt.legend(bbox_to_anchor=(1.2, 1), fontsize=16)
            axes[i].plot(
                xticks, error_array[1:, :number_values].T, alpha=0.1, color="k"
            )

    fig.supxlabel(
        "Fraction of samples removed from the training set", y=0.08, fontsize=16
    )
    fig.supylabel("$R^2$ of the model on the test set", x=0.08, fontsize=16)


def plot_rnd_simulations(
    company_slug: str,
    data_date: str,
    number_of_lines: int = 5,
    xticks: np.ndarray = np.linspace(0, 1, 21),
):
    # load data sets
    experimenter = ExperimentRunner(company_slug, data_date)
    X_train, y_train, recorded_at_train, _ = experimenter.prepare_data("train")
    X_test, y_test, recorded_at_test, _ = experimenter.prepare_data("test")
    X_train, y_train, recorded_at_train = remove_twentyeight_days(
        X_train, y_train, recorded_at_train, recorded_at_test[0]
    )
    # prepare plot figure
    fig, ax = plt.subplots(figsize=(20, 15))
    ax.set_xticks(xticks)
    for random_state in np.random.randint(0, 1000, number_of_lines):
        mae, mse, mape, r2 = get_metrics_randomly_removing_samples(
            X_train, y_train, X_test, y_test, xticks, random_state
        )
        plt.plot(
            xticks[:-1],
            mape,
            "--o",
            label=random_state,
            markersize=4,
        )
    plt.plot(xticks, np.ones(len(xticks)) * mape[0], "--k", label="full_set")
    plt.xticks(rotation=45)
    plt.legend()


def error_array_taking_out_random_samples(
    company_slug: str,
    data_date: str,
    n_rnd_samples: int,
    metric="r2",
    xticks: np.ndarray = np.linspace(0, 1, 21),
    random_split=False,
    random_state=42,
    keel_filename: str = None,
    keel_dir="keel",
):
    experimenter = ExperimentRunner(company_slug, data_date)
    if random_split == True:
        (
            X_train,
            X_test,
            y_train,
            y_test,
            recorded_at_train,
            recorded_at_test,
            one_hot_columns_idx,
        ) = experimenter.prepare_data_random_split(random_state)
    else:
        X_train, y_train, recorded_at_train, _ = experimenter.prepare_data("train")
        X_test, y_test, recorded_at_test, _ = experimenter.prepare_data("test")
        X_train, y_train, recorded_at_train = remove_twentyeight_days(
            X_train, y_train, recorded_at_train, recorded_at_test[0]
        )

    if keel_filename:
        file_path = os.path.join(keel_dir, keel_filename)
        df = pd.read_csv(file_path)
        X, y = df.iloc[:, :-1].to_numpy(), df.iloc[:, -1].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    errors_rnd_removal = np.empty((n_rnd_samples, len(xticks) - 1))
    random_states = np.random.randint(0, 1000, n_rnd_samples)
    for i in range(n_rnd_samples):
        mae, mse, mape, r2 = get_metrics_randomly_removing_samples(
            X_train, y_train, X_test, y_test, xticks, random_states[i]
        )
        if metric == "r2":
            errors_rnd_removal[i, :] = r2
        if metric == "mae":
            errors_rnd_removal[i, :] = mae
    return errors_rnd_removal


def plot_random_array(error_df, ax=None, title=""):
    if ax == None:
        fig, ax = plt.subplots(figsize=(15, 10))
    sns.boxplot(ax=ax, data=error_df, color="grey", saturation=0.01)
    sns.lineplot(
        ax=ax,
        x=np.arange(len(error_df.columns)),
        y=error_df.iloc[5, 0],
        linestyle="--",
        color="green",
        # label="full_set",
    )
    sns.lineplot(
        ax=ax,
        x=np.arange(len(error_df.columns)),
        y=error_df.iloc[:, :].mean(axis=0),
        linestyle="-",
        marker="o",
        # label="mean",
    )
    # plt.xticks(np.linspace(0,1,11))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title(title)
    if error_df.min().min() < 0:
        ax.set_ylim(0, 1)
    # plt.ylabel("mean absolute error")
    # plt.xlabel("% of samples from train set removed")
    # plt.legend(bbox_to_anchor=(0.0, 1.0), loc="upper left")
    # sns.lineplot(ax=ax, data=rnd_error_df.iloc[-1, :])


def plot_synthetic_metric_bars(
    selectors: list[str],
    noise_frac,
    metric="r2",
    nr_features=5,
    ax=None,
    add_figure_info=True,
    synthetic_results_dir="arnaiz_synthetic",
):
    syn_df = build_synthetic_df(
        selectors, noise_frac, nr_features, synthetic_results_dir
    )
    clean_noisy = pd.melt(
        syn_df,
        id_vars="selector",
        value_vars=[f"clean_selector_{metric}", f"noisy_selector_{metric}"],
        var_name=["clean_or_noisy"],
        value_name=metric,
    )
    if not ax:
        fig, ax = plt.subplots(figsize=(10, 7))
    graph = sns.barplot(
        data=clean_noisy,
        x="selector",
        y=metric,
        hue="clean_or_noisy",
        estimator="mean",
        width=0.6,
        errorbar="sd",
        palette=["C2", "C1"],
        ax=ax,
    )
    graph.axhline(
        syn_df[metric].mean(),
        label="Performance - trained with full clean set",
        color="green",
    )
    graph.axhline(
        syn_df[f"noisy_{metric}"].mean(),
        label="Performance - trained with full noisy set",
        color="orange",
    )
    rename_selector_xtick_labels(ax)
    letters = {0.1: "a", 0.2: "b", 0.3: "c"}
    ax.set_title(f"{letters[noise_frac]})      Noise level: {int(noise_frac * 100)}%")
    ax.get_legend().remove()
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_ylim(clean_noisy.groupby("selector").mean()[metric].min() - 0.1)
    if add_figure_info:
        rename_labels = {
            f"clean_selector_{metric}": "Performance - after applying algorithm to clean set",
            f"noisy_selector_{metric}": "Performance - after applying algorithm to noisy set",
        }
        handles, labels = graph.get_legend_handles_labels()
        labels = [rename_labels.get(l, l) for l in labels]
        ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))


def plot_synthetic_frac_kept(
    selectors,
    noise_frac,
    nr_features=5,
    ax=None,
    add_figure_info=True,
    synthetic_results_dir="arnaiz_synthetic",
):
    syn_df = build_synthetic_df(
        selectors, noise_frac, nr_features, synthetic_results_dir
    )
    frac_kept = pd.melt(
        syn_df,
        id_vars="selector",
        value_vars=["clean_frac_kept", "noisy_frac_kept"],
        var_name=["clean_or_noisy"],
        value_name="frac_kept",
    )
    if not ax:
        fig, ax = plt.subplots(figsize=(10, 7))
    graph = sns.barplot(
        data=frac_kept,
        x="selector",
        y="frac_kept",
        hue="clean_or_noisy",
        width=0.6,
        errorbar="sd",
        palette=["C2", "C1"],
        ax=ax,
    )
    graph.axhline(1, label="Full set", color="k")
    graph.axhline(
        1 - noise_frac, label="Fraction of clean data in noisy set", color="green"
    )
    rename_selector_xtick_labels(ax)
    letters = {0.1: "a", 0.2: "b", 0.3: "c"}
    ax.set_title(f"{letters[noise_frac]})      Noise level: {int(noise_frac*100)}%")
    ax.set_ylim(frac_kept.groupby("selector").mean()["frac_kept"].min() - 0.2)
    ax.get_legend().remove()
    ax.set_xlabel("")
    ax.set_ylabel("")

    if add_figure_info:
        rename_labels = {
            "clean_frac_kept": "Fraction of clean data set kept",
            "noisy_frac_kept": "Fraction of noisy data set kept",
        }
        handles, labels = graph.get_legend_handles_labels()
        labels = [rename_labels.get(l, l) for l in labels]
        ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))


def plot_synthetic_results(
    selectors,
    result_type: str,
    nr_features=5,
    synthetic_results_dir="arnaiz_synthetic",
    three_below=True,
):
    if three_below:
        fig, axes = plt.subplots(3, 1, figsize=(20, 40))
    else:
        fig = plt.figure(figsize=(15, 15), constrained_layout=True)
        gs = fig.add_gridspec(2, 8)
        # gs = gridspec.GridSpec(2, 6)
        gs.update(wspace=1)
        ax1 = plt.subplot(
            gs[0, :4],
        )
        ax2 = plt.subplot(gs[0, 4:])
        ax3 = plt.subplot(gs[1, 2:6])
        axes = np.array(
            [
                ax1,
                ax2,
                ax3,
            ]
        )
    noise_fracs = [0.1, 0.2, 0.3]
    for i in range(len(axes)):
        noise_frac = noise_fracs[i]
        figure_info = False
        if i == len(axes) - 1:
            figure_info = True
        if result_type == "frac_kept":
            plot_synthetic_frac_kept(
                selectors,
                noise_frac,
                nr_features,
                axes.flatten()[i],
                add_figure_info=figure_info,
                synthetic_results_dir=synthetic_results_dir,
            )
        elif result_type == "mae" or "r2":
            plot_synthetic_metric_bars(
                selectors,
                noise_frac,
                result_type,
                nr_features,
                ax=axes.flatten()[i],
                add_figure_info=figure_info,
                synthetic_results_dir=synthetic_results_dir,
            )
    fig.suptitle(f"Number of features {nr_features}", y=0.94, fontsize=16)
    if three_below:
        fig.supxlabel("IS Algorithm", y=0.10, fontsize=18)
    else:
        fig.supxlabel("IS Algorithm", y=0.06, fontsize=18)
    if result_type == "frac_kept":
        fig.supylabel("Fraction of the full set kept by algorithm", x=0.08, fontsize=18)
    elif result_type == "mae":
        fig.supylabel("Mean Absolute Error", x=0.08, fontsize=28)
    elif result_type == "r2":
        fig.supylabel("$R^2$", x=0.08, fontsize=28)


def plot_r2_synthetic_noise_level_in_one_plot(
    selectors,
    nr_features=5,
    synthetic_results_dir="arnaiz_synthetic",
    filename=None,
):
    fig, ax = plt.subplots(figsize=(15, 10))
    syn_df = build_synthetic_df(
        selectors,
        noise_frac=None,
        nr_features=nr_features,
        synthetic_results_dir=synthetic_results_dir,
    )
    zero_noise = pd.DataFrame(columns=syn_df.columns)
    for selector in syn_df.selector.unique():
        selectordf = pd.DataFrame(columns=syn_df.columns)
        selectordf["selector"] = np.repeat([selector], 100)
        selectordf["noise_frac"] = np.repeat([0.0], 100)
        selectordf["noisy_selector_r2"] = syn_df[
            syn_df.selector == selector
        ].clean_selector_r2.iloc[:100]
        zero_noise = pd.concat([zero_noise, selectordf])
    bar_df = pd.concat([syn_df, zero_noise])
    bar_df["selector"] = bar_df.selector.apply(
        lambda x: SELECTOR_NAMES_DISPLAY.get(x, x)
    )
    bar = sns.barplot(
        data=bar_df,
        x="selector",
        y="noisy_selector_r2",
        hue="noise_frac",
        ax=ax,
        width=0.6,
        palette=["C0", "C2", "C1", "C3"],
    )
    bar.axhline(bar_df.mean()["r2"], c="C0", linestyle="--")
    bar.axhline(
        bar_df.groupby("noise_frac").mean()["noisy_r2"].loc[0.1], c="C2", linestyle="--"
    )
    bar.axhline(
        bar_df.groupby("noise_frac").mean()["noisy_r2"].loc[0.2], c="C1", linestyle="--"
    )
    bar.axhline(
        bar_df.groupby("noise_frac").mean()["noisy_r2"].loc[0.3], c="C3", linestyle="--"
    )
    ax.set_xlabel("IS Algorithm", fontsize=16)
    ax.set_ylabel("$R^2$", fontsize=16)
    handles, labels = bar.get_legend_handles_labels()
    labels = [f"{int(float(l)*100)}%" for l in labels]
    ax.legend(
        handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), title="Noise Level"
    )
    ax.set_ylim(0.8)
    ax.set_title(f"Nr of Features: {nr_features}")
    if filename:
        fig.savefig(filename)


def plot_frac_kept_synthetic_noise_level_in_one_plot(
    selectors,
    nr_features=5,
    synthetic_results_dir="arnaiz_synthetic",
    filename=None,
):
    fig, ax = plt.subplots(figsize=(15, 10))
    syn_df = build_synthetic_df(
        selectors,
        noise_frac=None,
        nr_features=nr_features,
        synthetic_results_dir=synthetic_results_dir,
    )
    clean_frac = pd.DataFrame(columns=syn_df.columns)
    clean_frac["noisy_frac_kept"] = syn_df["clean_frac_kept"]
    clean_frac["selector"] = syn_df["selector"]
    clean_frac["noise_frac"] = 0.0
    bar_df = pd.concat([syn_df, clean_frac])
    bar_df["selector"] = bar_df["selector"].apply(
        lambda x: SELECTOR_NAMES_DISPLAY.get(x, x)
    )
    bar = sns.barplot(
        data=bar_df,
        x="selector",
        y="noisy_frac_kept",
        hue="noise_frac",
        palette=["C0", "C2", "C1", "C3"],
    )
    bar.axhline(1, c="C0", linestyle="--")
    bar.axhline(0.9, c="C2", linestyle="--")
    bar.axhline(0.8, c="C1", linestyle="--")
    bar.axhline(0.7, c="C3", linestyle="--")
    ax.set_xlabel("IS Algorithm", fontsize=16)
    ax.set_ylabel("Fraction of original training set kept by algorithm", fontsize=16)
    ax.set_title(f"Nr of Features: {nr_features}")
    handles, labels = bar.get_legend_handles_labels()
    labels = [f"{int(float(l) * 100)}%" for l in labels]
    ax.legend(
        handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), title="Noise Level"
    )
    if filename:
        fig.savefig(filename)


def plot_synth_nr_feature_diff(
    selectors, synthetic_results_dir="arnaiz_synthetic", filename=None
):

    fig, ax = plt.subplots(figsize=(15, 10))
    syn_df = build_synthetic_df(
        selectors,
        noise_frac=None,
        nr_features=None,
        synthetic_results_dir=synthetic_results_dir,
    )
    syn_df["selector"] = syn_df["selector"].apply(
        lambda x: SELECTOR_NAMES_DISPLAY.get(x, x)
    )
    syn_df = syn_df[syn_df.noise_frac == 0.3]
    syn_df = syn_df[syn_df.nr_features != 15]
    sns.barplot(
        data=syn_df, x="selector", y="noisy_selector_r2", hue="nr_features", width=0.6
    )
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("IS Algorithm", fontsize=16)
    ax.set_ylabel("$R^2$", fontsize=16)
    ax.set_title("30% Noise")
    ax.set_ylim(0.75)
    if filename:
        fig.savefig(filename)


def plot_synth_efficiency(
    selectors,
    noise_frac,
    nr_features,
    synthetic_results_dir="arnaiz_synthetic",
    filename=None,
):
    def calc_eff(grouped_df, a):
        return (
            a * grouped_df["noisy_selector_r2"]
            + (1 - a) * grouped_df["noisy_frac_kicked_out"]
        )

    fig, ax = plt.subplots(figsize=(10, 7))
    syn_df = build_synthetic_df(
        selectors,
        noise_frac=noise_frac,
        nr_features=nr_features,
        synthetic_results_dir=synthetic_results_dir,
    )
    syn_df["selector"] = syn_df["selector"].apply(
        lambda x: SELECTOR_NAMES_DISPLAY.get(x, x)
    )
    grouped_df = (
        syn_df[syn_df.noise_frac == 0.3]
        .groupby("selector")
        .mean()[["noisy_selector_r2", "noisy_frac_kicked_out"]]
    )
    # colors = sns.color_palette("tab20", len(grouped_df))
    color_counter = 0
    for i in range(len(grouped_df)):
        selector_eff = np.zeros(11)
        weights = np.linspace(0, 1, 11)
        for j in range(11):
            weight = weights[j]
            selector_eff[j] = calc_eff(grouped_df.iloc[i], weight)
        if color_counter < 10:
            ax.plot(
                weights, selector_eff, "-", linewidth=1.5, label=grouped_df.index[i]
            )
        else:
            ax.plot(
                weights, selector_eff, "--", linewidth=1.5, label=grouped_df.index[i]
            )
        # ax.plot(weights, selector_eff, "-", label=grouped_df.index[i], c=colors[i])
        color_counter += 1
    plt.title(f"{int(noise_frac*100)}% Noise, {nr_features} Input Features")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.xlabel("Performance Importance Weight $b$", fontsize=16)
    plt.ylabel("Efficiency", fontsize=16)
    if filename:
        fig.savefig(filename)


def plot_concept_drift(scores_dict, ax=None, title="", random_lines=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 10))
    current_min = 1
    color_counter = 0
    colors = sns.color_palette("tab20")
    for selector, scores in scores_dict.items():
        # ax.plot(range(50, 601, 50), scores, linewidth=1.5, label=SELECTOR_NAMES_DISPLAY[selector], c=colors[color_counter])
        if color_counter < 10:
            ax.plot(
                range(50, 601, 50),
                scores,
                linewidth=1.5,
                label=SELECTOR_NAMES_DISPLAY[selector],
            )
        else:
            ax.plot(
                range(50, 601, 50),
                scores,
                "--",
                linewidth=1.5,
                label=SELECTOR_NAMES_DISPLAY[selector],
            )

        color_counter += 1
        if np.min(scores) < current_min:
            current_min = np.min(scores)
        # sns.lineplot(x=range(50, 601, 50),y=scores,ax=ax ,linewidth=1.5, label=selector)
    ax.plot(
        range(50, 601, 50),
        np.array(random_lines).T,
        color="k",
        alpha=0.1,
        linewidth=0.5,
    )
    ax.set_title(title)
    # Otherwise when plotting "none" the lines would disappear
    if current_min < 0.9:
        ax.set_ylim(current_min - 0.1)


def plot_all_concept_drift(
    selectors, n=10, filename=None, ylims=[(None, None) for i in range(5)]
):
    drifter = DriftExperimenter()
    fig, axes = setup_5_plot_layout(15, 10)
    letters = ["a", "b", "c", "d", "e"]
    drift_type = ["none", "sudden", "gradual", "increment", "reoccurring"]
    for i in range(5):
        scores_dict = drifter.calc_mean_scores_dict(
            selectors + ["ground_truth"], drift_type[i], n=n
        )
        random_lines = drifter.generate_random_lines(drift_type[i], 100)
        title_type = drift_type[i]
        title_letter = letters[i]
        plot_concept_drift(
            scores_dict,
            axes[i],
            f"{title_letter}) Drift type: {title_type}",
            random_lines,
        )
        if ylims[i] != (None, None):
            axes[i].set_ylim(ylims[i])
    axes[-1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.supylabel("$R^2$", fontsize=18, x=0.06)
    fig.supxlabel("Number of Instances used for model training", fontsize=18, y=0.05)
    fig.suptitle(
        "Performance - Model trained with highest scoring instances", fontsize=18
    )
    if filename:
        fig.savefig(filename, dpi=300)


def plot_bars_concept_drift(
    selectors,
    drift_results_dir="results_drift",
):
    fig, axes = setup_5_plot_layout()
    letters = ["a", "b", "c", "d", "e"]
    drift_type = ["none", "sudden", "gradual", "increment", "reoccurring"]
    df = build_drift_df(selectors + ["full", "ground_truth"], drift_results_dir)
    df["selector_name"] = df["selector_name"].apply(
        lambda x: SELECTOR_NAMES_DISPLAY.get(x, x)
    )
    df["frac_kept"] = df["labels"].apply(lambda x: np.array(x).sum() / len(x))
    for i in range(5):
        df_bar = df[
            df["company_slug"] == drift_type[i]
        ]  # drift type is saved in column company_slug - dont ask why!
        full = df_bar[df_bar.selector_name == "full"]["r2"].mean()
        ground_truth = df_bar[df_bar.selector_name == "ground_truth"]["r2"].mean()
        df_bar = df_bar[~df_bar.selector_name.isin(["full", "ground_truth"])]
        bar = sns.barplot(
            data=df_bar,
            x="selector_name",
            y="r2",
            estimator="mean",
            # errorbar="sd",
            ax=axes[i],
            width=0.5,
            color="C1",
        )
        bar.axhline(full, label="Performance - Model trained with full set", color="k")
        bar.axhline(
            ground_truth,
            label="Performance - Model trained with instances from new concept",
            color="green",
        )
        # axes[i].tick_params(axis='x', rotation=45,ha='right', rotation_mode='anchor')
        labels = axes[i].get_xticklabels()
        axes[i].set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        axes[i].set_title(f"{letters[i]})     Drift type: {drift_type[i]}")
        axes[i].set_ylim(df_bar.groupby("selector_name").mean()["r2"].min() - 0.1)
    fig.supxlabel("IS Algorithm", y=0.05, fontsize=16)
    fig.supylabel("$R^2$", x=0.08, fontsize=16)
    # fig.suptitle("PCA")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))


def plot_concept_drifts_all_in_one(selectors, results_dir="results_drift"):
    df = build_drift_df(selectors, results_dir)
    df["selector_name"] = df["selector_name"].apply(
        lambda x: SELECTOR_NAMES_DISPLAY.get(x, x)
    )
    rename_dict = {
        "none": "None",
        "sudden": "Sudden",
        "gradual": "Gradual",
        "increment": "Incremental",
        "reoccurring": "Reoccurring",
    }
    df["company_slug"] = df.company_slug.apply(lambda x: rename_dict[x])
    fig, ax = plt.subplots(figsize=(15, 10))
    bar = sns.barplot(
        data=df[df.selector_name != "Full Set"],
        x="selector_name",
        y="r2",
        hue="company_slug",
        width=0.6,
        hue_order=["None", "Sudden", "Incremental", "Gradual", "Reoccurring"],
    )
    hline_df = df[df.selector_name == "Full Set"]
    bar.axhline(
        hline_df.groupby("company_slug").mean().loc["None", "r2"],
        linestyle="--",
        c="C0",
    )
    bar.axhline(
        hline_df.groupby("company_slug").mean().loc["Sudden", "r2"],
        linestyle="--",
        c="C1",
    )
    bar.axhline(
        hline_df.groupby("company_slug").mean().loc["Incremental", "r2"],
        linestyle="--",
        c="C2",
    )
    bar.axhline(
        hline_df.groupby("company_slug").mean().loc["Gradual", "r2"],
        linestyle="--",
        c="C3",
    )
    bar.axhline(
        hline_df.groupby("company_slug").mean().loc["Reoccurring", "r2"],
        linestyle="--",
        c="C4",
    )
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Drift Type")
    ax.set_xlabel("IS Algorithm", fontsize=16)
    ax.set_ylabel("$R^2$", fontsize=16)
    ax.set_ylim(0.8)
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")


def plot_pca(df_all, ax, title):
    sns.scatterplot(
        data=df_all,
        x="pc1",
        y="pc2",
        hue="set_type",
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.get_legend().remove()


def plot_pca_all_companies(
    data_date="2023_02_12",
    unnormalized_data_dir="data_not_normalized",
    filename=None,
):
    fig, axes = setup_5_plot_layout(15, 15)
    letters = ["a", "b", "c", "d", "e"]
    for i in range(5):
        company_slug = COMPANIES[i]
        ax = axes[i]
        df_pca, df_pca_test, pca = get_pca(
            company_slug,
            data_date,
            unnormalized_data_dir=unnormalized_data_dir,
            n_components=3,
        )
        df_pca["set_type"] = "Training Set"
        df_pca_test["set_type"] = "Test Set"
        df_all = pd.concat([df_pca, df_pca_test])
        title = f"{letters[i]})     {COMPANY_ANONYMIZER[company_slug]}"
        plot_pca(df_all, ax, title)
        """sns.scatterplot(
            ax=ax,
            data=df_pca,
            x="pc1",
            y="pc2",
            label="Training set",
            # hue=selector_name,
            # style="cement_slug",
            # hue="cement_slug",
        )
        sns.scatterplot(
            ax=ax,
            data=df_pca_test,
            x="pc1",
            y="pc2",
            color="green",
            size=None,
            label="Test set",
            # hue="cement_slug",
        )"""

    fig.supxlabel("Principal Component 1", y=0.08, fontsize=16)
    fig.supylabel("Principal Component 2", x=0.08, fontsize=16)
    # fig.suptitle("PCA")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=16)
    if filename:
        fig.savefig(filename, dpi=300)


def plot_pca_gradient_time(df_all, ax, title, cmap_name):
    df_all["time"] = df_all.recorded_at.apply(
        lambda x: pd.to_datetime(x).asm8.astype(float)
    )
    norm = plt.Normalize(df_all["time"].min(), df_all["time"].max())
    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
    sm.set_array([])
    scatter = sns.scatterplot(
        data=df_all,
        x="pc1",
        y="pc2",
        style="set_type",
        hue="time",
        palette=cmap_name,
        ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    scatter.get_legend().remove()
    ax.set_title(title)
    cbar = plt.colorbar(sm, ax=ax)
    ticks = pd.DatetimeIndex(
        np.linspace(
            pd.to_datetime(df_all.recorded_at).min().value,
            pd.to_datetime(df_all.recorded_at).max().value,
            len(cbar.get_ticks()),
            dtype=np.int64,
            endpoint=True,
        )
    )
    ticks = [f"{t.year}-{t.month}-{t.day}" for t in ticks]
    cbar.ax.set_yticklabels(ticks)


def plot_pca_all_companies_gradient_time(
    data_date="2023_02_12",
    unnormalized_data_dir="data_not_normalized",
):
    fig, axes = setup_5_plot_layout(30, 15)
    letters = ["a", "b", "c", "d", "e"]
    for i in range(5):
        company_slug = COMPANIES[i]
        ax = axes[i]
        df_pca, df_pca_test, pca = get_pca(
            company_slug,
            data_date,
            unnormalized_data_dir=unnormalized_data_dir,
            n_components=3,
        )
        df_pca["set_type"] = "train"
        df_pca_test["set_type"] = "test"
        df_all = pd.concat([df_pca, df_pca_test])
        title = f"{letters[i]})     {COMPANY_ANONYMIZER[company_slug]}"
        plot_pca_gradient_time(df_all, ax, title)
    fig.supxlabel("Principal Component 1", y=0.08, fontsize=16)
    fig.supylabel("Principal Component 2", x=0.08, fontsize=16)
    # fig.suptitle("PCA")
    # plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))


def plot_alcemy_bars_frac_kept(selectors, results_dir="results", filename=None):
    fig, ax = plt.subplots(figsize=(15, 10))
    frac_kept = df_frac_remaining_after_IS(COMPANIES, selectors, results_dir)
    frac_kept_bars = pd.melt(
        frac_kept.reset_index(),
        id_vars="index",
        value_vars=frac_kept.columns,
        var_name=["selector"],
        value_name="frac_kept",
    )
    frac_kept_bars["index"] = frac_kept_bars["index"].apply(
        lambda x: COMPANY_ANONYMIZER[x]
    )
    frac_kept_bars["selector"] = frac_kept_bars["selector"].apply(
        lambda x: SELECTOR_NAMES_DISPLAY[x]
    )
    sns.barplot(
        data=frac_kept_bars, x="selector", y="frac_kept", hue="index", ax=ax, width=0.6
    )
    plt.xlabel("IS Algorithm", fontsize=16)
    plt.ylabel("Fraction of the full set kept by algorithm", fontsize=16)
    ax.legend(title="", loc="center left", bbox_to_anchor=(1, 0.5))
    if filename:
        fig.savefig(filename, dpi=300)


def plot_alcemy_bars_performance(
    selectors, metric="r2", results_dir="results", filename=None
):
    df = create_test_error_df(
        COMPANIES,
        selectors,
        error_type=metric,
        pct_increase=False,
        results_dir=results_dir,
    )
    errors_bars = df.rename(
        index=COMPANY_ANONYMIZER, columns=SELECTOR_NAMES_DISPLAY
    ).T.reset_index()
    errors_bars = pd.melt(
        errors_bars,
        id_vars="index",
        value_vars=errors_bars.columns,
        var_name=["company"],
        value_name="r2_increase",
    )
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.barplot(
        data=errors_bars, x="index", y="r2_increase", hue="company", ax=ax, width=0.5
    )
    ax.axhline(0, linestyle="--", color="k", alpha=0.6)
    ax.set_ylim(-0.06, 0.06)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.xlabel("IS Algorithm", fontsize=16)
    plt.ylabel("$R^2$ Increase from Full Training Set", fontsize=16)
    if filename:
        fig.savefig(filename, dpi=300)


def plot_rnd_boxplots_alcemy(
    random_data_dir="../",
    metric="$R^2$",
    ylims=[(None, None), (None, None), (None, None), (None, None), (None, None)],
    random_split=False,
    filename=None,
):
    fig, axes = setup_5_plot_layout(15, 10)
    letters = ["a", "b", "c", "d", "e"]
    for i in range(len(COMPANIES)):
        if random_split == True:
            rnd_lines_filename = f"random_lines_{COMPANIES[i]}_random_split.csv"
        else:
            rnd_lines_filename = f"random_lines_{COMPANIES[i]}.csv"
        rnd_error_file_path = os.path.join(random_data_dir, rnd_lines_filename)
        rnd_error_df = pd.read_csv(rnd_error_file_path)
        plot_random_array(
            rnd_error_df.iloc[:, :],
            ax=axes[i],
            title=f"{letters[i]}) {COMPANY_ANONYMIZER[COMPANIES[i]]}",
        )
        if ylims[i] != (None, None):
            axes[i].set_ylim(ylims[i])
    green_line = Line2D(
        [0], [0], color="green", lw=2, linestyle="--", label=f"Full set {metric}"
    )
    blue_line = Line2D(
        [0], [0], color="blue", lw=2, linestyle="-", label=f"Mean {metric}"
    )
    axes[-1].legend(
        handles=[green_line, blue_line],
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        fontsize=12,
    )

    fig.supxlabel("Fraction of samples removed from training set", y=0.04, fontsize=16)
    fig.supylabel(
        f"{metric} of model with removed samples on test set", x=0.06, fontsize=16
    )
    if filename:
        fig.savefig(filename, dpi=300)
