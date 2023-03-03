import numpy as np


def convert_recorded_at_to_seconds(datetime_values):
    unix_epoch = np.datetime64(0, "s")
    one_second = np.timedelta64(1, "s")
    return (datetime_values - unix_epoch) / one_second


def transform_selector_output_into_mask(labels):
    """
    WHY THO
    Parameters
    ----------
    labels

    Returns
    -------

    """
    mask_labels = labels.copy()
    mask_labels[mask_labels == -1] = 0
    return mask_labels.astype(bool)


def normalize_array(array):
    min_value = np.min(array)
    max_value = np.max(array)
    abs_max_diff = np.abs(max_value - min_value)
    return (array - min_value) / abs_max_diff


def normalize_array_neg_plus_1(array):
    min_value = np.min(array)
    max_value = np.max(array)
    abs_max_diff = np.abs(max_value - min_value)
    return (2 * (array - min_value) / abs_max_diff) - 1


def calc_pct_increase(og_number, new_number):
    increase = new_number - og_number
    pct_increase = 100 * increase / og_number
    return pct_increase


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    return (average, np.sqrt(variance))
