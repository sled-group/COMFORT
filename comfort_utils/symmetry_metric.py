from copy import deepcopy
import numpy as np


def normalize_data(data, epsilon=1e-6):
    """
    Normalize the likelihoods of a token in the data.

    Args:
    data: list of datapoints. Each datapoint is a tuple of the form (angle, {token1:likelihood1, token2:likelihood2})
    """
    tokens = data[0][1].keys()
    normalize_data = deepcopy(data)
    for token in tokens:
        max_likelihood = max([dp[1][token] for dp in data])
        min_likelihood = min([dp[1][token] for dp in data])
        likelihood_range = max_likelihood - min_likelihood
        if likelihood_range == 0:
            likelihood_range = epsilon
        for i in range(len(data)):
            normalize_data[i][1][token] = (data[i][1][token] - min_likelihood) / likelihood_range
    return normalize_data


def spatial_symmetry_metric(data, token="Yes", normalize=True):
    """
    Calculate the spatial symmetry metric for a given token in the data.

    Spatial symmetry metric is calculated as the sum of the squared differences between the likelihoods
    of the token at each angle and its symmetric opposite angle.

    Args:
    data: list of datapoints. Each datapoint is a tuple of the form (angle, {token1:likelihood1, token2:likelihood2})
    token: the token for which the spatial symmetry metric is to be calculated
    normalize: whether to normalize the likelihoods of the token in the data
    """
    # an entry in the datapoint is of the form (angle, {token1:likelihood1, token2:likelihood2})
    if normalize:
        data = normalize_data(data)
    if len(data) == 37:
        data = data[:-1]

    n = len(data) # 36
    assert n % 2 == 0, "data length should be even"
    middle_index = n // 2 # 18
    symmetry_score = 0

    for i in range(1, middle_index): # 1, 2, 3, ..., 17
        opposite_index = n - i # (1, 35), (2, 34), (3, 33), ..., (17, 19)
        difference = data[i][1][token] - data[opposite_index][1][token]
        symmetry_score += difference**2

    return np.sqrt(symmetry_score / (len(data) - 2) * 2)


def reverse_relation_symmetry_metric(
    positive_data, negative_data, token="Yes", normalize=True, shift_angle=0
):
    """
    Calculate the reverse relation symmetry metric for a given token in the data.

    Reverse relation symmetry metric is calculated as the root sum of the squared differences between the
    likelihoods of the token in the positive relation and its symmetric opposite angle in the negative relation.

    Args:
    positive_data: list of datapoints for the positive relation. Each datapoint is a tuple of the form (angle, {token1:likelihood1, token2:likelihood2})
    negative_data: list of datapoints for the negative relation. Each datapoint is a tuple of the form (angle, {token1:likelihood1, token2:likelihood2})
    token: the token for which the reverse relation symmetry metric is to be calculated
    normalize: whether to normalize the likelihoods of the token in the data
    shift_angle: (degree) the angle by which to shift the negative relation to find the symmetric opposite angle
    """
    # an entry in the datapoint is of the form (angle, {token1:likelihood1, token2:likelihood2})
    symmetry_score = 0

    if normalize:
        positive_data = normalize_data(positive_data)
        negative_data = normalize_data(negative_data)
    assert len(positive_data) == 37, f"Data length should be 37, got: {len(positive_data)}"
    assert len(negative_data) == 37, f"Data length should be 37, got: {len(negative_data)}"
    
    if len(positive_data) == 37:
        positive_data = positive_data[:-1]
    if len(negative_data) == 37:
        negative_data = negative_data[:-1]
    
    assert len(positive_data) % 2 == 0, "data length should be even"
    assert len(negative_data) % 2 == 0, "data length should be even"

    for i in range(len(positive_data)):
        assert (
            shift_angle % 10 == 0
        ), f"shift_angle should be a multiple of 10, got: {shift_angle}"
        i_shifted = ((i * 10 + shift_angle) % 360) // 10
        difference = (
            positive_data[i][1][token] + negative_data[i_shifted][1][token] - 1
        )
        symmetry_score += difference**2

    return np.sqrt(symmetry_score / (len(positive_data)))


def eval_symmetry_metric(positive_data, negative_data):
    print(
        "spatial symmetry metric for yes: ",
        spatial_symmetry_metric(positive_data, token="Yes"),
    )
    print(
        "reverse relation symmetry metric for yes: ",
        reverse_relation_symmetry_metric(positive_data, negative_data, token="Yes"),
    )
    print(
        "spatial symmetry metric for no: ",
        spatial_symmetry_metric(negative_data, token="No"),
    )
    print(
        "reverse relation symmetry metric for no: ",
        reverse_relation_symmetry_metric(positive_data, negative_data, token="No"),
    )
    # return spatial_symmetry_metric(positive_data, token="Yes"), reverse_relation_symmetry_metric(positive_data, negative_data, token="Yes"), spatial_symmetry_metric(negative_data, token="No"), reverse_relation_symmetry_metric(positive_data, negative_data, token="No")
