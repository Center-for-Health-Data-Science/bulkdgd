#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    metrics.py
#
#    This module contains utility functions for clustering metrics.
#
#    Copyright (C) 2026 Valentina Sora
#                       <sora.valentina1@gmail.com>
#
#    This program is free software: you can redistribute it and/or
#    modify it under the terms of the GNU General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public
#    License along with this program.
#    If not, see <http://www.gnu.org/licenses/>.


#######################################################################


# Set the module's description.
__doc__ = \
    "This module contains function-based utilities to compute " \
    "clustering metrics used during model training and evaluation."


#######################################################################


# Import from the standard library.
import math
from typing import Optional

# Import from third-party libraries.
import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import LabelEncoder


#######################################################################


# Define supported unsupervised metrics.
UNSUPERVISED_METRICS = {

    # The Bayesian information criterion should be minimized.
    "bic": "min",

    # The silhouette score should be maximized.
    "silhouette_score": "max",

    # The Davies-Bouldin score should be minimized.
    "davies_bouldin_score": "min",

    # The Calinski-Harabasz score should be maximized.
    "calinski_harabasz_score": "max",

}


# Define supported supervised metrics.
SUPERVISED_METRICS = {

    # The adjusted Rand index should be maximized.
    "adjusted_rand_index_score": "max",

    # The normalized mutual information should be maximized.
    "normalized_mutual_info_score": "max",

    # The adjusted mutual information score should be maximized.
    "adjusted_mutual_info_score" : "max",
    
}


#######################################################################


def _to_numpy(x: object) -> np.ndarray:
    """Convert an input object to a NumPy array.

    Parameters
    ----------
    x : :class:`object`
        The input object. It can be a torch tensor, a NumPy array,
        a list, or any object accepted by :func:`numpy.asarray`.

    Returns
    -------
    x_numpy : :class:`numpy.ndarray`
        The input converted into a NumPy array.
    """

    #

    # If the object is a tensor-like object with a ``detach`` method
    # (e.g., a :class:`torch.Tensor`), detach and move it to CPU.
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()

    # Convert (or view) the input as a NumPy array.
    return np.asarray(x)


def _validate_clustering_inputs(
        X: object,
        labels: list[str]) -> \
            tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Validate inputs used by unsupervised clustering metrics.

    Parameters
    ----------
    X : :class:`object`
        The feature matrix.

    labels : :class:`list`
        The predicted labels associated with ``X``.

    Returns
    -------
    X_valid : :class:`numpy.ndarray` or :obj:`None`
        The validated feature matrix or :obj:`None` if invalid.

    labels_valid : :class:`numpy.ndarray` or :obj:`None`
        The validated labels array or :obj:`None` if invalid.
    """

    # Convert the feature matrix to a NumPy array.
    X = _to_numpy(X)

    # Convert the labels to a NumPy array.
    labels = _to_numpy(labels)

    # The feature matrix must be 2D and labels must be 1D.
    if X.ndim != 2 or labels.ndim != 1:
        return None, None

    # The number of labels must match the number of samples and there
    # must be at least two samples.
    if X.shape[0] != labels.shape[0] or X.shape[0] < 2:
        return None, None

    # Get the unique labels.
    unique_labels = np.unique(labels)

    # At least two clusters are required, and the number of
    # clusters must be less than the number of samples.
    if unique_labels.shape[0] < 2 \
        or unique_labels.shape[0] >= X.shape[0]:
        
        # Return None if inputs are invalid.
        return None, None

    # Return validated inputs.
    return X, labels


def encode_labels(labels_lists: list[list[str]]) -> list[np.ndarray]:
    """Encode string labels as integers.

    Parameters
    ----------
    labels_lists : :class:`list`
        A list of lists of string labels.

    Returns
    -------
    encoded_labels : :class:`list`
        A list of Numpy arrays of encoded integer labels, one for each 
        input list.
    """

    # Unpack the list of lists into a single list of labels.
    labels = [label for sub_list in labels_lists for label in sub_list]

    # Create the LabelEncoder.
    encoder = LabelEncoder()

    # Fit it on all the labels.
    encoder.fit(labels)

    # For each original list of labels, transform it into encoded
    # integers.
    encoded_labels = \
        [encoder.transform(sub_list) for sub_list in labels_lists]
    
    # Return the encoded labels.
    return encoded_labels 


def get_silhouette_score(X: object,
                         labels: list[str]) -> float:
    """Get the silhouette score.

    Parameters
    ----------
    X : :class:`object`
        The feature matrix.

    labels : :class:`list`
        The predicted labels.

    Returns
    -------
    score : :class:`float`
        The silhouette score, or ``nan`` if undefined.
    """

    # Validate and normalize inputs.
    X, labels = _validate_clustering_inputs(X, labels)

    # If inputs are invalid, return NaN.
    if X is None:
        return float("nan")

    # Return the silhouette score.
    return float(silhouette_score(X, labels))


def get_davies_bouldin_score(X: object,
                             labels: list[str]) -> float:
    """Get the Davies-Bouldin score.

    Parameters
    ----------
    X : :class:`object`
        The feature matrix.

    labels : :class:`list`
        The predicted labels.

    Returns
    -------
    score : :class:`float`
        The Davies-Bouldin score, or ``nan`` if undefined.
    """

    # Validate and normalize inputs.
    X, labels = _validate_clustering_inputs(X, labels)

    # If inputs are invalid, return NaN.
    if X is None:
        return float("nan")

    # Return the Davies-Bouldin score.
    return float(davies_bouldin_score(X, labels))


def get_calinski_harabasz_score(X: object,
                                labels: list[str]) -> float:
    """Get the Calinski-Harabasz score.

    Parameters
    ----------
    X : :class:`object`
        The feature matrix.

    labels : :class:`list`
        The predicted labels.

    Returns
    -------
    score : :class:`float`
        The Calinski-Harabasz score, or ``nan`` if undefined.
    """

    # Validate and normalize inputs.
    X, labels = _validate_clustering_inputs(X, labels)

    # If inputs are invalid, return NaN.
    if X is None:
        return float("nan")

    # Return the Calinski-Harabasz score.
    return float(calinski_harabasz_score(X, labels))


def get_adjusted_rand_score(y_true: object,
                            y_pred: object) -> float:
    """Get the adjusted Rand index.

    Parameters
    ----------
    y_true : :class:`object`
        The ground-truth labels.

    y_pred : :class:`object`
        The predicted labels.

    Returns
    -------
    score : :class:`float`
        The adjusted Rand index, or ``nan`` if undefined.
    """

    # Convert the ground-truth labels.
    y_true = _to_numpy(y_true)

    # Convert the predicted labels.
    y_pred = _to_numpy(y_pred)

    # Both arrays must be one-dimensional.
    if y_true.ndim != 1 or y_pred.ndim != 1:
        return float("nan")

    # Both arrays must have the same non-zero length.
    if y_true.shape[0] != y_pred.shape[0] or y_true.shape[0] == 0:
        return float("nan")

    # Return the adjusted Rand index.
    return float(adjusted_rand_score(y_true, y_pred))


def get_normalized_mutual_info_score(y_true: object,
                                     y_pred: object) -> float:
    """Get the normalized mutual information.

    Parameters
    ----------
    y_true : :class:`object`
        The ground-truth labels.

    y_pred : :class:`object`
        The predicted labels.

    Returns
    -------
    score : :class:`float`
        The normalized mutual information, or ``nan`` if undefined.
    """

    # Convert the ground-truth labels.
    y_true = _to_numpy(y_true)

    # Convert the predicted labels.
    y_pred = _to_numpy(y_pred)

    # Both arrays must be one-dimensional.
    if y_true.ndim != 1 or y_pred.ndim != 1:
        return float("nan")

    # Both arrays must have the same non-zero length.
    if y_true.shape[0] != y_pred.shape[0] or y_true.shape[0] == 0:
        return float("nan")

    # Return the normalized mutual information.
    return float(normalized_mutual_info_score(y_true, y_pred))


def get_adjusted_mutual_info_score(y_true: object,
                                   y_pred: object) -> float:
    """Get the adjusted mutual information.

    Parameters
    ----------
    y_true : :class:`object`
        The ground-truth labels.

    y_pred : :class:`object`
        The predicted labels.

    Returns
    -------
    score : :class:`float`
        The adjusted mutual information, or ``nan`` if undefined.
    """

    # Convert the ground-truth labels.
    y_true = _to_numpy(y_true)

    # Convert the predicted labels.
    y_pred = _to_numpy(y_pred)

    # Both arrays must be one-dimensional.
    if y_true.ndim != 1 or y_pred.ndim != 1:
        return float("nan")

    # Both arrays must have the same non-zero length.
    if y_true.shape[0] != y_pred.shape[0] or y_true.shape[0] == 0:
        return float("nan")

    # Return the adjusted mutual information.
    return float(adjusted_mutual_info_score(y_true, y_pred))


def get_bic_score(gmm_model: object,
                  X: object) -> float:
    """Get the Bayesian information criterion (BIC).

    Parameters
    ----------
    gmm_model : :class:`object`
        A fitted Gaussian mixture model exposing ``lower_bound_``,
        ``covariance_type``, and ``n_components`` (or ``n_comp``).

    X : :class:`object`
        The feature matrix used to fit/evaluate the model.

    Returns
    -------
    score : :class:`float`
        The BIC score, or ``nan`` if undefined.
    """

    # A fitted model is required.
    if gmm_model is None:
        return float("nan")

    # Convert and validate the feature matrix.
    X = _to_numpy(X)
    if X.ndim != 2 or X.shape[0] < 2:
        return float("nan")

    # Extract model attributes required by the BIC formula.
    covariance_type = str(getattr(gmm_model,
                                  "covariance_type",
                                  "")).strip().lower()
    lower_bound = getattr(gmm_model, "lower_bound_", None)
    n_components = getattr(gmm_model,
                           "n_components",
                           getattr(gmm_model, "n_comp", None))

    # If required attributes are missing, return NaN.
    if lower_bound is None or n_components is None:
        return float("nan")

    # Convert to numeric values.
    n_samples, n_features = X.shape
    k = int(n_components)
    lower_bound = float(lower_bound)

    # Compute covariance parameters according to covariance type.
    if covariance_type == "full":
        cov_params = k * n_features * (n_features + 1) / 2.0
    elif covariance_type == "diag":
        cov_params = k * n_features
    elif covariance_type == "spherical":
        cov_params = k
    elif covariance_type == "tied_full":
        cov_params = n_features * (n_features + 1) / 2.0
    elif covariance_type == "tied_diag":
        cov_params = n_features
    elif covariance_type == "tied_spherical":
        cov_params = 1
    else:
        errstr = \
            "Unsupported covariance type for BIC: " \
            f"'{covariance_type}'."
        raise ValueError(errstr)

    # Means + weights.
    mean_params = n_features * k
    weight_params = k - 1
    n_parameters = cov_params + mean_params + weight_params

    # Total log-likelihood.
    log_likelihood = lower_bound * n_samples

    # Return BIC (lower is better).
    return float(n_parameters * math.log(n_samples) - \
                 2.0 * log_likelihood)


def get_metric_score(metric_name: str,
                     X: Optional[object] = None,
                     labels: Optional[list[str]] = None,
                     gmm_model: Optional[object] = None,
                     y_true: Optional[np.ndarray] = None,
                     y_pred: Optional[np.ndarray] = None) -> float:
    """Dispatch and compute a metric from its name.

    Parameters
    ----------
    metric_name : :class:`str`
        The metric name.

    X : :class:`object`, optional
        The feature matrix used by unsupervised metrics.

    labels : :class:`object`, optional
        Predicted labels used by unsupervised metrics.

    gmm_model : :class:`object`, optional
        A fitted Gaussian mixture model used by the ``"bic"`` metric.

    y_true : :class:`object`, optional
        Ground-truth labels used by supervised metrics.

    y_pred : :class:`object`, optional
        Predicted labels used by supervised metrics.

    Returns
    -------
    score : :class:`float`
        The computed metric value.
    """

    # Dispatch to BIC.
    if metric_name == "bic":
        return get_bic_score(gmm_model, X)

    # Dispatch to silhouette score.
    if metric_name == "silhouette_score":
        return get_silhouette_score(X, labels)

    # Dispatch to Davies-Bouldin score.
    if metric_name == "davies_bouldin_score":
        return get_davies_bouldin_score(X, labels)

    # Dispatch to Calinski-Harabasz score.
    if metric_name == "calinski_harabasz_score":
        return get_calinski_harabasz_score(X, labels)

    # Dispatch to adjusted Rand index.
    if metric_name == "adjusted_rand_index_score":
        return get_adjusted_rand_score(y_true, y_pred)

    # Dispatch to normalized mutual information.
    if metric_name == "normalized_mutual_info_score":
        return get_normalized_mutual_info_score(y_true, y_pred)

    # Dispatch to adjusted mutual information.
    if metric_name == "adjusted_mutual_info_score":
        return get_adjusted_mutual_info_score(y_true, y_pred)

    # Raise if the metric is unsupported.
    raise ValueError(f"Unsupported metric '{metric_name}'.")


def get_metric_optimization_direction(metric_name: str) -> str:
    """Get optimization direction for a metric.

    Parameters
    ----------
    metric_name : :class:`str`
        The metric name.

    Returns
    -------
    direction : :class:`str`
        ``"min"`` if lower values are better, ``"max"`` otherwise.
    """

    # Normalize metric name.
    metric_name = str(metric_name).strip().lower()

    # Handle unsupervised metrics.
    if metric_name in UNSUPERVISED_METRICS:
        return UNSUPERVISED_METRICS[metric_name]

    # Handle supervised metrics.
    if metric_name in SUPERVISED_METRICS:
        return SUPERVISED_METRICS[metric_name]

    # Raise if the metric is unsupported.
    raise ValueError(f"Unsupported metric '{metric_name}'.")
