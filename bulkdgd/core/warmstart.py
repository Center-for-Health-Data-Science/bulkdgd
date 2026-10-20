#!/usr/bin/env python

#    warmstart.py
#
#    A data-driven starting point for the representation search.
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
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#    General Public License for more details.
#
#    You should have received a copy of the GNU General Public
#    License along with this program. If not, see
#    <http://www.gnu.org/licenses/>.

"""A ridge predictor from counts to representations, used as one of
the starting candidates of the representation search.
"""


# Import from the standard library.
import logging as log
import os

# Import from third-party libraries.
import numpy as np
import pandas as pd
import torch


# Get the module's logger.
logger = log.getLogger(__name__)


class RidgeWarmStart:

    """Class implementing a ridge predictor of a sample's
    representation from its counts.

    It is fitted on a model's training representations and applies to
    counts on the same genes.
    """

    def __init__(self,
                 weights,
                 mean,
                 scale,
                 lam,
                 genes):
        """Initialize the predictor from its fitted state.

        Parameters
        ----------
        weights : :class:`torch.Tensor`
            The primal ridge weights, of shape (n_genes, n_dim).

        mean : :class:`torch.Tensor`
            The per-gene mean of the training features.

        scale : :class:`torch.Tensor`
            The per-gene standard deviation of the training features.

        lam : :class:`float`
            The ridge parameter.

        genes : :class:`list`
            The genes, in the order the weights expect them.
        """

        # Set the weights.
        self.weights = weights

        # Set the per-gene mean and scale.
        self.mean = mean
        self.scale = scale

        # Set the ridge parameter.
        self.lam = lam

        # Set the genes.
        self.genes = list(genes)

    #-----------------------------------------------------------------#

    @staticmethod
    def _featurise(counts,
                   mean = None,
                   scale = None):
        """Scale each sample by its median count, apply log1p, and
        standardise each gene.

        Parameters
        ----------
        counts : :class:`numpy.ndarray` or :class:`torch.Tensor`
            The counts (samples x genes).

        mean : :class:`torch.Tensor`, optional
            The per-gene mean to standardise with. If not given, it is
            computed from ``counts``.

        scale : :class:`torch.Tensor`, optional
            The per-gene standard deviation to standardise with. If
            not given, it is computed from ``counts``.

        Returns
        -------
        x : :class:`torch.Tensor`
            The standardised features.

        mean : :class:`torch.Tensor`
            The per-gene mean used.

        scale : :class:`torch.Tensor`
            The per-gene standard deviation used.
        """

        # Convert the counts to a double-precision tensor.
        x = torch.as_tensor(counts, dtype = torch.float64)

        # Divide each sample by its median count (at least 1), then
        # take log1p.
        scal = x.median(dim = 1, keepdim = True).values.clamp(min = 1.0)
        x = torch.log1p(x / scal)

        # If no mean was given, compute it.
        if mean is None:
            mean = x.mean(0)

        # If no scale was given, compute it.
        if scale is None:
            scale = x.std(0).clamp(min = 1.0e-6)

        # Return the standardised features, the mean, and the scale.
        return (x - mean) / scale, mean, scale

    #-----------------------------------------------------------------#

    @classmethod
    def fit(cls,
            counts,
            representations,
            genes,
            lambdas = (1.0e3, 1.0e4, 3.0e4),
            n_val = 500,
            seed = 0):
        """Fit the predictor, choosing the ridge parameter on a random
        held-out split.

        Parameters
        ----------
        counts : :class:`numpy.ndarray`
            The training counts (samples x genes).

        representations : :class:`numpy.ndarray`
            The training representations (samples x latent
            dimensions).

        genes : :class:`list`
            The genes, in the order of the columns of ``counts``.

        lambdas : :class:`tuple`, optional
            The ridge parameters to try.

        n_val : :class:`int`, optional
            The number of held-out samples (at least 2, at most a
            quarter of all samples).

        seed : :class:`int`, optional
            The seed for the random split.

        Returns
        -------
        ws : :class:`RidgeWarmStart`
            The fitted predictor.
        """

        # Get the features and the targets.
        X, mean, scale = cls._featurise(counts)
        Y = torch.as_tensor(representations, dtype = torch.float64)

        # Randomly permute the samples.
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(X), generator = g)

        # Get the number of held-out samples.
        n_val = min(n_val, len(X) // 4)

        # If fewer than two samples can be held out
        if n_val < 2:

            # Raise an error.
            errstr = \
                "The ridge warm start needs at least 8 samples and " \
                "an 'n_val' of at least 2, but got " \
                f"{len(X)} sample(s), leaving {n_val} held out."
            raise ValueError(errstr)

        # Split the samples into validation and training sets.
        va, tr = perm[:n_val], perm[n_val:]

        # Initialize the best ridge parameter and score.
        best_lam, best_r2 = None, -np.inf

        # For each ridge parameter
        for lam in lambdas:

            # Solve the dual ridge problem on the training samples.
            G = X[tr] @ X[tr].T
            A = G + lam * torch.eye(len(tr), dtype = torch.float64)
            alpha = torch.linalg.solve(A, Y[tr])

            # Predict the held-out representations.
            pred = (X[va] @ X[tr].T) @ alpha

            # Get the R^2 on the held-out samples.
            ss_res = ((pred - Y[va]) ** 2).sum()
            ss_tot = ((Y[va] - Y[va].mean(0)) ** 2).sum()
            r2 = float(1.0 - ss_res / ss_tot)

            # Inform the user about the score.
            logger.info(
                f"The ridge warm start scored R^2 = {r2:.4f} on the "
                f"held-out samples with lambda = {lam:g}.")

            # If the score is the best so far, keep it.
            if r2 > best_r2:
                best_lam, best_r2 = lam, r2

        # Refit on all samples with the chosen lambda.
        G = X @ X.T
        A = G + best_lam * torch.eye(len(X), dtype = torch.float64)
        alpha = torch.linalg.solve(A, Y)

        # Collapse the dual solution into primal weights.
        weights = X.T @ alpha

        # Inform the user about the fit.
        logger.info(
            f"The ridge warm start was fitted with lambda = "
            f"{best_lam:g}, which explained {100*best_r2:.1f}% of the "
            f"variance of the held-out representations.")

        # Return the fitted predictor.
        return cls(weights = weights,
                   mean = mean,
                   scale = scale,
                   lam = best_lam,
                   genes = genes)

    #-----------------------------------------------------------------#

    def predict(self,
                counts,
                device = None):
        """Predict the representation of each row of ``counts``.

        Parameters
        ----------
        counts : :class:`numpy.ndarray` or :class:`torch.Tensor`
            The counts (samples x genes), on the predictor's genes.

        device : :class:`str` or :class:`torch.device`, optional
            The device to move the predictions to.

        Returns
        -------
        z : :class:`torch.Tensor`
            The predicted representations.
        """

        # Get the features.
        X, _, _ = self._featurise(counts,
                                  mean = self.mean,
                                  scale = self.scale)

        # Get the predicted representations.
        z = X @ self.weights

        # Return them, on the requested device.
        return z.to(device) if device is not None else z

    #-----------------------------------------------------------------#

    def save(self,
             path):
        """Save the fitted state to a file.

        Parameters
        ----------
        path : :class:`str`
            The output file.
        """

        # Save the fitted state.
        torch.save({"weights": self.weights.cpu(),
                    "mean": self.mean.cpu(),
                    "scale": self.scale.cpu(),
                    "lam": self.lam,
                    "genes": self.genes},
                   path)

        # Inform the user that the predictor was saved.
        logger.info(f"The ridge warm start was saved in '{path}'.")

    #-----------------------------------------------------------------#

    @classmethod
    def from_file(cls,
                  path):
        """Load a fitted predictor from a file.

        Parameters
        ----------
        path : :class:`str`
            The file written by :meth:`save`.

        Returns
        -------
        ws : :class:`RidgeWarmStart`
            The fitted predictor.
        """

        # Load the fitted state.
        d = torch.load(path,
                       map_location = "cpu",
                       weights_only = False)

        # Return the predictor.
        return cls(weights = d["weights"],
                   mean = d["mean"],
                   scale = d["scale"],
                   lam = d["lam"],
                   genes = d["genes"])


#---------------------------------------------------------------------#


def fit_from_model_dir(model_dir,
                       counts_file,
                       genes,
                       output_file = None):
    """Fit a warm start from a model's own training representations.

    Parameters
    ----------
    model_dir : :class:`str`
        A trained model's directory, holding
        ``representations_train`` as Parquet or CSV.

    counts_file : :class:`str`
        The CSV file of the counts the model was trained on.

    genes : :class:`list`
        The genes, in the order the model expects them.

    output_file : :class:`str`, optional
        The file to save the fitted predictor to.

    Returns
    -------
    ws : :class:`RidgeWarmStart`
        The fitted predictor.
    """

    # Get the path stem of the training representations' file.
    _stem = os.path.join(model_dir, "representations_train")

    # Get the first existing file among the Parquet and CSV formats.
    _path = next((f"{_stem}{e}" for e in (".parquet", ".pq", ".csv")
                  if os.path.isfile(f"{_stem}{e}")), None)

    # If no file was found, raise an error.
    if _path is None:
        raise FileNotFoundError(
            f"no 'representations_train.{{parquet,csv}}' in "
            f"'{model_dir}'.")

    # Load the representations.
    reps = (pd.read_parquet(_path)
            if _path.endswith((".parquet", ".pq"))
            else pd.read_csv(_path, index_col = 0))

    # Keep only the latent dimensions.
    reps = reps[[c for c in reps.columns
                 if c.startswith("latent_dim_")]]

    # Load the counts.
    counts = pd.read_csv(counts_file, index_col = 0)

    # Get the samples shared by the representations and the counts.
    common = reps.index.intersection(counts.index)

    # If no sample is shared, raise an error.
    if len(common) == 0:
        raise ValueError(
            "No sample is shared between the model's training "
            "representations and the counts file.")

    # Inform the user about the samples used.
    logger.info(
        f"The ridge warm start will be fitted on {len(common)} "
        f"sample(s) shared between the model's training "
        f"representations and '{counts_file}'.")

    # Fit the predictor.
    ws = RidgeWarmStart.fit(
        counts = counts.loc[common, genes].to_numpy(),
        representations = reps.loc[common].to_numpy(),
        genes = genes)

    # If an output file was given, save the predictor.
    if output_file is not None:
        ws.save(output_file)

    # Return the predictor.
    return ws
