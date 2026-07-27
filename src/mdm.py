import os

# ============================================================
# IMPORTANT: set R_HOME BEFORE importing rpy2
# ============================================================

os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources"


import itertools
import numpy as np
import torch

import rpy2.robjects as ro
from rpy2.robjects.packages import importr


# Load mice
mice = importr("mice")



# ============================================================
# Generate all unique MAR-compatible patterns
# ============================================================

def generate_unique_patterns(d):
    """
    Generate all patterns where the last variable is observed.

    1 = observed
    0 = missing

    Number of patterns:
        2^(d-1)
    """

    patterns = np.array(
        list(itertools.product([0, 1], repeat=d-1)),
        dtype=np.int32
    )

    patterns = np.column_stack(
        [
            patterns,
            np.ones(patterns.shape[0], dtype=np.int32)
        ]
    )

    return patterns



# ============================================================
# Sample missingness pattern for each row
# ============================================================

def sample_patterns(n, d, pattern_prob, seed=None):

    rng = np.random.default_rng(seed)

    patterns = generate_unique_patterns(d)

    pattern_prob = np.asarray(pattern_prob)

    assert len(pattern_prob) == len(patterns)

    assert np.isclose(
        pattern_prob.sum(),
        1
    )


    idx = rng.choice(
        len(patterns),
        size=n,
        replace=True,
        p=pattern_prob
    )


    sampled_patterns = patterns[idx]


    # rows that stay fully observed
    all_ones = np.all(
        sampled_patterns == 1,
        axis=1
    )


    return sampled_patterns, all_ones



# ============================================================
# numpy -> R conversions
# ============================================================

def numpy_to_r_dataframe(X):

    """
    Convert numpy matrix to R dataframe.
    """

    cols = {}

    for j in range(X.shape[1]):

        cols[f"X{j+1}"] = ro.FloatVector(
            X[:, j]
        )

    return ro.DataFrame(cols)



def numpy_to_r_matrix(X):

    """
    Convert numpy matrix to R matrix.

    R uses column-major ordering.
    """

    return ro.r.matrix(
        ro.IntVector(
            X.T.flatten()
        ),
        nrow=X.shape[0],
        ncol=X.shape[1]
    )



# ============================================================
# Main function
# ============================================================

def ampute_mar(
        X,
        pattern_prob,
        seed=None,
        prop=0.99
):
    """
    Generate MAR missingness using mice::ampute.

    Parameters
    ----------
    X:
        torch.Tensor (n,d)

    pattern_prob:
        probability of each unique pattern

    prop:
        proportion of incomplete rows amputated


    Returns
    -------
    X_NA:
        torch.Tensor with NaN

    M:
        mask tensor
        1 = observed
        0 = missing
    """


    # ----------------------------
    # torch -> numpy
    # ----------------------------

    device = X.device

    X_np = (
        X.detach()
        .cpu()
        .numpy()
        .astype(float)
    )


    n, d = X_np.shape



    # ----------------------------
    # sample patterns
    # ----------------------------

    sampled_patterns, all_ones = sample_patterns(
        n=n,
        d=d,
        pattern_prob=pattern_prob,
        seed=seed
    )


    # only incomplete rows go to ampute
    X_inc = X_np[~all_ones]

    patterns_inc = (
        sampled_patterns[~all_ones]
    )



    # ----------------------------
    # unique patterns + frequencies
    # ----------------------------

    unique_patterns, counts = np.unique(
        patterns_inc,
        axis=0,
        return_counts=True
    )


    # MAR requires at least one observed variable
    if np.any(
        unique_patterns.sum(axis=1) == 0
    ):
        raise ValueError(
            "MAR pattern cannot be all zeros"
        )


    freq = counts / counts.sum()



    # ----------------------------
    # numpy -> R
    # ----------------------------

    r_X = numpy_to_r_dataframe(
        X_inc
    )


    r_patterns = numpy_to_r_matrix(
        unique_patterns
    )


    r_freq = ro.FloatVector(
        freq
    )



    # ----------------------------
    # mice::ampute
    # ----------------------------

    result = mice.ampute(
        r_X,
        patterns=r_patterns,
        freq=r_freq,
        prop=prop,
        mech="MAR",
        bycases=True
    )



    # ----------------------------
    # R -> numpy
    # ----------------------------

    X_inc_NA = np.array(
        result.rx2("amp")
    ).T


    assert X_inc_NA.shape == X_inc.shape



    # ----------------------------
    # reconstruct full dataset
    # ----------------------------

    X_NA_np = X_np.copy()


    X_NA_np[~all_ones] = X_inc_NA



    # mask
    M_np = (
        ~np.isnan(X_NA_np)
    ).astype(np.float32)



    # ----------------------------
    # numpy -> torch
    # ----------------------------

    X_NA = torch.tensor(
        X_NA_np,
        dtype=torch.float32,
        device=device
    )


    M = torch.tensor(
        M_np,
        dtype=torch.float32,
        device=device
    )


    return X_NA, M