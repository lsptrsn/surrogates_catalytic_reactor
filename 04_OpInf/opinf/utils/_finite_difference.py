#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: peterson
"""

"""Finite-difference schemes for estimating snapshot time derivatives."""

__all__ = [
    "ddt_uniform",
    "ddt_nonuniform",
]

import numpy as np
from scipy.signal import savgol_filter

# Finite difference stencils ==================================================
def _fwd4(y, dt):
    """Compute the first derivative of a uniformly-spaced-in-time array with a
    fourth-order forward difference scheme.

    Parameters
    ----------
    y : (5, ...) ndarray
        Data to differentiate. The derivative is taken along the first axis.
    dt : float
        Time step (the uniform spacing).

    Returns
    -------
    dy0 : float or (...) ndarray
        Approximate derivative of y at the first entry, i.e., dy[0] / dt.
    """
    return (-25*y[0] + 48*y[1] - 36*y[2] + 16*y[3] - 3*y[4]) / (12*dt)


def _fwd6(y, dt):
    """Compute the first derivative of a uniformly-spaced-in-time array with a
    sixth-order forward difference scheme.

    Parameters
    ----------
    y : (7, ...) ndarray
        Data to differentiate. The derivative is taken along the first axis.
    dt : float
        Time step (the uniform spacing).

    Returns
    -------
    dy0 : float or (...) ndarray
        Approximate derivative of y at the first entry, i.e., dy[0] / dt.
    """
    return (- 147*y[0] + 360*y[1] - 450*y[2]
            + 400*y[3] - 225*y[4] + 72*y[5] - 10*y[6]) / (60*dt)


def apply_savitzky_golay(states, window_length=11, polyorder=3):
    """Apply a Savitzky-Golay smoothing filter to the input states.

    Parameters
    ----------
    states : (n, k) ndarray
        Input data to be smoothed.
    window_length : int, optional
        Length of the filter window. Default is 5.
    polyorder : int, optional
        Polynomial order of the smoothing filter. Default is 2.

    Returns
    -------
    smoothed_states : (n, k) ndarray
        Smoothed data after applying the Savitzky-Golay filter.
    """
    smoothed_states = savgol_filter(states, window_length, polyorder, axis=1)
    return smoothed_states



def ddt_uniform(states, dt, order=4, smoothing_method='savgol'):
    """Approximate the time derivatives for a chunk of snapshots that are
    uniformly spaced in time.

    Parameters
    ----------
    states : (n, k) ndarray
        States to estimate the derivative of. The jth column is a snapshot
        that corresponds to the jth time step, i.e., states[:, j] = x(t[j]).
    dt : float
        The time step between the snapshots, i.e., t[j+1] - t[j] = dt.
    order : int {2, 4, 6}
        The order of the derivative approximation.
        See https://en.wikipedia.org/wiki/Finite_difference_coefficient.

    Returns
    -------
    ddts : (n, k) ndarray
        Approximate time derivative of the snapshot data. The jth column is
        the derivative dx / dt corresponding to the jth snapshot, states[:, j].
    """
    # Check dimensions and input types.
    if states.ndim != 2:
        raise ValueError("states must be two-dimensional")
    if not np.isscalar(dt):
        dt = dt.item()
    if not np.isscalar(dt):
        raise TypeError("time step dt must be a scalar (e.g., float)")

    if smoothing_method == "savgol":
        states = apply_savitzky_golay(states)

    if order == 2:
        return np.gradient(states, dt, edge_order=2, axis=1)

    Q = states
    ddts = np.empty_like(states)
    n, k = states.shape
    if order == 4:
        # Central difference on interior.
        ddts[:, 2:-2] = (Q[:, :-4]
                         - 8*Q[:, 1:-3] + 8*Q[:, 3:-1]
                         - Q[:, 4:])/(12*dt)

        # Forward / backward differences on the front / end.
        # TODO: don't use fully forward / fully backward for interior points.
        for j in range(2):
            ddts[:, j] = _fwd4(Q[:, j:j+5].T, dt)                 # Forward
            ddts[:, -j-1] = -_fwd4(Q[:, -j-5:k-j].T[::-1], dt)    # Backward

    elif order == 6:
        # Central difference on interior.
        ddts[:, 3:-3] = (- Q[:, :-6] + 9*Q[:, 1:-5]
                         - 45*Q[:, 2:-4] + 45*Q[:, 4:-2]
                         - 9*Q[:, 5:-1] + Q[:, 6:]) / (60*dt)

        # TODO: don't use fully forward / fully backward for interior points.
        # Forward / backward differences on the front / end.
        for j in range(3):
            ddts[:, j] = _fwd6(Q[:, j:j+7].T, dt)                 # Forward
            ddts[:, -j-1] = -_fwd6(Q[:, -j-7:k-j].T[::-1], dt)    # Backward

    else:
        raise NotImplementedError(f"invalid order '{order}'; "
                                  "valid options: {2, 4, 6}")

    return ddts


def ddt_nonuniform(states, t):
    """Approximate the time derivatives for a chunk of snapshots with a
    second-order finite difference scheme.

    Parameters
    ----------
    states : (n, k) ndarray
        States to estimate the derivative of. The jth column is a snapshot
        that corresponds to the jth time step, i.e., states[:, j] = x(t[j]).
    t : (k,) ndarray
        The times corresponding to the snapshots. May not be uniformly spaced.
        See ddt_uniform() for higher-order computation in the case of
        evenly-spaced-in-time snapshots.

    Returns
    -------
    ddts : (n, k) ndarray
        Approximate time derivative of the snapshot data. The jth column is
        the derivative dx / dt corresponding to the jth snapshot, states[:, j].
    """
    # Check dimensions.
    if states.ndim != 2:
        raise ValueError("states must be two-dimensional")
    if t.ndim != 1:
        raise ValueError("time t must be one-dimensional")
    if states.shape[-1] != t.shape[0]:
        raise ValueError("states not aligned with time t")

    # Compute the derivative with a second-order difference scheme.
    return np.gradient(states, t, edge_order=2, axis=-1)
