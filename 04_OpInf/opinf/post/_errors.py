# post/_errors.py
"""Tools for accuracy and error evaluation."""
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la

from ..utils import (
    plot_PDE_dynamics,
    plot_PDE_dynamics_2D,
    plot_compare_PDE_data
                         )

__all__ = [
            "frobenius_error",
            "run_postprocessing",
            "lp_error"
          ]


def _absolute_and_relative_error(Qtrue, Qapprox, norm):
    """Compute the absolute and relative errors between Qtrue and Qapprox,
    where Qapprox approximates Qtrue:

        absolute_error = ||Qtrue - Qapprox||,
        relative_error = ||Qtrue - Qapprox|| / ||Qtrue||
                       = absolute_error / ||Qtrue||,

    with ||Q|| defined by norm(Q).
    """
    norm_of_data = norm(Qtrue)
    absolute_error = norm(Qtrue - Qapprox)
    return absolute_error, absolute_error / norm_of_data


def frobenius_error(Qtrue, Qapprox):
    """Compute the absolute and relative Frobenius-norm errors between the
    snapshot sets Qtrue and Qapprox, where Qapprox approximates Qtrue:

        absolute_error = ||Qtrue - Qapprox||_F,
        relative_error = ||Qtrue - Qapprox||_F / ||Qtrue||_F.

    Parameters
    ----------
    Qtrue : (n, k)
        "True" data. Each column is one snapshot, i.e., Qtrue[:, j] is the data
        at some time t[j].
    Qapprox : (n, k)
        An approximation to Qtrue, i.e., Qapprox[:, j] approximates Qtrue[:, j]
        and corresponds to some time t[j].

    Returns
    -------
    abs_err : float
        Absolute error ||Qtrue - Qapprox||_F.
    rel_err : float
        Relative error ||Qtrue - Qapprox||_F / ||Qtrue||_F.
    """
    # Check dimensions.
    if Qtrue.shape != Qapprox.shape:
        raise ValueError("Qtrue and Qapprox not aligned")
    if Qtrue.ndim != 2:
        raise ValueError("Qtrue and Qapprox must be two-dimensional")

    # Compute the errors.
    return _absolute_and_relative_error(Qtrue, Qapprox,
                                        lambda Z: la.norm(Z, ord="fro"))


def run_postprocessing(Q_true, sol, parameters, X_test, T_test, z_all, t,
                       r_X, r_T, X_all, T_all, k=''):
    z = z_all[1:]
    X_pred = sol[:X_test.shape[0], :]
    T_pred = sol[T_test.shape[0]:, :]
    abs_froerr, rel_froerr = frobenius_error(Qtrue=Q_true,
                                             Qapprox=sol)
    print(f"Relative Frobenius-norm error: {rel_froerr:.4%}")
    if parameters.plot_results is True:
        ## graphics
        # conversion
        plot_compare_PDE_data(X_test, X_pred, z, t,
                              "conversion-OpInf with rank:"+str(r_X),
                              function_name='X')
        X_0 = X_all[0, :].reshape(1, -1)
        X_pred_merge = np.vstack((X_0, X_pred))
        plot_PDE_dynamics_2D(z_all, t, X_all, X_pred_merge,
                             ['conversion_'+str(k), 'conversion in [-] - truth',
                              'conversion in [-] - OpInf-rank:'+str(r_X),
                              'conversion in [-] - residual'],
                             function_name='conversion X in [-]')
        plot_PDE_dynamics(z_all, t, X_all, X_pred_merge,
                             ['conversion_'+str(k), 'conversion in [-] - truth',
                              'conversion in [-] - OpInf-rank:'+str(r_X),
                              'conversion in [-] - residual'],
                             function_name='conversion X in [-]')
        # temperature
        plot_compare_PDE_data(T_test, T_pred, z, t,
                              "temperature-OpInf with rank:"+str(r_T),
                              function_name='temperature T in K')
        T_0 = T_all[0, :].reshape(1, -1)
        T_pred_merge = np.vstack((T_0, T_pred))
        plot_PDE_dynamics_2D(z_all, t, T_all, T_pred_merge,
                             ['temperature_'+str(k), 'temperature in K - truth',
                              'temperature in K - OpInf-rank:'+str(r_T),
                              'temperature in K - residual'],
                             function_name='temperature T in K')
        plot_PDE_dynamics(z_all, t, T_all, T_pred_merge,
                             ['temperature_'+str(k), 'temperature in K - truth',
                              'temperature in K - OpInf-rank:'+str(r_T),
                              'temperature in K - residual'],
                             function_name='temperature T in K')
    return rel_froerr


def lp_error(t, Qtrue, Qapprox, p=2, normalize=False):
    """Compute the absolute and relative lp-norm errors between the snapshot
    sets Qtrue and Qapprox, where Qapprox approximates to Qtrue:

        absolute_error_j = ||Qtrue_j - Qapprox_j||_p,
        relative_error_j = ||Qtrue_j - Qapprox_j||_p / ||Qtrue_j||_p.

    Parameters
    ----------
    t: (n,) ndayarray
        An array corresponding to the time
    Qtrue : (n, k) or (n,) ndarray
        "True" data. Each column is one snapshot, i.e., Qtrue[:, j] is the data
        at some time t[j]. If one-dimensional, all of Qtrue is a single
        snapshot.
    Qapprox : (n, k) or (n,) ndarray
        An approximation to Qtrue, i.e., Qapprox[:, j] approximates Qtrue[:, j]
        and corresponds to some time t[j]. If one-dimensional, all of Qapprox
        is a single snapshot approximation.
    p : float
        Order of the lp norm (default p=2 is the Euclidean norm). Used as
        the `ord` argument for scipy.linalg.norm(); see options at
        docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.norm.html.
    normalize : bool
        If true, compute the normalized absolute error instead of the relative
        error, defined by

            normalized_absolute_error_j
                = ||Qtrue_j - Qapprox_j||_2 / max_k{||Qtrue_k||_2}.

    Returns
    -------
    abs_err : (k,) ndarray or float
        Absolute error of each pair of snapshots Qtrue[:, j] and Qapprox[:, j].
        If Qtrue and Qapprox are one-dimensional, Qtrue and Qapprox are treated
        as single snapshots, so the error is a float.
    rel_err : (k,) ndarray or float
        Relative or normed absolute error of each pair of snapshots Qtrue[:, j]
        and Qapprox[:, j]. If Qtrue and Qapprox are one-dimensional, Qtrue and
        Qapprox are treated as single snapshots, so the error is a float.
    """
    # Check p.
    if not np.isscalar(p) or p <= 0:
        raise ValueError("norm order p must be positive (np.inf ok)")

    # Check dimensions.
    if Qtrue.shape != Qapprox.shape:
        raise ValueError("Qtrue and Qapprox not aligned")
    if Qtrue.ndim not in (1, 2):
        raise ValueError("Qtrue and Qapprox must be one- or two-dimensional")

    # Compute the error.
    norm_of_data = la.norm(Qtrue, ord=p, axis=0)
    if normalize:
        norm_of_data = norm_of_data.max()
    absolute_error = la.norm(Qtrue - Qapprox, ord=p, axis=0)

    plt.semilogy(t, absolute_error/ norm_of_data)
    plt.title(r"Relative $\ell^{2}$ error over time")
    plt.xlabel('time in s')
    plt.ylabel('relative $\ell^{2}$ error ')
    plt.show()
    return absolute_error, absolute_error / norm_of_data
