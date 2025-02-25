"""Tools for basis computation and reduced-dimension selection."""

__all__ = [
    "pod",
    "polynomial_form",
    "get_basis_and_reduced_data",
    "basis_multi",
    "basis_nonlin_multi",
    "svdval_decay",
    "cumulative_energy",
    "svd_results",
    "residual_energy"
]

from joblib import Parallel, delayed  # Import Joblib for parallel processing
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.optimize as opt
import sklearn.utils.extmath as sklmath

import opinf.parameters
import opinf.post
Params = opinf.parameters.Params()  # call parameters from dataclass

###############################################################################
# COLORS
###############################################################################
# Define your colors
mpi_colors = {
    'mpi_blue': (51/255, 165/255, 195/255),
    'mpi_red': (120/255, 0/255, 75/255),
    'mpi_green': (0/255, 118/255, 117/255),
    'mpi_grey': (135/255, 135/255, 141/255),
    'mpi_beige': (236/255, 233/255, 212/255)
}


def pod(
    states,
    r: int = "full",
    mode: str = "dense",
    return_W: bool = False,
    **options,
):
    """Compute the POD basis of rank r corresponding to the states.

    Parameters
    ----------
    states : (n, k) ndarray
        Matrix of k snapshots. Each column is a single snapshot of dimension n.
    r : int or "full"
        Number of POD basis vectors and singular values to compute.
        If "full" (default), compute the full SVD.
    mode : str
        Strategy to use for computing the truncated SVD of the states. Options:

        * "dense" (default): Use scipy.linalg.svd() to compute the SVD.
            May be inefficient for very large matrices.
        * "randomized": Compute an approximate SVD with a randomized approach
            using sklearn.utils.extmath.randomized_svd(). This gives faster
            results at the cost of some accuracy.

    return_W : bool
        If True, also return the first r *right* singular vectors.
    options
        Additional parameters for the SVD solver, which depends on `mode`:

        * "dense": scipy.linalg.svd()
        * "randomized": sklearn.utils.extmath.randomized_svd()

    Returns
    -------
    V : (n, r) ndarray
        First r POD basis vectors (left singular vectors).
        Each column is a single basis vector of dimension n.
    svdvals : (n,), (k,), or (r,) ndarray
        Singular values in descending order. Always returns as many as are
        calculated: r for mode="randomize", min(n, k) for "dense".
    W : (k, r) ndarray
        First r **right** singular vectors, as columns.
        **Only returned if return_W=True.**
    """
    # Validate the rank.
    rmax = min(states.shape)
    if r == "full":
        r = rmax
    if r > rmax or r < 1:
        raise ValueError(f"invalid POD rank r = {r} (need 1 ≤ r ≤ {rmax})")

    if mode == "dense" or mode == "simple":
        V, svdvals, Wt = la.svd(states, full_matrices=False, **options)
        W = Wt.T

    elif mode == "randomized":
        if "random_state" not in options:
            options["random_state"] = None
        V, svdvals, Wt = sklmath.randomized_svd(states, r, **options)
        W = Wt.T

    else:
        raise NotImplementedError(f"invalid mode '{mode}'")

    if return_W:
        return V[:, :r], svdvals, W[:, :r]

    return V[:, :r], svdvals


def polynomial_form(x, p=3):
    """Generate polynomial terms up to degree p for the input array x."""
    return [x**degree for degree in range(2, p + 1)]


def _relative_error(S_exact, S_reconstructed, reference_states):
    """Calculate the relative squared Frobenius-norm error."""
    error_norm = np.linalg.norm(S_exact - S_reconstructed, 'fro')
    ref_norm = np.linalg.norm(S_exact - reference_states, 'fro')
    return error_norm / ref_norm


def _linear_pod_basis(V_X, V_T, Q_shifted, reference_states):
    """Perform linear POD basis reduction and compute projection error."""
    r_X = Params.r_X
    r_T = Params.r_T
    # Compute reduced basis using a multi-variable approach
    V_reduced = basis_multi(V_X, r_X, V_T, r_T)
    # Project the shifted dataset onto the reduced basis
    Q_reduced = opinf.utils.reduced_state(Q_shifted, V_reduced)
    # Reconstruct the full state from the reduced representation
    states_reproduced_POD = reference_states + V_reduced @ Q_reduced
    Q_true = Q_shifted + reference_states

    abs_froerr, rel_froerr = opinf.post.frobenius_error(Qtrue=Q_true,
                                                        Qapprox=states_reproduced_POD)

    print("Traditional POD reconstruction error: "
          f"{rel_froerr:.4%}")

    return Q_reduced, V_reduced, None, None


def _nonlinear_pod_basis(V_X, V_T, Q_shifted, reference_states):
    """Perform nonlinear POD with polynomial expansion for the reduced states."""
    # Step 1: Linear POD for initial guess
    Q_reduced, V_reduced, _, _ = _linear_pod_basis(V_X, V_T,
                                                   Q_shifted,
                                                   reference_states)
    # Prepare parameters for the nonlinear part
    r_X = Params.r_X
    r_T = Params.r_T
    q_X, q_T, p, gamma = 100, 100, 3, 0
    # Compute nonlinear reduced basis
    V_reduced_nonlin = basis_nonlin_multi(V_X, r_X, q_X, V_T, r_T, q_T)
    # Compute projection error after linear POD
    proj_error = Q_shifted - (V_reduced @ Q_reduced)
    # Polynomial expansion of the reduced states
    poly = np.concatenate(polynomial_form(Q_reduced, p), axis=0)
    # Solve for the nonlinear correction coefficients
    rhs = np.linalg.inv(poly @ poly.T + gamma
                        * np.identity((p - 1) * (r_X + r_T)))
    Xi = V_reduced_nonlin.T @ proj_error @ poly.T @ rhs
    # Reconstruct the state using nonlinear POD
    states_reproduced_NLPOD = reference_states + V_reduced @ Q_reduced \
        + V_reduced_nonlin @ Xi @ poly
    Q_true = Q_shifted + reference_states

    abs_froerr, rel_froerr = opinf.post.frobenius_error(Qtrue=Q_true,
                                                        Qapprox=states_reproduced_NLPOD)

    print("Nonlinear POD Reconstruction error: "
          f"{rel_froerr:.4%}")

    return Q_reduced, V_reduced, V_reduced_nonlin, Xi


def _nonlinear_basis(V_X, V_T, Q_shifted, reference_states):
    """Perform alternating minimization-based nonlinear manifold learning."""

    def _representation_learning_obj(x, snapshot):
        """Objective function for alternating minimization."""
        poly_terms = np.concatenate(polynomial_form(x), axis=0)
        residual = Q_true[:, snapshot] - reference_states.flatten() \
            - V_reduced @ x - V_reduced_nonlin @ Xi @ poly_terms
        return np.linalg.norm(residual) ** 2

    # Step 1: Linear POD as the initial guess
    Q_reduced, V_reduced, _, _ = _linear_pod_basis(V_X, V_T,
                                                   Q_shifted,
                                                   reference_states)
    # Prepare parameters for the nonlinear part
    Q_true = Q_shifted + reference_states
    X_shifted = Q_shifted[:int(Q_shifted.shape[0]/2), :]
    T_shifted = Q_shifted[int(Q_shifted.shape[0]/2):, :]

    # Prepare parameters
    r_X, r_T = Params.r_X, Params.r_T
    q_X, q_T, p, gamma = 100, 100, 3, 0
    max_iter, tol = 100, 1e-10
    num_snapshots = Q_shifted.shape[1]

    # Compute nonlinear basis and projection error
    V_reduced_nonlin_X = V_X[:, r_X:r_X + q_X]
    V_reduced_nonlin_T = V_T[:, r_T:r_T + q_T]
    proj_error = Q_shifted - (V_reduced @ Q_reduced)

    # Polynomial expansion for reduced states
    poly = np.concatenate(polynomial_form(Q_reduced), axis=0)
    poly_X = np.concatenate(polynomial_form(Q_reduced[:r_X, :], p), axis=0)
    poly_T = np.concatenate(polynomial_form(Q_reduced[r_X:, :], p), axis=0)

    # Solve for initial coefficients
    rhs_X = np.linalg.inv(poly_X @ poly_X.T
                          + gamma * np.identity((p - 1) * r_X))
    rhs_T = np.linalg.inv(poly_T @ poly_T.T
                          + gamma * np.identity((p - 1) * r_T))
    proj_error_X = proj_error[:int(Q_true.shape[0]/2), :]
    proj_error_T = proj_error[int(Q_true.shape[0]/2):, :]
    Xi_X = V_reduced_nonlin_X.T @ proj_error_X @ poly_X.T @ rhs_X
    Xi_T = V_reduced_nonlin_T.T @ proj_error_T @ poly_T.T @ rhs_T

    # Alternating minimization loop
    nrg_old = 0
    print('Starting alternating minimization')
    for niter in range(max_iter):
        # Step 1: Update basis vectors using orthogonal Procrustes
        Um_X, _, Vm_X = np.linalg.svd(X_shifted @ np.concatenate([Q_reduced[:r_X, :], Xi_X @ poly_X]).T, full_matrices=False)
        Um_T, _, Vm_T = np.linalg.svd(T_shifted @ np.concatenate([Q_reduced[r_X:, :], Xi_T @ poly_T]).T, full_matrices=False)
        Omega_X, Omega_T = Um_X @ Vm_X, Um_T @ Vm_T
        V_reduced = opinf.basis.basis_multi(Omega_X, r_X, Omega_T, r_T)
        V_reduced_nonlin = opinf.basis.basis_nonlin_multi(Omega_X, r_X, q_X,
                                                             Omega_T, r_T, q_T)

        # Step 2: Update nonlinear coefficients
        proj_error = Q_shifted - (V_reduced @ Q_reduced)
        rhs = np.linalg.inv(poly @ poly.T + gamma
                            * np.identity((p - 1) * (r_X + r_T)))
        Xi = V_reduced_nonlin.T @ proj_error @ poly.T @ rhs

        # Step 3: Update reduced states using nonlinear regression
        Q_reduced = np.array([opt.minimize(_representation_learning_obj,
                                           Q_reduced[:, snapshot],
                                           args=(snapshot,),
                                           method='L-BFGS-B',
                                           tol=1e-9).x
                              for snapshot in range(num_snapshots)]).T
        poly = np.concatenate(polynomial_form(Q_reduced), axis=0)

        # Convergence check
        energy = np.linalg.norm(V_reduced @ Q_reduced
                                + V_reduced_nonlin @ Xi @ poly, 'fro') ** 2 \
            / np.linalg.norm(Q_shifted, 'fro') ** 2
        diff = abs(energy - nrg_old)
        print(f"\titeration: {niter+1}\tsnapshot energy: {energy:e}\tdiff: {diff:e}")
        if diff < tol:
            print("***Convergence criterion active!")
            break
        nrg_old = energy

    # Final state reconstruction
    states_reproduced_MAM = reference_states + V_reduced @ Q_reduced + V_reduced_nonlin @ Xi @ poly

    abs_froerr, rel_froerr = opinf.post.frobenius_error(Qtrue=Q_true,
                                                        Qapprox=states_reproduced_MAM)

    print("MAM Reconstruction error: "
          f"{rel_froerr:.4%}")

    return Q_reduced, V_reduced, V_reduced_nonlin, Xi


def get_basis_and_reduced_data(V_X, V_T, Q_shifted, reference_states):
    """Select the appropriate basis method (POD, NL-POD, or AM) and compute
    reduced data."""
    if Params.basis == 'POD':
        Q_reduced, V_reduced, V_reduced_nonlin, Xi = _linear_pod_basis(
            V_X, V_T, Q_shifted, reference_states)
        return Q_reduced, V_reduced, V_reduced_nonlin, Xi
    elif Params.basis == 'NL-POD':
        Q_reduced, V_reduced, V_reduced_nonlin, Xi = _nonlinear_pod_basis(
            V_X, V_T, Q_shifted, reference_states)
        return Q_reduced, V_reduced, V_reduced_nonlin, Xi
    elif Params.basis == 'AM':
        Q_reduced, V_reduced, V_reduced_nonlin, Xi = _nonlinear_basis(
            V_X, V_T, Q_shifted, reference_states)
        return Q_reduced, V_reduced, V_reduced_nonlin, Xi
    else:
        print("Chosen basis unknown. Please choose POD, NL-POD, or AM.")
        return None, None, None, None


def basis_multi(*args):
    """
    Function to stack multiple matrices.

    Parameters:
    ----------
    *args: Alternating sequence of matrices and their respective ranks. For
    example, U1, r1, U2, r2, , Un, rn where Ui is a matrix and ri is its rank.

    Returns:
    -------
    basis_reduced: Block diagonal matrix consisting of the input matrices
    reduced to their respective ranks.
    """
    # Check if the number of arguments is even
    if len(args) % 2 != 0:
        raise ValueError("The number of arguments must be even.")

    # Split the arguments into matrices and ranks
    matrices = args[::2]
    ranks = args[1::2]

    # Reduce each matrix to its rank and stack them
    reduced_matrices = [U[:, :r] for U, r in zip(matrices, ranks)]
    basis_reduced = la.block_diag(*reduced_matrices)
    # print(f"Shape of projection matrix: {basis_reduced.shape}")
    return basis_reduced


def basis_nonlin_multi(*args):
    """
    Function to stack multiple matrices.

    Parameters:
    ----------
    *args: Alternating sequence of matrices and their respective ranks. For
    example, U1, r1, U2, r2, , Un, rn where Ui is a matrix and ri is its rank.

    Returns:
    -------
    basis_reduced: Block diagonal matrix consisting of the input matrices
    reduced to their respective ranks.
    """
    # Check if the number of arguments is even
    if len(args) % 2 != 0:
        raise ValueError("The number of arguments must be even.")

    # Split the arguments into matrices and ranks
    matrices = args[::3]
    rank_lin = args[1::3]
    rank_nonlin = args[2::3]

    # Reduce each matrix to its rank and stack them
    reduced_matrices = [U[:, r:r+q] for U, r, q in zip(matrices, rank_lin, rank_nonlin)]
    basis_reduced = la.block_diag(*reduced_matrices)
    # print(f"Shape of projection matrix: {basis_reduced.shape}")
    return basis_reduced



# https://willcox-research-group.github.io/rom-operator-inference-Python3/_modules/opinf/basis/_pod.html#svdval_decay
def cumulative_energy(singular_values, thresh=.9999,
                      plot=True, ax=None, name_tag=None):
    """Compute the number of singular values needed to surpass a given
    energy threshold. The energy of j singular values is defined by

        energy_j = sum(singular_values[:j]**2) / sum(singular_values**2).

    Parameters
    ----------
    singular_values : (n,) ndarray
        Singular values of a snapshot set, e.g., scipy.linalg.svdvals(states).
    thresh : float or list(float)
        Energy capture threshold(s). Default is 99.99%.
    plot : bool
        If True, plot the singular values and the cumulative energy against
        the singular value index (linear scale).
    ax : plt.Axes or None
        Matplotlib Axes to plot the results on if plot = True.
        If not given, a new single-axes figure is created.

    Returns
    -------
    ranks : int or list(int)
        The number of singular values required to capture more than each
        energy capture threshold.
    """
    # Calculate the cumulative energy.
    svdvals2 = np.sort(singular_values)[::-1]**2
    cum_energy = np.cumsum(svdvals2) / np.sum(svdvals2)

    # Determine the points at which the cumulative energy passes the threshold.
    one_thresh = np.isscalar(thresh)
    if one_thresh:
        thresh = [thresh]
    ranks = [int(np.searchsorted(cum_energy, xi)) + 1 for xi in thresh]

    print(f"r = {ranks[0]} singular values " + f"exceed {thresh[0]} energy")

    if plot:
        cum_energy_percent = cum_energy*100
        plt.figure(figsize=(8, 6), dpi=300)
        ax = plt.subplot(111)
        j = np.arange(1, singular_values.size + 1)
        # Visualize cumulative energy using the institute's color palette
        # Set lw (line width) to 0 to have no line between the stars
        if name_tag == 'temperature T in K':
            ax.plot(j, cum_energy_percent, marker='o',
                    color=mpi_colors['mpi_red'], ms=12, lw=0, zorder=3)
        else:
            ax.plot(j, cum_energy_percent, marker='d',
                    color=mpi_colors['mpi_green'], ms=12, lw=0, zorder=3)
        # # Set the y-axis to logarithmic scale
        # ax.set_yscale('log')

        # Set the y-axis to display numerical values only
        # The formatter might need to be updated to work with logarithmic scale
        ax.get_yaxis().set_major_formatter(
            plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))

        # Set axis limits and labels with improved readability
        ax.set_xlim(-0.2, 30)
        # The ylim might need to be updated because the log scale does not
        # support 0 or negative values
        # ax.set_ylim(bottom=99.9, top=100.01)
        ax.set_ylim(bottom=95.0, top=100.2)

        ax.set_xlabel(r"singular value index $j$", fontsize=20)
        ax.set_ylabel(r"cumulative variance $\xi$ (\%)", fontsize=20)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))  # Set the number of x-ticks
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))  # Set the number of y-ticks
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_title(f"cumulative energy {name_tag}", fontsize=20)
        plt.tight_layout()

        file_name = (f'cumulative_energy_{name_tag}.svg'
                     if name_tag is not None else 'cumulative_energy.svg')
        plt.savefig(f'./results/{file_name}',
                    bbox_inches='tight', transparent=True)

        # Show the plot
        plt.show()

    return ranks[0] if one_thresh else ranks


# https://willcox-research-group.github.io/rom-operator-inference-Python3/_modules/opinf/basis/_pod.html#svdval_decay
def svdval_decay(singular_values, tol=1e-8, normalize=True,
                 plot=True, ax=None, name_tag=None):
    """Count the number of normalized singular values that are greater than
    the specified tolerance.

    Parameters
    ----------
    singular_values : (n,) ndarray
        Singular values of a snapshot set, e.g., scipy.linalg.svdvals(states).
    tol : float or list(float)
        Cutoff value(s) for the singular values.
    normalize : bool
        If True, normalize so that the maximum singular value is 1.
    plot : bool
        If True, plot the singular values and the cutoff value(s) against the
        singular value index.
    ax : plt.Axes or None
        Matplotlib Axes to plot the results on if plot = True.
        If not given, a new single-axes figure is created.

    Returns
    -------
    ranks : int or list(int)
        The number of singular values greater than the cutoff value(s).
    """
    # Calculate the number of singular values above the cutoff value(s).
    one_tol = np.isscalar(tol)
    if one_tol:
        tol = [tol]
    singular_values = np.sort(singular_values)[::-1]
    if normalize:
        singular_values /= singular_values[0]
    ranks = [np.count_nonzero(singular_values > epsilon) for epsilon in tol]

    print(f"{ranks[0]} normalized singular values are greater than " +
          f"10^({int(np.log10(tol)):d})")

    if plot:
        # Visualize singular values and cutoff value(s).
        plt.figure(figsize=(10, 6), dpi=300)
        ax = plt.subplot(111)
        # Visualize singular values using the institute's color palette
        j = np.arange(1, singular_values.size + 1)
        ax.semilogy(j, singular_values, marker='o', lw=0,
                    ms=12, color=mpi_colors['mpi_blue'], mew=0, zorder=3)
        # Add cutoff lines
        # for epsilon, r in zip(tol, ranks):
        #     ax.axhline(epsilon, color=mpi_colors['mpi_grey'], linewidth=1,
        #                alpha=.75)
        #     ax.axvline(r, color=mpi_colors['mpi_grey'], linewidth=1, alpha=.75)
        # limits on axis
        # Set axis limits
        ax.set_xlim(1, 200)
        # ax.set_ylim()  # Uncomment and set specific y-limits if required
        # Set labels with improved readability
        ax.set_xlabel(r"singular value index $j$", fontsize=14)
        ax.set_ylabel(r"relative singular values", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_title(f"singular value decay {name_tag}", fontsize=16)
        # Tight layout and save the figure
        plt.tight_layout()
        file_name = (f'svdval_decay_{name_tag}.svg'
                     if name_tag is not None else 'cumulative_energy.svg')
        plt.savefig(f'./results/{file_name}',
                    bbox_inches='tight', transparent=True)
        plt.show()
    return ranks[0] if one_tol else ranks


def residual_energy(singular_values, tol=1e-6, plot=True, ax=None):
    """Compute the number of singular values needed such that the residual
    energy drops beneath the given tolerance. The residual energy of j
    singular values is defined by

        residual_j = 1 - sum(singular_values[:j]**2) / sum(singular_values**2).

    Parameters
    ----------
    singular_values : (n,) ndarray
        Singular values of a snapshot set, e.g., scipy.linalg.svdvals(states).
    tol : float or list(float)
        Energy residual tolerance(s). Default is 10^-6.
    plot : bool
        If True, plot the singular values and the residual energy against
        the singular value index (log scale).
    ax : plt.Axes or None
        Matplotlib Axes to plot the results on if plot = True.
        If not given, a new single-axes figure is created.

    Returns
    -------
    ranks : int or list(int)
        Number of singular values required to for the residual energy to drop
        beneath each tolerance.
    """
    # Calculate the residual energy.
    svdvals2 = np.sort(singular_values)[::-1] ** 2
    res_energy = 1 - (np.cumsum(svdvals2) / np.sum(svdvals2))

    # Determine the points when the residual energy dips under the tolerance.
    one_tol = np.isscalar(tol)
    if one_tol:
        tol = [tol]
    ranks = [np.count_nonzero(res_energy > epsilon) + 1 for epsilon in tol]

    if plot:
        # Visualize residual energy and tolerance value(s).
        if ax is None:
            ax = plt.figure().add_subplot(111)
        j = np.arange(1, singular_values.size + 1)
        ax.semilogy(j, res_energy, "C1.-", ms=10, lw=1, zorder=3)
        ax.set_xlim(0, j.size)
        for epsilon, r in zip(tol, ranks):
            ax.axhline(epsilon, color="black", linewidth=0.5, alpha=0.5)
            ax.axvline(r, color="black", linewidth=0.5, alpha=0.5)
        ax.set_xlabel(r"Singular value index")
        ax.set_ylabel(r"Residual energy")

    return ranks[0] if one_tol else ranks


def svd_results(singular_values, name_tag):
    """
    Function to plot the singular values and cumulative energy of a dataset.

    Parameters:
    -----------
    singular_values (numpy.ndarray): The singular values of the dataset.
    name_tag (str): A tag to add to the title and filename of the plot.

    Returns:
    -------
    None
    """
    # Create a figure with improved aesthetics for publication
    plt.figure(figsize=(12, 8), dpi=300)  # Increase the size of the plot
    ax1 = plt.subplot(111)

    # Calculate the square of the sorted singular values
    svdvals2 = np.sort(singular_values)[::-1]**2

    # Calculate the cumulative energy percentage
    cum_energy = np.cumsum(svdvals2) / np.sum(svdvals2)
    cum_energy_percent = cum_energy*100

    # Visualize singular values using the institute's color palette
    j = np.arange(1, singular_values.size + 1)
    ax1.semilogy(j, singular_values, marker='o', ms=10, lw=0,
                 color=mpi_colors['mpi_blue'], mew=0, zorder=3)

    # Set axis limits and labels for the left axis
    ax1.set_xlim(1, 30)
    ax1.set_xlabel(r"singular value index $j$", fontsize=24)
    ax1.set_ylabel(r"relative singular values", fontsize=24)
    ax1.yaxis.label.set_color(mpi_colors['mpi_blue'])
    ax1.tick_params(axis='both', which='major', labelsize=24)

    # Set the lower y-limit to 10^-4
    # ax1.set_ylim(10e-3, 10e5)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(5))  # Set the number of x-ticks to 5

    # Create a second set of axes for the cumulative energy percentage
    ax2 = ax1.twinx()
    ax2.plot(j, cum_energy_percent, marker='d',
             color=mpi_colors['mpi_green'], ms=10, lw=0, zorder=3)

    # Set axis limits and labels for the right axis
    ax2.set_xlim(1, 30)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(5))  # Set the number of y-ticks to 5
    ax2.set_ylim(99.9, 100.01)
    ax2.set_ylabel(r"cumulative energy $\xi$ in \%", fontsize=24)
    ax2.yaxis.label.set_color(mpi_colors['mpi_green'])
    ax2.tick_params(axis='both', which='major', labelsize=24)

    # Set the y-axis to display numerical values with two decimal places
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y,
                                                _: '{:.2f}'.format(y)))

    # Set the title of the plot
    title_text = f"SVD results - {name_tag}" if name_tag else "SVD results"
    plt.title(title_text, fontsize=28)

    # Tight layout and save the figure
    plt.tight_layout()
    file_name = (f'combined_svd_{name_tag}.svg'
                 if name_tag is not None else 'combined_svd.svg')
    plt.savefig(f'./results/{file_name}',
                bbox_inches='tight', transparent=True)

    # Show the plot
    plt.show()

    return
