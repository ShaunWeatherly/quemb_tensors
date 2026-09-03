"""
h_oep_poc.py

Proof-of-concept implementation of a holomorphic optimized effective potential
(h-OEP) fit to a correlated, non-idempotent target 1RDM in a non-orthogonal AO basis.

Features
--------
- Non-orthogonal AO basis with user-provided overlap matrix S
- Zeroth-order RHF Fock matrix as input
- Auxiliary finite-temperature Fermi-Dirac occupations
- beta and mu included as optimization variables
- Low-rank nonlocal factorized ansatz, initialized from leading singular vectors
  of the RHF/target 1RDM mismatch
- Hermitian and anti-Hermitian sectors optimized by alternating block optimization
- Frobenius-norm cost
- Analytic parameter gradients built from the 1RDM density response
- scipy.optimize.dual_annealing with L-BFGS-B local minimizer
- Optional soft penalties for:
    * particle number
    * parity   (placeholder)
    * time-reversal (placeholder / simple default)
- Optional L2 regularization of the anti-Hermitian sector

Notes
-----
1. This is a proof of concept.
2. For non-Hermitian effective Hamiltonians, occupations are defined from the REAL parts
   of the generalized eigenvalues.
3. The target 1RDM is assumed to be supplied in the same AO representation as the input
   Fock matrix, using the convention N = Tr(gamma @ S).
4. The parity/time-reversal constraints are currently placeholders.

Dependencies
------------
numpy, scipy

Example
-------
Run this file directly:
    python h_oep_poc.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.linalg import eig, eigh, svd
from scipy.optimize import dual_annealing, linear_sum_assignment

from quemb.shared.helper import unused

# =========================
# Linear algebra utilities
# =========================


def frob2(a: np.ndarray) -> float:
    """Returns Frobenius norm (real+PSD ) of a matrix, a."""
    return float(np.real(np.vdot(a, a)))


def make_spd(n: int, seed: int = 0) -> np.ndarray:
    """Returns random symmetric positive definite matrix, s."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, n))
    s = x.T @ x + n * np.eye(n)
    return s


def symm(a: np.ndarray) -> np.ndarray:
    """Returns Hermitian operator from a matrix, a."""
    return 0.5 * (a + a.conj().T)


def anti_symm(a: np.ndarray) -> np.ndarray:
    """Returns Anti-Hermitian operator from a matrix, a."""
    return 0.5 * (a - a.conj().T)


def kpt_interpolate(path, values, icell, bz2ibz, size, offset=(0, 0, 0), pad_width=2):
    """Interpolate values from Monkhorst-Pack sampling.

    Takes a set of `values`, for example
    eigenvalues, that are resolved on a Monkhorst Pack k-point grid given by
    `size` and `offset` and interpolates the values onto the k-points
    in `path`.

    Note
    ----
    For the interpolation to work, path has to lie inside the domain
    that is spanned by the MP kpoint grid given by size and offset.

    To try to ensure this we expand the domain slightly by adding additional
    entries along the edges and sides of the domain with values determined by
    wrapping the values to the opposite side of the domain. In this way we
    assume that the function to be interpolated is a periodic function in
    k-space. The padding width is determined by the `pad_width` parameter.

    Parameters
    ----------
    path: (nk, 3) array-like
        Desired path in units of reciprocal lattice vectors.
    values: (nibz, ...) array-like
        Values on Monkhorst-Pack grid.
    icell: (3, 3) array-like
        Reciprocal lattice vectors.
    bz2ibz: (nbz,) array-like of int
        Map from nbz points in BZ to nibz reduced points in IBZ.
    size: (3,) array-like of int
        Size of Monkhorst-Pack grid.
    offset: (3,) array-like
        Offset of Monkhorst-Pack grid.
    pad_width: int
        Padding width to aid interpolation

    Returns
    -------
    (nbz,) array-like
        *values* interpolated to *path*.
    """
    interpolator = LinearNDInterpolator

    path = (np.asarray(path) + 0.5) % 1 - 0.5
    path = np.dot(path, icell)
    # Fold out values from IBZ to BZ:
    v = np.asarray(values)[bz2ibz]
    v = v.reshape(tuple(size) + v.shape[1:])

    # Create padded Monkhorst-Pack grid:
    size = np.asarray(size)
    i = np.indices(size + 2 * pad_width).transpose((1, 2, 3, 0)).reshape((-1, 3))
    k = (i - pad_width + 0.5) / size - 0.5 + offset
    k = np.dot(k, icell)

    # Fill in boundary values:
    V = np.pad(v, [(pad_width, pad_width)] * 3 + [(0, 0)] * (v.ndim - 3), mode="wrap")

    interpolate = interpolator(k, V.reshape((-1,) + V.shape[3:]))
    interpolated_points = interpolate(path)

    # NaN values indicate points outside interpolation domain, if fail
    # try increasing padding
    assert not np.isnan(interpolated_points).any(), (
        "Points outside interpolation domain. Try increasing pad_width."
    )

    return interpolated_points


def get_bands(
    hcore: np.ndarray,
    v_eff: np.ndarray,
    ovlp: np.ndarray,
    nkpts_band: int,
    u_corr: np.ndarray | None = None,
    hermitian: bool = True,
) -> tuple:
    """
    Compute energy bands from an effective one-body Hamiltonian.

    Assumes an effective Hamiltonian of the form:
        h = F + u_corr
        h = [hcore + v_eff] + u_corr
    where 'hcore' is the true one-body Hamiltonian, 'v_eff' is the
    Hartree-Fock one-body potential, and 'u_corr' is an effective
    correlation potential dressing the Fock Hamiltonian.

    Parameters
    ----------
    hcore :
        True one-body (core) component of the Hamiltonian.
    v_eff :
        Effective one-body potential (typically Hartree-Fock).
    ovlp :
        Atomic orbital overlap matrix (S_ij).
    nkpts_band :
        Integer number of k-points in the band path.
    u_corr :
        Effective one-body correlation potential.

    Returns
    -------
    mo_energy : (nmo,) ndarray or a list of (nmo,) ndarray
        Bands energies E_n(k)
    mo_coeff : (nao, nmo) ndarray or a list of (nao,nmo) ndarray
        Band orbitals psi_n(k)
    """
    fock = hcore + v_eff
    if u_corr is not None:
        fock = fock + u_corr

    eig_kpts = []
    lmo_coeff_kpts = []
    rmo_coeff_kpts = []
    for k in range(0, nkpts_band):
        s, U = np.linalg.eigh(ovlp[k])
        X = U @ np.diag(s ** (-0.50))
        F = X.T.conj() @ (fock[k] @ X)
        if hermitian:
            eigs, vecs = np.linalg.eigh(F, UPLO="U")
            idx = np.argmax(abs(vecs.real), axis=0)
            C_mo = np.dot(X, vecs)
            C_mo[:, C_mo[idx, np.arange(len(eigs))].real < 0] *= -1
            eig_kpts.append(np.complex128(eigs))
            rmo_coeff_kpts.append(C_mo)
        else:
            eigs, lvecs, rvecs = eig(
                F,
                left=True,
                right=True,
            )
            idx = np.argmax(abs(lvecs.real), axis=0)
            lC_mo = np.dot(X, lvecs)
            rC_mo = np.dot(X, rvecs)
            lC_mo[:, lC_mo[idx, np.arange(len(eigs))].real < 0] *= -1
            rC_mo[:, rC_mo[idx, np.arange(len(eigs))].real < 0] *= -1
            eig_kpts.append(eigs)
            lmo_coeff_kpts.append(lC_mo)
            rmo_coeff_kpts.append(rC_mo)

    return (eig_kpts, lmo_coeff_kpts, rmo_coeff_kpts)


def matrix_sqrt_and_invsqrt_spd(
    s: np.ndarray, thresh: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    """Returns square root (+ inverse) of the matrix s."""
    eigs, evecs = eigh(s)
    eigs = np.clip(eigs, thresh, None)
    sq = evecs @ np.diag(np.sqrt(eigs)) @ evecs.conj().T
    isq = evecs @ np.diag(1.0 / np.sqrt(eigs)) @ evecs.conj().T
    return sq, isq


def project_out_identity(X: np.ndarray) -> np.ndarray:
    """Returns X with sectors prop. to identity removed."""
    n = X.shape[0]
    return X - np.trace(X) / n * np.eye(n, dtype=X.dtype)


def homo_lumo_midpoint_mu(eigs_real: np.ndarray, n_occ: int) -> float:
    """
    Returns chemical potential as the midpoint between HOMO and LUMO eigs.

    Parameters
    ----------
    eigs_real : ndarray
        Real parts of eigenvalues, assumed sorted ascending.
    n_occ : int
        Number of occupied orbitals used to define HOMO/LUMO.

    Returns
    -------
    mu : float
    """
    n = len(eigs_real)

    if n_occ <= 0:
        # No occupied states: place mu below the spectrum
        return float(eigs_real[0] - 1.0)
    if n_occ >= n:
        # All occupied: place mu above the spectrum
        return float(eigs_real[-1] + 1.0)

    eps_homo = eigs_real[n_occ - 1]
    eps_lumo = eigs_real[n_occ]
    return float(0.5 * (eps_homo + eps_lumo))


def generalized_biorthogonal_eig(
    h: np.ndarray,
    s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve h R = S R e
          h^H L = S^H L e^*
    and normalize so that L^H S R ~ I.
    """
    eigs, vl, vr = eig(h, b=s, left=True, right=True)

    # Sort by real parts only
    eigs_r = np.real(eigs)
    idx = np.argsort(eigs_r)
    eigs = eigs[idx]
    vl = vl[:, idx]
    vr = vr[:, idx]

    # Pairwise normalization
    m = vl.conj().T @ s @ vr
    for i in range(len(eigs)):
        di = m[i, i]
        if abs(di) < 1e-13:
            continue
        fac = np.sqrt(di)
        vr[:, i] /= fac
        vl[:, i] /= np.conj(fac)

    # Final cleanup by diagonal rescaling
    m = vl.conj().T @ s @ vr
    for i in range(len(eigs)):
        di = m[i, i]
        if abs(di) < 1e-13:
            continue
        vr[:, i] /= di

    return eigs, vl, vr


def fermi_occ(
    eigs_real: np.ndarray, mu: float, beta: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns occupations f and derivative df/deps.
    f = 1/(exp(beta*(eps-mu))+1)
    df/deps = -beta f(1-f)
    """
    x = beta * (eigs_real - mu)
    x = np.clip(x, -100, 100)
    f = 1.0 / (np.exp(x) + 1)
    # f = expit(-x)
    df_de = -beta * f * (1.0 - f)
    return f, df_de


def density_from_hermitian_spectrum(
    h: np.ndarray,
    s: np.ndarray,
    C_mo: np.ndarray,
    n_occ: int,
    beta: float,
    mu: float | None = None,
) -> dict[str, np.ndarray]:
    """ """
    s, U = np.linalg.eigh(s)
    X = U @ np.diag(s ** (-0.50))
    _F = X.T.conj() @ (h @ X)
    eigs, vecs = np.linalg.eigh(_F)
    eigs_r = np.real(eigs)
    idx = np.argsort(eigs_r)
    eigs = eigs[idx]
    vecs = np.dot(vecs[:, idx], C_mo)
    if mu is None:
        mu = homo_lumo_midpoint_mu(eigs, n_occ=n_occ)
    occ, df_de = fermi_occ(eigs, mu, beta)
    _C_mo = np.dot(X, vecs)
    gamma = _C_mo @ np.diag(occ) @ _C_mo.conj().T
    return {
        "eigs": eigs,
        "eigs_r": eigs,
        "L": _C_mo,
        "R": _C_mo,
        "occ": occ,
        "df_de": df_de,
        "gamma": gamma,
        "mu": mu,
    }


def density_from_generalized_spectrum(
    h: np.ndarray,
    s: np.ndarray,
    C_mo: np.ndarray,
    n_occ: int,
    beta: float,
    mu: float | None = None,
) -> dict[str, np.ndarray]:
    """
    Build gamma = R diag(f) L^H
    with occupations from Re(eigs).
    """
    eigs, L, R = generalized_biorthogonal_eig(h, s)
    eigs_r = np.real(eigs)
    if mu is None:
        mu = homo_lumo_midpoint_mu(eigs_r, n_occ=n_occ)
    occ, df_de = fermi_occ(eigs_r, mu, beta)
    L = np.dot(L, C_mo)[:, occ > 0]
    R = np.dot(R, C_mo)[:, occ > 0]
    gamma = R @ np.diag(occ[occ > 0]) @ L.conj().T
    return {
        "eigs": eigs,
        "eigs_r": eigs_r,
        "L": L,
        "R": R,
        "occ": occ,
        "df_de": df_de,
        "gamma": gamma,
        "mu": mu,
    }


def _biorthonormalize_columns(L, R, S, eps=1e-14):
    """
    Rescale left/right eigenvector columns so that, approximately,
        L^dagger S R = I
    on the diagonal.

    This is a diagonal rescaling only; it assumes the eigensystem is already
    reasonably biorthogonal.
    """
    L = np.array(L, dtype=np.complex128, copy=True)
    R = np.array(R, dtype=np.complex128, copy=True)

    M = L.conj().T @ S @ R
    n = M.shape[0]

    for i in range(n):
        di = M[i, i]
        if abs(di) < eps:
            continue
        fac = np.sqrt(di)
        R[:, i] /= fac
        L[:, i] /= np.conj(fac)

    M = L.conj().T @ S @ R
    for i in range(n):
        di = M[i, i]
        if abs(di) < eps:
            continue
        R[:, i] /= di

    return L, R


def _cluster_degenerate_bands(evals, tol=1e-6):
    """
    Group neighboring bands into clusters if their eigenvalues are close.
    Clustering is done on Re(evals), assuming evals are already sorted.

    Returns
    -------
    clusters : list[list[int]]
        Example: [[0], [1,2], [3], [4,5,6]]
    """
    x = np.real(evals)
    n = len(x)
    if n == 0:
        return []

    clusters = [[0]]
    for i in range(1, n):
        if abs(x[i] - x[i - 1]) < tol:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    return clusters


def _neighbor_overlap(L_prev, R_curr, S_curr):
    """
    Biorthogonal overlap matrix
        O_ij = <L_i(prev)| S(curr) |R_j(curr)>
    """
    return L_prev.conj().T @ S_curr @ R_curr


def _hungarian_perm_from_overlap(O):
    """
    Permutation maximizing total |O_ij|.

    Returns
    -------
    perm : ndarray, shape (n,)
        perm[i] = column index in 'current' matched to row i in 'previous'
    score : float
    """
    W = np.abs(O)
    row_ind, col_ind = linear_sum_assignment(-W)
    perm = np.zeros_like(col_ind)
    perm[row_ind] = col_ind
    score = float(W[row_ind, col_ind].sum())
    return perm, score


def _phase_align_columns(L_prev, R_curr, L_curr, S_curr, eps=1e-14):
    """
    Fix the residual per-band gauge by enforcing
        <L_i(prev)|S(curr)|R_i(curr)>
    to be real-positive.

    Transformation:
        R_i -> e^{+i theta} R_i
        L_i -> e^{-i theta} L_i
    so that projectors R_i L_i^dagger are unchanged.
    """
    R_curr = np.array(R_curr, dtype=np.complex128, copy=True)
    L_curr = np.array(L_curr, dtype=np.complex128, copy=True)

    n = R_curr.shape[1]
    for i in range(n):
        z = L_prev[:, i].conj().T @ S_curr @ R_curr[:, i]
        if abs(z) < eps:
            continue
        phase = np.exp(-1j * np.angle(z))
        R_curr[:, i] *= phase
        L_curr[:, i] /= phase

    return L_curr, R_curr


def _permute_clustered_by_subspace_overlap(
    L_prev, R_curr, S_curr, evals_curr, cluster_tol=1e-6
):
    """
    More robust matching that respects near-degenerate clusters in the current spectrum.

    Strategy:
    1. Find clusters of nearly degenerate current bands.
    2. For isolated bands, standard maximum-overlap matching works.
    3. For degenerate clusters, keep the whole cluster together and assign the
       best-matching set of previous bands to it.
    4. Within each cluster, order by overlap greedily/Hungarian on the sub-block.

    Returns
    -------
    perm : ndarray
        Global permutation for current bands.
    score : float
    """
    n = len(evals_curr)
    O = _neighbor_overlap(L_prev, R_curr, S_curr)
    clusters = _cluster_degenerate_bands(evals_curr, tol=cluster_tol)

    if all(len(c) == 1 for c in clusters):
        return _hungarian_perm_from_overlap(O)

    # First, do a global Hungarian assignment.
    # Then refine within each degenerate cluster.
    perm, score = _hungarian_perm_from_overlap(O)

    perm_refined = perm.copy()
    used_prev = set()

    for cluster in clusters:
        if len(cluster) == 1:
            used_prev.add(cluster[0])
            continue

        # Current-band columns in this cluster
        curr_cols = np.array(cluster, dtype=int)

        # Previous rows matched to these columns under the global assignment
        prev_rows = np.array([i for i in range(n) if perm[i] in curr_cols], dtype=int)

        if len(prev_rows) != len(curr_cols):
            continue

        Osub = O[np.ix_(prev_rows, curr_cols)]
        subperm, _ = _hungarian_perm_from_overlap(Osub)

        # subperm maps local prev-row index -> local curr-col index
        for a, prow in enumerate(prev_rows):
            perm_refined[prow] = curr_cols[subperm[a]]
            used_prev.add(prow)

    return perm_refined, score


def compute_overlap_matrix(
    right_prev,
    right_curr,
    S=None,
    left_prev=None,
):
    """
    Compute band-overlap matrix between neighboring k-points.

    Parameters
    ----------
    right_prev : (nao, nbands) complex ndarray
        Right eigenvectors at k.
    right_curr : (nao, nbands) complex ndarray
        Right eigenvectors at k+1.
    S : (nao, nao) complex ndarray or None
        AO overlap matrix. If None, identity is assumed.
    left_prev : (nao, nbands) complex ndarray or None
        Left eigenvectors at k. If provided, use biorthogonal overlap
            <L_i(k)|S|R_j(k+1)>
        Otherwise use
            <R_i(k)|S|R_j(k+1)>.

    Returns
    -------
    O : (nbands, nbands) complex ndarray
        Overlap matrix.
    """
    if S is None:
        S = np.eye(right_prev.shape[0], dtype=right_prev.dtype)

    if left_prev is None:
        O = right_prev.conj().T @ S @ right_curr
    else:
        O = left_prev.conj().T @ S @ right_curr
    return O


def match_bands_by_overlap(overlap_matrix):
    """
    Find permutation maximizing total |overlap|.

    Returns
    -------
    perm : (nbands,) ndarray of int
        perm[i] = column index in current k-point matched to
            band i from previous k-point
    score : float
        Sum of absolute overlaps for the chosen assignment.
    """
    weight = np.abs(overlap_matrix)
    # Hungarian solves a minimization problem, so minimize -weight
    row_ind, col_ind = linear_sum_assignment(-weight)
    perm = np.zeros_like(col_ind)
    perm[row_ind] = col_ind
    score = float(weight[row_ind, col_ind].sum())
    return perm, score


def apply_permutation_to_kpoint(evals_k, right_k, left_k, perm):
    """
    Reorder eigenvalues/eigenvectors at a single k-point.
    """
    evals_new = evals_k[perm]
    right_new = right_k[:, perm]
    left_new = None if left_k is None else left_k[:, perm]
    return evals_new, right_new, left_new


def phase_align_kpoint(
    right_prev,
    right_curr,
    S=None,
    left_prev=None,
    left_curr=None,
):
    """
    Smooth phases after permutation so neighboring matched bands have
    overlaps as real-positive as possible.

    For each band i, compute
        z_i = <L_i(prev)|S|R_i(curr)>
    if left_prev is available, otherwise use right-right overlap.

    Then rotate current vectors by exp(-i arg z_i).

    If left_curr is provided, rotate it by the inverse phase so that
    outer products |R><L| remain unchanged.
    """
    if S is None:
        S = np.eye(right_prev.shape[0], dtype=right_prev.dtype)

    nbands = right_prev.shape[1]
    right_out = right_curr.copy()
    left_out = None if left_curr is None else left_curr.copy()

    for i in range(nbands):
        rp = right_prev[:, i]
        rc = right_out[:, i]

        if left_prev is None:
            z = rp.conj().T @ S @ rc
        else:
            lp = left_prev[:, i]
            z = lp.conj().T @ S @ rc

        if abs(z) > 1e-14:
            phase = np.exp(-1j * np.angle(z))
            right_out[:, i] *= phase
            if left_out is not None:
                left_out[:, i] *= np.conj(phase) ** (-1)  # equivalent to *= phase
                # For |R><L| invariance under R->e^{iθ}R, need L->e^{-iθ}L:
                left_out[:, i] /= phase

    return right_out, left_out


def enforce_band_continuity(
    evals,
    right_vecs,
    S=None,
    left_vecs=None,
    start_k=0,
    do_phase_alignment=True,
    return_scores=True,
):
    """
    Enforce smooth band connectivity across k-points.

    Parameters
    ----------
    evals : (nk, nbands) complex ndarray
        Eigenvalues at each k-point.
    right_vecs : (nk, nao, nbands) complex ndarray
        Right eigenvectors at each k-point.
    S : (nao, nao) complex ndarray or None
        AO overlap matrix.
    left_vecs : (nk, nao, nbands) complex ndarray or None
        Left eigenvectors. Recommended for non-Hermitian h-OEP.
    start_k : int
        k-point used as the anchor. Continuity is propagated forward from here.
    do_phase_alignment : bool
        Whether to smooth phases after band matching.
    return_scores : bool
        Whether to return overlap scores and permutations.

    Returns
    -------
    evals_ord : (nk, nbands) complex ndarray
    right_ord : (nk, nao, nbands) complex ndarray
    left_ord : (nk, nao, nbands) complex ndarray or None
    info : dict
        Contains:
            "permutations": list of length nk
            "scores": list of length nk-1
            "overlap_matrices": list of overlap matrices used
    """
    evals = np.array(evals, copy=True)
    right_vecs = np.array(right_vecs, copy=True)
    left_vecs = None if left_vecs is None else np.array(left_vecs, copy=True)

    nk, nao, nbands = right_vecs.shape

    evals_ord = evals.copy()
    right_ord = right_vecs.copy()
    left_ord = None if left_vecs is None else left_vecs.copy()

    permutations = [np.arange(nbands, dtype=int)]
    scores = []
    overlaps = []

    # Forward sweep from start_k to end
    for k in range(start_k, nk - 1):
        _S = S[k]
        O = compute_overlap_matrix(
            right_prev=right_ord[k],
            right_curr=right_ord[k + 1],
            S=_S,
            left_prev=None if left_ord is None else left_ord[k],
        )
        perm, score = match_bands_by_overlap(O)

        ev_new, r_new, l_new = apply_permutation_to_kpoint(
            evals_ord[k + 1],
            right_ord[k + 1],
            None if left_ord is None else left_ord[k + 1],
            perm,
        )

        if do_phase_alignment:
            r_new, l_new = phase_align_kpoint(
                right_prev=right_ord[k],
                right_curr=r_new,
                S=_S,
                left_prev=None if left_ord is None else left_ord[k],
                left_curr=l_new,
            )

        evals_ord[k + 1] = ev_new
        right_ord[k + 1] = r_new
        if left_ord is not None:
            left_ord[k + 1] = l_new

        permutations.append(perm)
        scores.append(score)
        overlaps.append(O)

    info = {
        "permutations": permutations,
        "scores": scores,
        "overlap_matrices": overlaps,
    }

    if return_scores:
        return evals_ord, right_ord, left_ord, info
    return evals_ord, right_ord, left_ord


def fix_k_gauge(
    spec_k,
    overlaps_k,
    start_k=0,
    previous_spec=None,
    cluster_tol=1e-6,
    do_phase_alignment=True,
    rebiorthonormalize=True,
):
    """
    Enforce a smooth k-space gauge on a sequence of AO-basis generalized eigensystems.

    Parameters
    ----------
    spec_k : list[dict]
        One dict per k-point. Each dict must contain at least:
            {
              "evals": (nband,) complex ndarray,
              "L":     (nao, nband) complex ndarray,
              "R":     (nao, nband) complex ndarray,
            }
        Optional extra keys are preserved and copied through.

        Assumes the eigensystem at each k solves, in the AO basis,
            H(k) R(k) = S(k) R(k) E(k)
        with left eigenvectors satisfying
            L(k)^dagger S(k) R(k) = I
        approximately.

    overlaps_k : ndarray, shape (nk, nao, nao)
        AO overlap matrices S(k).

    start_k : int
        Anchor k-point for forward propagation.

    previous_spec : dict or None
        Optional reference eigensystem from the previous optimization iteration,
        used to fix the gauge at start_k before propagating in k.
        Must contain:
            {
              "L": (nao, nband),
              "R": (nao, nband),
            }
        and is interpreted as the ordered gauge at the same k-point index `start_k`.

    cluster_tol : float
        Tolerance for grouping near-degenerate bands by Re(evals).

    do_phase_alignment : bool
        If True, align residual per-band phases after permutation.

    rebiorthonormalize : bool
        If True, reapply diagonal biorthonormalization after permutation/phase fixing.

    Returns
    -------
    spec_out : list[dict]
        Gauge-fixed copies of the input spec dicts, with reordered/phased:
            spec_out[k]["evals"], spec_out[k]["L"], spec_out[k]["R"]

    info : dict
        Diagnostics:
            "permutations"      : list of perms for each k
            "scores"            : list of neighbor matching scores
            "overlap_matrices"  : list of overlap matrices used
            "clusters"          : list of cluster lists at each propagated k
    """
    nk = len(spec_k)
    if overlaps_k.shape[0] != nk:
        raise ValueError("len(spec_k) must match overlaps_k.shape[0]")

    spec_out = []
    for sp in spec_k:
        sp_new = dict(sp)
        sp_new["evals"] = np.array(sp["evals"], dtype=np.complex128, copy=True)
        sp_new["L"] = np.array(sp["L"], dtype=np.complex128, copy=True)
        sp_new["R"] = np.array(sp["R"], dtype=np.complex128, copy=True)
        spec_out.append(sp_new)

    perms = [np.arange(spec_out[0]["R"].shape[1], dtype=int) for _ in range(nk)]
    scores = []
    overlap_mats = []
    clusters_all = [None] * nk

    # Optional initial gauge lock to previous iteration at the anchor k-point
    if previous_spec is not None:
        # O0 = (
        # previous_spec["L"].conj().T @
        # overlaps_k[start_k] @
        # spec_out[start_k]["R"]
        # )
        perm0, _ = _permute_clustered_by_subspace_overlap(
            previous_spec["L"],
            spec_out[start_k]["R"],
            overlaps_k[start_k],
            spec_out[start_k]["evals"],
            cluster_tol=cluster_tol,
        )

        spec_out[start_k]["evals"] = spec_out[start_k]["evals"][perm0]
        spec_out[start_k]["L"] = spec_out[start_k]["L"][:, perm0]
        spec_out[start_k]["R"] = spec_out[start_k]["R"][:, perm0]
        perms[start_k] = perm0

        if do_phase_alignment:
            Lfix, Rfix = _phase_align_columns(
                previous_spec["L"],
                spec_out[start_k]["R"],
                spec_out[start_k]["L"],
                overlaps_k[start_k],
            )
            spec_out[start_k]["L"] = Lfix
            spec_out[start_k]["R"] = Rfix

        if rebiorthonormalize:
            Lfix, Rfix = _biorthonormalize_columns(
                spec_out[start_k]["L"],
                spec_out[start_k]["R"],
                overlaps_k[start_k],
            )
            spec_out[start_k]["L"] = Lfix
            spec_out[start_k]["R"] = Rfix

    # Forward propagation in k
    for k in range(start_k, nk - 1):
        L_prev = spec_out[k]["L"]
        evals_curr = spec_out[k + 1]["evals"]
        L_curr = spec_out[k + 1]["L"]
        R_curr = spec_out[k + 1]["R"]
        S_curr = overlaps_k[k + 1]

        O = _neighbor_overlap(L_prev, R_curr, S_curr)
        perm, score = _permute_clustered_by_subspace_overlap(
            L_prev, R_curr, S_curr, evals_curr, cluster_tol=cluster_tol
        )

        evals_new = evals_curr[perm]
        L_new = L_curr[:, perm]
        R_new = R_curr[:, perm]

        if do_phase_alignment:
            L_new, R_new = _phase_align_columns(L_prev, R_new, L_new, S_curr)

        if rebiorthonormalize:
            L_new, R_new = _biorthonormalize_columns(L_new, R_new, S_curr)

        spec_out[k + 1]["evals"] = evals_new
        spec_out[k + 1]["L"] = L_new
        spec_out[k + 1]["R"] = R_new

        perms[k + 1] = perm
        scores.append(score)
        overlap_mats.append(O)
        clusters_all[k + 1] = _cluster_degenerate_bands(evals_new, tol=cluster_tol)

    info = {
        "permutations": perms,
        "scores": scores,
        "overlap_matrices": overlap_mats,
        "clusters": clusters_all,
    }
    return spec_out, info


# ===========================================
# Low-rank mismatch-informed factor subspace
# ===========================================


def mismatch_subspaces(
    gamma_target: np.ndarray,
    gamma_rhf: np.ndarray,
    s: np.ndarray,
    rank: int,
    rank_threshold: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build left/right low-rank subspaces from the leading singular vectors of the
    orthonormalized mismatch:
        Delta_tilde = S^{1/2} (gamma_target - gamma_rhf) S^{1/2}
    Then map singular vectors back to AO space via S^{-1/2}.

    Returns
    -------
    UL, VR : (n, rank) arrays
    """
    sq, isq = matrix_sqrt_and_invsqrt_spd(s)
    delta = gamma_target - gamma_rhf
    delta_tilde = sq @ delta @ sq
    U, vs, Vh = svd(delta_tilde, full_matrices=False)
    rank = min(rank, np.count_nonzero(vs > rank_threshold))
    r = min(rank, U.shape[1])
    print(f"Factorized operator subspace rank: {r} ({len(vs) - r} dropped)")
    print(f"Sum of discarded weights: {np.sum(vs[r:]):0.3E}")
    UL = isq @ U[:, :r]
    VR = isq @ Vh.conj().T[:, :r]
    return UL, VR


# ======================
# Penalty configuration
# ======================


@dataclass
class PenaltyConfig:
    particle_number_on: bool = True
    particle_number_lambda: float = 1.0

    commuting_on: bool = False
    commuting_lambda: float = 1.0

    spectral_anchor_on: bool = False
    spectral_anchor_lambda: float = 1.0

    noncommuting_on: bool = False
    noncommuting_lambda: float = 1.0

    parity_on: bool = False
    parity_lambda: float = 10.0
    parity_matrix: np.ndarray | None = None  # placeholder

    time_reversal_on: bool = False
    time_reversal_lambda: float = 10.0
    time_reversal_matrix: np.ndarray | None = None  # placeholder

    total_l2_lambda: float = 0.0
    antihermitian_l2_lambda: float = 0.0

    remove_identity: bool = False
    fixed_chem_pot: bool = True


# ======================
# Main optimizer object
# ======================


class HOEPOptimizer:
    def __init__(
        self,
        fock0: np.ndarray,
        overlap: np.ndarray,
        gamma_target: np.ndarray,
        hoep0: np.ndarray | None = None,
        rank: int = 2,
        working_basis: str = "AO",
        objective_type: str = "density",
        penalties: PenaltyConfig | None = None,
        default_bound: float = 1.0,
        default_beta: float = 10.0,
        verbose: bool = True,
    ):
        if working_basis.lower() in ["ao"]:
            self.fock0 = np.array(fock0, dtype=np.complex128)
            self.s = np.array(overlap, dtype=np.complex128)
            self.C_mo = np.eye(self.fock0.shape[0])
            self.working_basis = "ao"

        elif working_basis.lower() in ["mo"]:
            _F = np.array(fock0, dtype=np.complex128)
            _s = np.array(overlap, dtype=np.complex128)
            s, U = np.linalg.eigh(_s)
            X = U @ np.diag(s ** (-0.50))
            F = X.T.conj() @ (_F @ X)
            eigs, vecs = np.linalg.eigh(F, UPLO="U")
            self.fock0 = np.diag(eigs)
            self.s = np.eye(self.fock0.shape[0])
            self.C_mo = np.dot(X, vecs)
            self.working_basis = "mo"

        if objective_type.lower() in ["density"]:
            self.objective = self.density_objective
        elif objective_type.lower() in ["induced"]:
            self.objective = self.induced_objective
        else:
            raise NotImplementedError

        self.gamma_target = np.array(gamma_target, dtype=np.complex128)
        self.n = self.fock0.shape[0]
        self.n_occ = int(np.rint(np.real(np.trace(self.gamma_target @ self.s))))
        self.rank = rank
        self.penalties = penalties or PenaltyConfig()
        self.default_bound = default_bound
        self.verbose = verbose
        self.global_iter = 0

        # RHF reference density from the supplied Fock0 using a large initial beta.
        ref = density_from_hermitian_spectrum(
            self.fock0, self.s, self.C_mo, n_occ=self.n_occ, beta=default_beta
        )
        # ref = density_from_generalized_spectrum(
        # self.fock0,
        # self.s,
        # n_occ=self.n_occ,
        # beta=default_beta
        # )
        self.gamma_rhf = ref["gamma"]

        # Build mismatch-informed low-rank subspaces
        self.UL, self.VR = mismatch_subspaces(
            self.gamma_target, self.gamma_rhf, self.s, rank=self.rank
        )
        self.r = self.UL.shape[1]

        self.target_particle_number = float(
            np.real(np.trace(self.gamma_target @ self.s))
        )

        # Initialize factor matrices
        # self.AH_A = np.zeros((self.r, self.r), dtype=np.complex128)
        # self.AH_B = np.zeros((self.r, self.r), dtype=np.complex128)

        # Use projected mismatch to initialize Hermitian block
        if hoep0 is None:
            _init_H_A = (
                self.UL.conj().T
                @ self.s
                @ (self.gamma_target - self.gamma_rhf)
                @ self.s
                @ self.VR
            )
        else:
            _init_H_A = self.UL.conj().T @ self.s @ hoep0 @ self.s @ self.VR
        self.H_A = np.array(_init_H_A, dtype=np.complex128)
        self.H_B = np.eye(self.r, dtype=np.complex128)
        self.AH_A = 1.0j * self.H_A
        self.AH_B = 1.0j * self.H_B

        # Global spectral variables
        self.mu = ref["mu"]
        self.log_beta = np.log(default_beta)

    # -------------------------
    # Block parameterization
    # -------------------------

    def build_block_operator(
        self, Ared: np.ndarray, Bred: np.ndarray, sector: str
    ) -> np.ndarray:
        """
        Build low-rank AO-space operator from reduced factors.
        A = UL @ Ared
        B = VR @ Bred
        H sector:  (A B^H + B A^H)/2
        AH sector: (A B^H - B A^H)/2
        C (full) sector: A B^H
        """
        A = self.UL @ Ared
        B = self.VR @ Bred
        if sector == "H":
            return symm(A @ B.conj().T)
        elif sector == "AH":
            return anti_symm(A @ B.conj().T)
        elif sector == "C":
            return A @ B.conj().T
        else:
            raise ValueError("sector must be 'H', 'AH', or 'C'")

    def total_operator(
        self,
        H_A: np.ndarray | None = None,
        H_B: np.ndarray | None = None,
        AH_A: np.ndarray | None = None,
        AH_B: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        H_A = self.H_A if H_A is None else H_A
        H_B = self.H_B if H_B is None else H_B
        AH_A = self.AH_A if AH_A is None else AH_A
        AH_B = self.AH_B if AH_B is None else AH_B

        uH = self.build_block_operator(H_A, H_B, "H")
        if self.penalties.remove_identity:
            uH = project_out_identity(uH)
        uAH = self.build_block_operator(AH_A, AH_B, "AH")
        return uH + uAH, uH, uAH

    # -------------------------
    # Packing / unpacking
    # -------------------------

    def _pack_complex_matrix(self, X: np.ndarray) -> np.ndarray:
        return np.concatenate([np.real(X).ravel(), np.imag(X).ravel()])

    def _unpack_complex_matrix(self, x: np.ndarray) -> np.ndarray:
        m = self.r * self.r
        xr = x[:m].reshape(self.r, self.r)
        xi = x[m : 2 * m].reshape(self.r, self.r)
        return xr + 1j * xi

    def pack_block_params(
        self, Ared: np.ndarray, Bred: np.ndarray, log_beta: float
    ) -> np.ndarray:
        return np.concatenate(
            [
                self._pack_complex_matrix(Ared),
                self._pack_complex_matrix(Bred),
                np.array([log_beta], dtype=float),
            ]
        )

    def unpack_block_params(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        # m = 2 * self.r * self.r
        Ared = self._unpack_complex_matrix(x[: 2 * self.r * self.r])
        Bred = self._unpack_complex_matrix(x[2 * self.r * self.r : 4 * self.r * self.r])
        log_beta = float(x[4 * self.r * self.r])
        return Ared, Bred, log_beta

    def bounds_for_block(self, bound: float | None = None):
        b = self.default_bound if bound is None else float(bound)
        nparam = 4 * self.r * self.r + 1
        bounds = [(-b, b)] * nparam
        # log_beta
        bounds[-1] = (np.log(1e-2), np.log(1e2))
        return bounds

    # -------------------------
    # Penalties
    # -------------------------

    def penalty_value(
        self, gamma: np.ndarray, u_total: np.ndarray, uAH: np.ndarray
    ) -> float:
        pen = 0.0
        p = self.penalties

        if p.particle_number_on:
            n_eff = np.real(np.trace(gamma @ self.s))
            dn = n_eff - self.target_particle_number
            pen += p.particle_number_lambda * dn * dn

        if p.commuting_on:
            SPF = self.s @ (gamma @ u_total)
            FPS = u_total @ (gamma @ self.s)
            pen += p.commuting_lambda * frob2((SPF - FPS))

        if p.noncommuting_on:
            SPF = self.s @ (gamma @ u_total)
            FPS = u_total @ (gamma @ self.s)
            pen += p.noncommuting_lambda * 1 / (1 + frob2((SPF - FPS)))

        if p.parity_on and p.parity_matrix is not None:
            P = p.parity_matrix
            comm = u_total @ P - P @ u_total
            pen += p.parity_lambda * frob2(comm)

        if p.time_reversal_on:
            if p.time_reversal_matrix is not None:
                T = p.time_reversal_matrix
                # Placeholder antiunitary-like penalty:
                # penalize deviation from u = T u* T^{-1}
                Tin = np.linalg.inv(T)
                dev = u_total - T @ u_total.conj() @ Tin
                pen += p.time_reversal_lambda * frob2(dev)
            else:
                # Simple placeholder: in this basis, TR ~ complex conjugation
                dev = u_total - u_total.conj()
                pen += p.time_reversal_lambda * frob2(dev)

        if p.total_l2_lambda > 0.0:
            l2 = np.linalg.trace(u_total.conj().T @ self.s @ u_total @ self.s)
            pen += p.total_l2_lambda * l2**2

        if p.antihermitian_l2_lambda > 0.0:
            AH_l2 = np.linalg.trace(uAH.conj().T @ self.s @ uAH @ self.s)
            pen += p.antihermitian_l2_lambda * AH_l2**2

        return pen

    # -------------------------
    # Density response
    # -------------------------

    def density_response_for_matrix_perturbation(
        self,
        spec: dict[str, np.ndarray],
        Bmat: np.ndarray,
    ) -> np.ndarray:
        """
        Analytic 1RDM response for a perturbation in the Hamiltonian matrix:
            d gamma = sum_p dn_p P_p
                    + sum_{p!=q} (n_p-n_q)/(e_p-e_q) P_p B P_q

        In a generalized non-orthogonal basis with biorthogonality L^H S R = I:
            P_p = R_p L_p^H

        Occupations depend on the real parts of the eigenvalues only:
            dn_p = f'_p * d Re(e_p)
            d Re(e_p) = Re( L_p^H B R_p )
        """
        eigs = spec["eigs"]
        # eigs_r = spec["eigs_r"]
        L = spec["L"]
        R = spec["R"]
        occ = spec["occ"]
        df_de = spec["df_de"]

        n = len(eigs)
        gamma_resp = np.zeros((self.n, self.n), dtype=np.complex128)

        # Diagonal occupation-response term
        for p in range(n):
            Rp = R[:, p : p + 1]
            LpH = L[:, p : p + 1].conj().T
            mpp = (LpH @ Bmat @ Rp)[0, 0]
            d_eigs_real = np.real(mpp)
            dn_p = df_de[p] * d_eigs_real
            gamma_resp += dn_p * (Rp @ LpH)

        # Off-diagonal projector-response term
        for p in range(n):
            Rp = R[:, p : p + 1]
            LpH = L[:, p : p + 1].conj().T
            for q in range(n):
                if p == q:
                    continue
                dq = eigs[p] - eigs[q]
                if abs(dq) < 1e-10:
                    continue
                Rq = R[:, q : q + 1]
                LqH = L[:, q : q + 1].conj().T
                mpq = (LpH @ Bmat @ Rq)[0, 0]
                coef = (occ[p] - occ[q]) / dq
                gamma_resp += coef * mpq * (Rp @ LqH)

        return gamma_resp

    # -------------------------
    # Objective and gradient
    # -------------------------

    def density_objective(self, gamma: np.ndarray, u_total: np.ndarray):
        unused(u_total)
        residual = gamma - self.gamma_target
        return residual, frob2(residual)

    def induced_objective(self, gamma: np.ndarray, u_total: np.ndarray):
        unused(u_total)
        residual = gamma - self.gamma_target
        return residual, frob2(residual)

    def evaluate_from_blocks(
        self,
        H_A: np.ndarray,
        H_B: np.ndarray,
        AH_A: np.ndarray,
        AH_B: np.ndarray,
        log_beta: float,
    ) -> dict[str, np.ndarray]:
        beta = float(np.exp(log_beta))
        u_total, uH, uAH = self.total_operator(H_A=H_A, H_B=H_B, AH_A=AH_A, AH_B=AH_B)
        h = self.fock0 + u_total
        if self.penalties.fixed_chem_pot:
            mu = self.mu
        else:
            mu = None
        spec = density_from_generalized_spectrum(
            h, self.s, self.C_mo, n_occ=self.n_occ, beta=beta, mu=mu
        )
        gamma = spec["gamma"]

        residual, cost = self.objective(gamma, u_total)
        cost = cost + self.penalty_value(gamma, u_total, uAH)
        return {
            "h": h,
            "u_total": u_total,
            "uH": uH,
            "uAH": uAH,
            "spec": spec,
            "gamma": gamma,
            "residual": residual,
            "cost": cost,
            "mu": spec["mu"],
            "beta": beta,
        }

    def block_objective(self, x: np.ndarray, sector: str) -> float:
        Ared, Bred, log_beta = self.unpack_block_params(x)
        if sector in ["H", "C"]:
            out = self.evaluate_from_blocks(Ared, Bred, self.AH_A, self.AH_B, log_beta)
        elif sector in ["AH"]:
            out = self.evaluate_from_blocks(self.H_A, self.H_B, Ared, Bred, log_beta)
        else:
            raise ValueError("sector must be 'H' or 'AH'")
        return float(out["cost"])

    def _tangent_block_operator(
        self, deltaAred: np.ndarray, deltaBred: np.ndarray, sector: str
    ) -> np.ndarray:
        A = self.UL @ deltaAred
        if sector in ["H", "C"]:
            return 0.5 * (A @ (self.VR @ np.zeros_like(deltaBred)).conj().T)  # not used
        elif sector in ["AH"]:
            return 0.5 * (A @ (self.VR @ np.zeros_like(deltaBred)).conj().T)  # not used
        raise ValueError

    def block_gradient(self, x: np.ndarray, sector: str) -> np.ndarray:
        """
        Analytic gradient by explicit tangent matrices for each parameter.
        This is not the most efficient implementation, just a proof of concept.
        """
        Ared, Bred, log_beta = self.unpack_block_params(x)

        if sector in ["H", "C"]:
            out = self.evaluate_from_blocks(Ared, Bred, self.AH_A, self.AH_B, log_beta)
        elif sector in ["AH"]:
            out = self.evaluate_from_blocks(self.H_A, self.H_B, Ared, Bred, log_beta)

        spec = out["spec"]
        gamma = out["gamma"]
        residual = out["residual"]
        beta = out["beta"]

        # d/dgamma ||gamma-target||_F^2 = 2 Re <residual, dgamma>
        def directional_cost(dgamma: np.ndarray) -> float:
            return 2.0 * np.real(np.vdot(residual, dgamma))

        grad = np.zeros_like(x, dtype=float)
        m = self.r * self.r

        # Build current AO-space A,B for this block
        Aao = self.UL @ Ared
        Bao = self.VR @ Bred

        def sector_tangent_from_delta(
            deltaAred: np.ndarray, deltaBred: np.ndarray
        ) -> np.ndarray:
            dA = self.UL @ deltaAred
            dB = self.VR @ deltaBred
            if sector in ["H"]:
                return 0.5 * (
                    dA @ Bao.conj().T
                    + Bao @ dA.conj().T
                    + Aao @ dB.conj().T
                    + dB @ Aao.conj().T
                )
            elif sector in ["AH"]:
                return 0.5 * (
                    dA @ Bao.conj().T
                    - Bao @ dA.conj().T
                    + Aao @ dB.conj().T
                    - dB @ Aao.conj().T
                )
            elif sector in ["C"]:
                return dA @ Bao.conj().T + Aao @ dB.conj().T
            raise ValueError

        # Real/imag parts of Ared
        for idx in range(m):
            i, j = divmod(idx, self.r)

            E = np.zeros((self.r, self.r), dtype=np.complex128)
            E[i, j] = 1.0
            dU = sector_tangent_from_delta(E, np.zeros_like(E))
            dgamma = self.density_response_for_matrix_perturbation(spec, dU, beta=beta)
            g = directional_cost(dgamma)

            # Penalty derivatives
            g += self.penalty_directional_derivative(out, dU)
            grad[idx] = g

            Ei = np.zeros((self.r, self.r), dtype=np.complex128)
            Ei[i, j] = 1.0j
            dU = sector_tangent_from_delta(Ei, np.zeros_like(E))
            dgamma = self.density_response_for_matrix_perturbation(spec, dU, beta=beta)
            g = directional_cost(dgamma)
            g += self.penalty_directional_derivative(out, dU)
            grad[m + idx] = g

        # Real/imag parts of Bred
        offset = 2 * m
        for idx in range(m):
            i, j = divmod(idx, self.r)

            E = np.zeros((self.r, self.r), dtype=np.complex128)
            E[i, j] = 1.0
            dU = sector_tangent_from_delta(np.zeros_like(E), E)
            dgamma = self.density_response_for_matrix_perturbation(spec, dU, beta=beta)
            g = directional_cost(dgamma)
            g += self.penalty_directional_derivative(out, dU)
            grad[offset + idx] = g

            Ei = np.zeros((self.r, self.r), dtype=np.complex128)
            Ei[i, j] = 1.0j
            dU = sector_tangent_from_delta(np.zeros_like(E), Ei)
            dgamma = self.density_response_for_matrix_perturbation(spec, dU, beta=beta)
            g = directional_cost(dgamma)
            g += self.penalty_directional_derivative(out, dU)
            grad[offset + m + idx] = g

        occ = spec["occ"]
        L = spec["L"]
        R = spec["R"]
        mu = spec["mu"]
        eigs_r = spec["eigs_r"]

        # mu derivative
        # dgamma_dmu = np.zeros_like(gamma)
        # for p in range(len(occ)):
        #    Rp = R[:, p:p + 1]
        #    LpH = L[:, p:p + 1].conj().T
        #    df_dmu = beta * occ[p] * (1.0 - occ[p])
        #    dgamma_dmu += df_dmu * (Rp @ LpH)
        # grad[-2] = (
        #     directional_cost(dgamma_dmu)
        #     + self.penalty_scalar_derivative_mu(out, dgamma_dmu)
        #    )
        # log_beta derivative

        dgamma_dlogbeta = np.zeros_like(gamma)
        for p in range(len(occ)):
            Rp = R[:, p : p + 1]
            LpH = L[:, p : p + 1].conj().T
            # d/dlogbeta = beta * d/dbeta
            df_dlogbeta = -beta * (eigs_r[p] - mu) * occ[p] * (1.0 - occ[p])
            dgamma_dlogbeta += df_dlogbeta * (Rp @ LpH)
        grad[-1] = directional_cost(
            dgamma_dlogbeta
        ) + self.penalty_scalar_derivative_mu(out, dgamma_dlogbeta)

        return grad

    # -------------------------
    # Penalty directional derivatives
    # -------------------------

    def penalty_directional_derivative(
        self, out: dict[str, np.ndarray], dU: np.ndarray
    ) -> float:
        """
        Directional derivative of penalties wrt a Hamiltonian perturbation dU.
        Particle-number penalty depends on dgamma.
        Operator penalties depend on dU directly.
        """
        val = 0.0
        p = self.penalties
        spec = out["spec"]
        # mu = spec["mu"]
        beta = out["beta"]

        dgamma = self.density_response_for_matrix_perturbation(spec, dU, beta=beta)

        if p.particle_number_on:
            n_eff = np.real(np.trace(out["gamma"] @ self.s))
            dn = n_eff - self.target_particle_number
            ddn = np.real(np.trace(dgamma @ self.s))
            val += 2.0 * p.particle_number_lambda * dn * ddn

        if p.commuting_on:
            comm = self.s @ (out["gamma"] @ out["u_total"]) - out["u_total"] @ (
                out["gamma"] @ self.s
            )
            dcomm = self.s @ (out["gamma"] @ dU) - dU @ (out["gamma"] @ self.s)
            # comm = (out["u_total"] @ out["gamma"] -
            #        out["gamma"] @ out["u_total"])
            # dcomm = dU @ out["gamma"] - out["gamma"] @ dU
            val += 2.0 * p.commuting_lambda * np.real(np.vdot(comm, dcomm))

        if p.noncommuting_on:
            # This is incorrect...
            comm = self.s @ (out["gamma"] @ out["u_total"]) - out["u_total"] @ (
                out["gamma"] @ self.s
            )
            dcomm = self.s @ (out["gamma"] @ dU) - dU @ (out["gamma"] @ self.s)
            # comm = (out["u_total"] @ out["gamma"] -
            #        out["gamma"] @ out["u_total"])
            # dcomm = dU @ out["gamma"] - out["gamma"] @ dU
            val += 2.0 * p.noncommuting_lambda * np.real(np.vdot(comm, dcomm))

        if p.parity_on and p.parity_matrix is not None:
            P = p.parity_matrix
            comm = out["u_total"] @ P - P @ out["u_total"]
            dcomm = dU @ P - P @ dU
            val += 2.0 * p.parity_lambda * np.real(np.vdot(comm, dcomm))

        if p.time_reversal_on:
            if p.time_reversal_matrix is not None:
                T = p.time_reversal_matrix
                Tin = np.linalg.inv(T)
                dev = out["u_total"] - T @ out["u_total"].conj() @ Tin
                ddev = dU - T @ dU.conj() @ Tin
                val += 2.0 * p.time_reversal_lambda * np.real(np.vdot(dev, ddev))
            else:
                dev = out["u_total"] - out["u_total"].conj()
                ddev = dU - dU.conj()
                val += 2.0 * p.time_reversal_lambda * np.real(np.vdot(dev, ddev))

        # anti-Hermitian regularization only contributes when dU touches uAH.
        # Here we conservatively detect by comparing antihermitian part of dU.
        if p.antihermitian_l2_lambda > 0.0:
            dU_AH = anti_symm(dU)
            val += 2.0 * p.antihermitian_l2_lambda * np.real(np.vdot(out["uAH"], dU_AH))

        if p.total_l2_lambda > 0.0:
            val += 2.0 * p.total_l2_lambda * np.real(np.vdot(out["u_total"], dU))

        return float(val)

    def penalty_scalar_derivative_mu(
        self, out: dict[str, np.ndarray], dgamma_scalar: np.ndarray
    ) -> float:
        val = 0.0
        p = self.penalties
        if p.particle_number_on:
            n_eff = np.real(np.trace(out["gamma"] @ self.s))
            dn = n_eff - self.target_particle_number
            ddn = np.real(np.trace(dgamma_scalar @ self.s))
            val += 2.0 * p.particle_number_lambda * dn * ddn
        return float(val)

    # -------------------------
    # Alternating optimization
    # -------------------------

    def optimize_block(
        self,
        sector: str,
        jac_method: str = "analytic",
        maxiter: int = 100,
        anneal_bound: float | None = None,
        seed: int = 123,
    ):
        if sector in ["H", "C"]:
            x0 = self.pack_block_params(self.H_A, self.H_B, self.log_beta)
        elif sector in ["AH"]:
            x0 = self.pack_block_params(self.AH_A, self.AH_B, self.log_beta)
        else:
            raise ValueError("sector must be 'H', 'AH', or 'C'")

        bounds = self.bounds_for_block(anneal_bound)

        obj: callable[[np.ndarray], float] = lambda x: self.block_objective(  # noqa: E731
            x, sector=sector
        )
        if jac_method.lower() in ["analytic"]:
            grad: callable[[np.ndarray], np.ndarray] = lambda x: self.block_gradient(  # noqa: E731
                x, sector=sector
            )
        else:
            grad = jac_method

        ret = dual_annealing(
            obj,
            bounds=bounds,
            x0=x0,
            seed=seed,
            maxiter=maxiter,
            callback=self.annealing_callback,
            minimizer_kwargs={
                "method": "L-BFGS-B",
                "jac": grad,
                "bounds": bounds,
            },
            no_local_search=False,
        )

        Ared, Bred, log_beta = self.unpack_block_params(ret.x)
        if sector in ["H", "C"]:
            self.H_A, self.H_B = Ared, Bred
        else:
            self.AH_A, self.AH_B = Ared, Bred
        self.log_beta = log_beta

        if self.verbose:
            print(f"Sector: [{sector}], best cost = {ret.fun:.8e}")

        return ret

    def quasiparticle_optimize(
        self,
        opt_method: int = 0,
        tol_QP: float = 1e-4,
        maxiter_QP: int = 10,
        jac_method: str = "analytic",
        cycles: int = 4,
        maxiter_H: int = 200,
        maxiter_AH: int = 200,
        anneal_bound: float | None = None,
        seed: int = 123,
    ):

        if opt_method == 0:
            opt_function = self.alternating_optimize
        elif opt_method == 1:
            opt_function = self.full_optimize

        history = opt_function(
            jac_method=jac_method,
            cycles=cycles,
            maxiter_H=maxiter_H,
            maxiter_AH=maxiter_AH,
            anneal_bound=anneal_bound,
            seed=seed,
        )
        init_out = self.current_state()
        _fock0 = init_out["h"]

        for iter_QP in range(0, maxiter_QP):
            if self.verbose:
                print(f"\n=== QP Opt Cycle {iter_QP + 1} ===")
            iter_opt = HOEPOptimizer(
                fock0=_fock0,
                overlap=self.s,
                gamma_target=self.gamma_target,
                rank=self.rank,
                penalties=self.penalties,
                working_basis=self.working_basis,
                default_bound=self.default_bound,
                default_beta=self.log_beta,
                verbose=self.verbose,
            )
            if opt_method == 0:
                _opt_function = iter_opt.alternating_optimize
            elif opt_method == 1:
                _opt_function = iter_opt.full_optimize
            _history = _opt_function(
                jac_method=jac_method,
                cycles=cycles,
                maxiter_H=maxiter_H,
                maxiter_AH=maxiter_AH,
                anneal_bound=anneal_bound,
                seed=seed,
            )
            history.append(entry for entry in _history)
            iter_out = iter_opt.current_state()
            iter_delta = frob2(iter_out["h"] - _fock0)
            print(f"QP Iter {iter_QP + 1}: ||delta H^QP||={iter_delta:0.4E}")
            if iter_delta <= tol_QP:
                print("Converged?")
                self.current_state = iter_opt.current_state
                self.H_A = iter_opt.H_A
                self.H_B = iter_opt.H_B
                self.AH_A = iter_opt.AH_A
                self.AH_B = iter_opt.AH_B
                self.log_beta = iter_opt.log_beta
                return history
            else:
                _fock0 = iter_out["h"]
        return history

    def alternating_optimize(
        self,
        jac_method: str = "analytic",
        cycles: int = 4,
        maxiter_H: int = 200,
        maxiter_AH: int = 200,
        anneal_bound: float | None = None,
        seed: int = 123,
    ):
        history = []
        for cyc in range(0, cycles):
            if self.verbose:
                print(f"\n=== H/AH Alt cycle {cyc + 1}/{cycles} ===")
            retH = self.optimize_block(
                "H",
                jac_method=jac_method,
                maxiter=maxiter_H,
                anneal_bound=anneal_bound,
                seed=seed + 2 * cyc,
            )
            retA = self.optimize_block(
                "AH",
                jac_method=jac_method,
                maxiter=maxiter_AH,
                anneal_bound=anneal_bound,
                seed=seed + 2 * cyc + 1,
            )
            out = self.current_state()
            history.append(
                {
                    "cycle": cyc + 1,
                    "cost": out["cost"],
                    "N_eff": np.real(np.trace(out["gamma"] @ self.s)),
                    "mu": self.mu,
                    "beta": np.exp(self.log_beta),
                    "retH": retH,
                    "retAH": retA,
                }
            )
            if self.verbose:
                print(
                    f"Cycle {cyc + 1}: cost={out['cost']:.8e},",
                    f"N_eff={history[-1]['N_eff']:.6f}, beta={history[-1]['beta']:.6f}",
                )
        return history

    def full_optimize(
        self,
        jac_method: str = "analytic",
        maxiter: int = 100,
        anneal_bound: float | None = None,
        seed: int = 123,
    ):
        self.AH_A = np.zeros((self.r, self.r), dtype=np.complex128)
        self.AH_B = np.zeros((self.r, self.r), dtype=np.complex128)
        history = []
        retH = self.optimize_block(
            "C",
            jac_method=jac_method,
            maxiter=maxiter,
            anneal_bound=anneal_bound,
            seed=seed,
        )
        out = self.current_state()
        history.append(
            {
                "cycle": None,
                "cost": out["cost"],
                "N_eff": np.real(np.trace(out["gamma"] @ self.s)),
                "mu": self.mu,
                "beta": np.exp(self.log_beta),
                "retH": retH,
                "retAH": None,
            }
        )
        if self.verbose:
            print(
                f"Complex Opt: cost={out['cost']:.8e},",
                f"N_eff={history[-1]['N_eff']:.6f},",
                f"beta={history[-1]['beta']:.6f}",
            )
        return history

    def current_state(self):
        return self.evaluate_from_blocks(
            self.H_A, self.H_B, self.AH_A, self.AH_B, self.log_beta
        )

    def annealing_callback(self, x, f, context):
        print(
            f"Annealing Iter. {self.global_iter}: e_val={f:0.3E} (Context: {context})"
        )
        print(f" -x_val: {x:0.3E}")
        self.global_iter += 1
