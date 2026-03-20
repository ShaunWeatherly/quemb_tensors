# Author(s): Oinam Romesh Meitei
from __future__ import annotations

import numpy as np
from numpy.linalg import multi_dot
from pyscf import scf

from quemb.shared.helper import unused


def set_fermi(
    e_kn: np.ndarray,
    n_electron: int,
    vbmax: float = -99.0,
    cbmin: float = 99.0,
):
    _cbmin = cbmin
    _vbmax = vbmax
    _cbm_kidx = 0
    _vbm_kidx = 0
    for kidx, en in enumerate(e_kn):
        cb_k = en[n_electron // 2]
        if cb_k < _cbmin:
            _cbmin = cb_k
            _cbm_kidx = kidx
        vb_k = en[n_electron // 2 - 1]
        if vb_k > _vbmax:
            _vbmax = vb_k
            _vbm_kidx = kidx
    e_kn = [en - _vbmax for en in e_kn]
    band_summary = {
        "cbmin": _cbmin,
        "cbmin_kidx": _cbm_kidx,
        "vbmax": _vbmax,
        "vbmax_kidx": _vbm_kidx,
        "gap": np.abs(_cbmin - _vbmax),
    }

    return (e_kn, band_summary)


def get_bands(
    hcore: np.ndarray,
    v_eff: np.ndarray,
    ovlp: np.ndarray,
    nkpts_band: int,
    u_corr: np.ndarray | None = None,
) -> tuple:
    """
    Compute energy bands from an effective one-body Hamiltonian.

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
    mo_coeff_kpts = []
    for k in range(0, nkpts_band):
        s, U = np.linalg.eigh(ovlp[k])
        X = U @ np.diag(s ** (-0.50))
        F = X.T.conj() @ (fock[k] @ X)
        eigs, vecs = np.linalg.eigh(F, UPLO="U")
        idx = np.argmax(abs(vecs.real), axis=0)
        C_mo = np.dot(X, vecs)
        C_mo[:, C_mo[idx, np.arange(len(eigs))].real < 0] *= -1
        eig_kpts.append(eigs)
        mo_coeff_kpts.append(C_mo)

    return (eig_kpts, mo_coeff_kpts)


def get_veff(eri_, dm, S, TA, hf_veff, return_veff0=False):
    """
    Calculate the effective HF potential (Veff) for a given density matrix
    and electron repulsion integrals.

    This function computes the effective potential by transforming the density matrix,
    computing the Coulomb (J) and exchange (K) integrals.

    Parameters
    ----------
    eri_ : numpy.ndarray
        Electron repulsion integrals.
    dm : numpy.ndarray
        Density matrix. 2D array.
    S : numpy.ndarray
        Overlap matrix.
    TA : numpy.ndarray
        Transformation matrix.
    hf_veff : numpy.ndarray
        Hartree-Fock effective potential for the full system.

    """

    # construct rdm
    nk, nao, neo = TA.shape
    unused(nao)
    P_ = np.zeros((neo, neo), dtype=np.complex128)
    for k in range(nk):
        Cinv = TA[k].conj().T @ S[k]
        P_ += multi_dot((Cinv, dm[k], Cinv.conj().T))
    P_ /= float(nk)

    P_ = np.asarray(P_.real, dtype=np.float64)

    eri_ = np.asarray(eri_, dtype=np.float64)
    vj, vk = scf.hf.dot_eri_dm(eri_, P_, hermi=1, with_j=True, with_k=True)
    Veff_ = vj - 0.5 * vk

    # remove core contribution from hf_veff

    Veff0 = np.zeros((neo, neo), dtype=np.complex128)
    for k in range(nk):
        Veff0 += multi_dot((TA[k].conj().T, hf_veff[k], TA[k]))
    Veff0 /= float(nk)

    Veff = Veff0 - Veff_

    if return_veff0:
        return (Veff0, Veff)

    return Veff
