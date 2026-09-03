# Author(s): Shaun Weatherly, Oinam Romesh Meitei
from __future__ import annotations

import numpy as np
from numpy.linalg import multi_dot
from pyscf import scf

from quemb.shared.helper import unused


def pack_block(M, mask):
    """ """
    k = 0
    _M = M[mask]
    r = np.shape(_M)[0]
    c = np.shape(_M)[1]
    v = np.empty((2 * r * c), dtype=np.float64)
    for i in range(r):
        for j in range(c):
            hij = _M[i, j]
            v[k] = hij.real
            v[k + 1] = hij.imag
            k += 2
    return v


def unpack_block(v, n, mask, M=None):
    """ """
    if M is None:
        M = np.zeros((n, n), dtype=np.complex128)
    else:
        _n, _m = np.shape(M)[0], np.shape(M)[1]
        assert _n == _m
        assert _n == n

    k = 0
    _M = np.zeros_like(M[mask], dtype=np.complex128)
    r = np.shape(_M)[0]
    c = np.shape(_M)[1]
    # Off-diagonal block
    for i in range(r):
        for j in range(c):
            re = v[k]
            im = v[k + 1]
            _M[i, j] = re + 1j * im
            k += 2
    M[mask] = _M[:, :]
    # M[nocc:n, :nocc] = _M.conj().T[:,:]
    return M


def pack_block_hermitian(M, nocc):
    """
    Packs the occupied-virtual block of M into a real vector of length:
        2 * (nocc * nvir)
    """
    n = M.shape[0]  # Assumes (n x n)
    k = 0
    nvir = n - nocc
    _M = M[:nocc, nocc:n]
    v = np.empty((2 * nocc * nvir), dtype=np.float64)
    for i in range(nocc):
        for j in range(nvir):
            hij = _M[i, j]
            v[k] = hij.real
            v[k + 1] = hij.imag
            k += 2
    return v


def unpack_block_hermitian(v, n, nocc, M=None):
    """
    Rebuild Hermitian matrix from packed occupied-virtual block.
    """
    if M is None:
        M = np.zeros((n, n), dtype=np.complex128)
    else:
        _n, _m = np.shape(M)[0], np.shape(M)[1]
        assert _n == _m
        assert _n == n

    k = 0
    nvir = n - nocc
    _M = np.zeros((nocc, nvir), dtype=np.complex128)

    # Off-diagonal block
    for i in range(nocc):
        for j in range(nvir):
            re = v[k]
            im = v[k + 1]
            _M[i, j] = re + 1j * im
            k += 2
    M[:nocc, nocc:n] = _M[:, :]
    M[nocc:n, :nocc] = _M.conj().T[:, :]
    return M


# def pack_block_nonhermitian(M, nocc):
#     return None


# def unpack_block_nonhermitian(v, n, nocc):
#     return None


def pack_triu_hermitian(M, *args):
    """
    Packs an n×n Hermitian matrix into a real vector of length:
        n + 2 * (n*(n-1)//2)
    Order:
      - First n diagonal entries, H_ii (purely real)
      - Then (for i<j): Re(H_ij) -> Im(H_ij)
    """
    unused(args)
    n = M.shape[0]
    k = 0
    v = np.empty((n * (n - 1)), dtype=np.float64)  # = 2*(n*(n-1)/2)
    # Upper triangle (complex)
    for i in range(n):
        for j in range(i + 1, n):
            hij = M[i, j]
            v[k] = hij.real
            v[k + 1] = hij.imag
            k += 2
    return v


def pack_hermitian(M, *args):
    """
    Packs an n×n Hermitian matrix into a real vector of length:
        n + 2 * (n*(n-1)//2)
    Order:
      - First n diagonal entries, H_ii (purely real)
      - Then (for i<j): Re(H_ij) -> Im(H_ij)
    """
    unused(args)
    n = M.shape[0]
    k = 0
    v = np.empty(n + (n * (n - 1)), dtype=np.float64)  # = n + 2*(n*(n-1)/2)
    # Diagonal (real)
    for i in range(n):
        v[k] = M[i, i].real
        k += 1
    # Upper triangle (complex)
    for i in range(n):
        for j in range(i + 1, n):
            hij = M[i, j]
            v[k] = hij.real
            v[k + 1] = hij.imag
            k += 2
    return v


def unpack_triu_hermitian(v, n, *args):
    """
    Rebuild Hermitian matrix from packed representation.
    """
    unused(args)
    M = np.zeros((n, n), dtype=np.complex128)
    k = 0

    # Hermitian off-diagonals
    for i in range(n):
        for j in range(i + 1, n):
            re = v[k]
            im = v[k + 1]
            M[i, j] = re + 1j * im
            M[j, i] = re - 1j * im
            k += 2

    return M


def unpack_hermitian(v, n, *args):
    """
    Rebuild Hermitian matrix from packed representation.
    """
    unused(args)
    M = np.zeros((n, n), dtype=np.complex128)
    k = 0

    # Unpack hermitian
    for i in range(n):
        M[i, i] = v[k] + 0.0j
        k += 1

    # Off-diagonal
    for i in range(n):
        for j in range(i + 1, n):
            re = v[k]
            im = v[k + 1]
            M[i, j] = re + 1j * im
            M[j, i] = re - 1j * im
            k += 2

    return M


def pack_nonhermitian(M, *args):
    """
    Packs an n×n square matrix into a real vector of length:
        2 * (n + n*(n-1))
    (i.e., exactly twice the length of the packed Hermitian vector)
    Order:
      - Re(H_ii) -> Im(H_ii) for all 'n' diagonals
      - Then (for i<j): Re(H_ij) -> Im(H_ij) -> Re(H_ji) -> Im(H_ji)
    """
    unused(args)
    n = M.shape[0]
    v = np.empty(2 * (n + (n * (n - 1))), dtype=np.float64)
    k = 0

    # Diagonal (complex)
    for i in range(n):
        v[k] = M[i, i].real
        v[k + 1] = M[i, i].imag
        k += 2
    # Upper and Lower Triangles
    for i in range(n):
        for j in range(i + 1, n):
            hij = M[i, j]
            hji = M[j, i]
            v[k] = hij.real
            v[k + 1] = hij.imag
            v[k + 2] = hji.real
            v[k + 3] = hji.imag
            k += 4

    return v


def block_split(M, nrows, ncols=None):
    """
    Split a matrix into sub-matrices.
    """
    if ncols is None:
        ncols = nrows
    r, h = M.shape
    return (
        M.reshape(h // nrows, nrows, -1, ncols).swapaxes(1, 2).reshape(-1, nrows, ncols)
    )


def unpack_nonhermitian(v, n, *args):
    """
    Rebuild square matrix from packed representation.
    """
    unused(args)
    M = np.zeros((n, n), dtype=np.complex128)
    k = 0

    # Diagonal
    if len(v) == 2 * (n + (n * (n - 1))):
        for i in range(n):
            M[i, i] = v[k] + v[k + 1] * 1j
            k += 2

    # Off-diagonal
    for i in range(n):
        for j in range(i + 1, n):
            # re = v[k]
            # im = v[k + 1]
            M[i, j] = v[k] + 1j * v[k + 1]
            M[j, i] = v[k + 2] + 1j * v[k + 3]
            k += 4

    return M


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


def unwrap_n_tensor(arr):
    _dims = np.shape(arr)
    if len(_dims) == 2:
        # Good to go, do nothing.
        pass
    elif len(_dims) == 3:
        # Assume broadcasting over the first index to flatten
        # a 3-D array into 2-dimensions.
        arr = unwrap_3_tensor(arr)
    elif len(_dims) == 4:
        # Simply reshape into a 2-dimensional array.
        arr = unwrap_4_tensor(arr)
    elif len(_dims) >= 5:
        raise SystemExit("Can't unwrap n-tensors for `n>=5`!")
    r_a = np.real(arr)
    # i_arr = np.imag(arr)
    return r_a


def unwrap_3_tensor(arr):
    # Assume broadcasting over the first index.
    _dims = np.shape(arr)
    _arr = np.zeros((_dims[0] * _dims[1], _dims[0] * _dims[2]), dtype=arr.dtype)
    for i in range(_dims[0]):
        _ofs = (i * _dims[1], i * _dims[2])
        _arr[_ofs[0] : _ofs[0] + _dims[1], _ofs[1] : _ofs[1] + _dims[2]] = arr[i][:, :]
    return _arr


def unwrap_4_tensor(arr):
    _dims = np.shape(arr)
    _arr = np.reshape(arr, (_dims[0] * _dims[1], _dims[2] * _dims[3]), order="C")
    return _arr
