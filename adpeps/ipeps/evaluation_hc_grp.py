import cmath

import jax.numpy as np
import numpy as onp
import scipy.linalg as linalg
from jax import random

import adpeps.ipeps.config as sim_config
from adpeps.tensor.contractions import ncon
from adpeps.utils.empty_tensor import EmptyT
from adpeps.utils.nested import Nested
from adpeps.utils.printing import print
from adpeps.utils.tlist import TList, cur_loc, set_pattern
from .models.common import sigmam, sigmap, sigmaz, id2
from .models.hamiltonian import Hamiltonian

"""
    Evaluation module for iPEPS simulations

    This module contains the contractions of the reduced density matrices 
    and the computation of the expectation values for iPEPS ground- and 
    excited states
"""


def get_gs_energy(H, tensors):
    """Returns ground-state energy and norm of the iPEPS"""
    E, nrm, *_ = get_obs(H, tensors, measure_obs=False)
    return E[0], nrm


def get_gs_energy_bondwise(H, tensors):
    """Returns ground-state energy and norm of the iPEPS"""
    E, nrm, _, E0s = get_obs(H, tensors, measure_obs=False)
    return E[0], nrm, E0s


def get_all_energy(H, tensors):
    """Returns only energy and norm of the iPEPS"""
    E, nrm, *_ = get_obs(H, tensors, measure_obs=False)
    return E


def get_obs(H, tensors, measure_obs=True, only_gs=False):
    """Returns the energy and norm of the state

    The energy will be returned as a `Nested` tensor

    More observables can be added here
    """
    A = tensors.A
    Ad = tensors.Ad
    # Ehs = TList(shape=A.size, pattern=A.pattern)  # Horizontal terms
    # Evs = TList(shape=A.size, pattern=A.pattern)  # Vertical terms
    # Eds = TList(shape=A.size, pattern=A.pattern)  # Diagonal terms
    # En2x1ys = TList(shape=A.size, pattern=A.pattern)  # Horizontal terms
    # En1x2ys = TList(shape=A.size, pattern=A.pattern)  # Vertical terms
    # E1x1ys = TList(shape=A.size, pattern=A.pattern)  # Diagonal terms
    # Ehs_exci = TList(shape=A.size, pattern=A.pattern)  # Horizontal terms
    # Evs_exci = TList(shape=A.size, pattern=A.pattern)  # Vertical terms
    # Eds_exci = TList(shape=A.size, pattern=A.pattern)  # Diagonal terms
    # nrmhs = TList(shape=A.size, pattern=A.pattern)  # Horizontal terms
    # nrmvs = TList(shape=A.size, pattern=A.pattern)  # Vertical terms
    # nrmds = TList(shape=A.size, pattern=A.pattern)  # Diagonal terms
    # nrmn2x1ys = TList(shape=A.size, pattern=A.pattern)  # NNN_3x2 terms
    # nrmn1x2ys = TList(shape=A.size, pattern=A.pattern)  # NNN_2x3 terms
    # nrm1x1ys = TList(shape=A.size, pattern=A.pattern)  # NNN Diagonal
    # # obs_evs = [TList(shape=A.size, pattern=A.pattern) for _ in tensors.observables]
    # obs_evs = [TList(shape=A.size, pattern=A.pattern) for _ in [sigmaz, sigmap, sigmam]]

    #                   a1 -- b1
    #                 / |   /
    #                /  |  /                          ------ T1           ----- (1,0)
    #       a0 -- b0 -- c1                           /       /           /       /
    #       |   / |   / |              =>    T0(a,b,c)      /          (0,0)    /
    #       |  /  |  /  |                          /       T3          /      (1,1)
    #       c0 -- a3 -- b3                        /       /           /       /
    #       |   / |   /                          T2 ------          (0,1) ----
    #       |  /  |  /
    # a2 -- b2 -- b3
    # |   /
    # |  /
    # c2

    E0a0bs = TList(shape=A.size, pattern=A.pattern)
    E0b1as = TList(shape=A.size, pattern=A.pattern)
    E0b2as = TList(shape=A.size, pattern=A.pattern)

    E0a1as = TList(shape=A.size, pattern=A.pattern)
    E0a2as = TList(shape=A.size, pattern=A.pattern)
    E1a2as = TList(shape=A.size, pattern=A.pattern)
    E0b1bs = TList(shape=A.size, pattern=A.pattern)
    E0b2bs = TList(shape=A.size, pattern=A.pattern)
    E1b2bs = TList(shape=A.size, pattern=A.pattern)

    E0a3bs = TList(shape=A.size, pattern=A.pattern)
    E1a2bs = TList(shape=A.size, pattern=A.pattern)
    E1b2as = TList(shape=A.size, pattern=A.pattern)

    nrm0a0bs = TList(shape=A.size, pattern=A.pattern)
    nrm0b1as = TList(shape=A.size, pattern=A.pattern)
    nrm0b2as = TList(shape=A.size, pattern=A.pattern)

    nrm0a1as = TList(shape=A.size, pattern=A.pattern)
    nrm0a2as = TList(shape=A.size, pattern=A.pattern)
    nrm1a2as = TList(shape=A.size, pattern=A.pattern)
    nrm0b1bs = TList(shape=A.size, pattern=A.pattern)
    nrm0b2bs = TList(shape=A.size, pattern=A.pattern)
    nrm1b2bs = TList(shape=A.size, pattern=A.pattern)

    nrm0a3bs = TList(shape=A.size, pattern=A.pattern)
    nrm1a2bs = TList(shape=A.size, pattern=A.pattern)
    nrm1b2as = TList(shape=A.size, pattern=A.pattern)

    # obs_evs = [TList(shape=A.size, pattern=A.pattern) for _ in [sigmaz, sigmap, sigmam]]
    obs_evs = {s: [TList(shape=A.size, pattern=A.pattern) for _ in [sigmaz, sigmap, sigmam]] for s in ['a', 'b']}

    hpattern = np.array([[0, 1]])
    E0s = Hamiltonian(pattern=hpattern)

    for i in A.x_major():
        with cur_loc(i):
            if not E0a0bs.is_changed(0, 0):
                # Construct all the 2-body density matrices individually for the energy evaluation
                if abs(sim_config.model_params['J2']) > 0:
                    if abs(sim_config.model_params['J3']) > 0:
                        ro0a0b, ro0b1a, ro0b2a, ro0a1a, ro0a2a, ro1a2a, ro0b1b, ro0b2b, ro1b2b, ro0a3b, ro1a2b, ro1b2a = get_dms(tensors)
                    else:
                        ro0a0b, ro0b1a, ro0b2a, ro0a1a, ro0a2a, ro1a2a, ro0b1b, ro0b2b, ro1b2b = get_dms(tensors)
                else:
                    if abs(sim_config.model_params['J3']) > 0:
                        ro0a0b, ro0b1a, ro0b2a, ro0a3b, ro1a2b, ro1b2a = get_dms(tensors)
                    else:
                        ro0a0b, ro0b1a, ro0b2a = get_dms(tensors)
                nrm0a0b = np.trace(np.reshape(ro0a0b[0], (4, 4))).real
                nrm0b1a = np.trace(np.reshape(ro0b1a[0], (4, 4))).real
                nrm0b2a = np.trace(np.reshape(ro0b2a[0], (4, 4))).real

                nrm0a0bs[0, 0] = nrm0a0b
                nrm0b1as[0, 0] = nrm0b1a
                nrm0b2as[0, 0] = nrm0b2a
                ro0a0b = ro0a0b / nrm0a0b
                ro0b1a = ro0b1a / nrm0b1a
                ro0b2a = ro0b2a / nrm0b2a

                E0a0bs[0, 0] = ncon([ro0a0b, H[(0, 0), (1, 1)]], ([1,2,3,4], [1,2,3,4])).real
                E0b1as[0, 0] = ncon([ro0b1a, H[(1, 1), (2, 1)]], ([1,2,3,4], [1,2,3,4])).real
                E0b2as[0, 0] = ncon([ro0b2a, H[(1, 1), (1, 2)]], ([1,2,3,4], [1,2,3,4])).real
                E0s[((0, 0), (1, 1))] = E0a0bs[0, 0][0]
                E0s[((1, 1), (2, 1))] = E0b1as[0, 0][0]
                E0s[((1, 1), (1, 2))] = E0b2as[0, 0][0]

                if abs(sim_config.model_params['J2']) > 0:
                    nrm0a1a = np.trace(np.reshape(ro0a1a[0], (4, 4))).real
                    nrm0a2a = np.trace(np.reshape(ro0a2a[0], (4, 4))).real
                    nrm1a2a = np.trace(np.reshape(ro1a2a[0], (4, 4))).real
                    nrm0b1b = np.trace(np.reshape(ro0b1b[0], (4, 4))).real
                    nrm0b2b = np.trace(np.reshape(ro0b2b[0], (4, 4))).real
                    nrm1b2b = np.trace(np.reshape(ro1b2b[0], (4, 4))).real

                    nrm0a1as[0, 0] = nrm0a1a
                    nrm0a2as[0, 0] = nrm0a2a
                    nrm1a2as[0, 0] = nrm1a2a
                    nrm0b1bs[0, 0] = nrm0b1b
                    nrm0b2bs[0, 0] = nrm0b2b
                    nrm1b2bs[0, 0] = nrm1b2b
                    ro0a1a = ro0a1a / nrm0a1a
                    ro0a2a = ro0a2a / nrm0a2a
                    ro1a2a = ro1a2a / nrm1a2a
                    ro0b1b = ro0b1b / nrm0b1b
                    ro0b2b = ro0b2b / nrm0b2b
                    ro1b2b = ro1b2b / nrm1b2b

                    E0a1as[0, 0] = ncon([ro0a1a, H[(0, 0), (2, 1)]], ([1, 2, 3, 4], [1, 2, 3, 4])).real
                    E0a2as[0, 0] = ncon([ro0a2a, H[(0, 0), (1, 2)]], ([1, 2, 3, 4], [1, 2, 3, 4])).real
                    E1a2as[0, 0] = ncon([ro1a2a, H[(0, 0), (-1, 1)]], ([1, 2, 3, 4], [1, 2, 3, 4])).real
                    E0b1bs[0, 0] = ncon([ro0b1b, H[(1, 1), (3, 2)]], ([1, 2, 3, 4], [1, 2, 3, 4])).real
                    E0b2bs[0, 0] = ncon([ro0b2b, H[(1, 1), (2, 3)]], ([1, 2, 3, 4], [1, 2, 3, 4])).real
                    E1b2bs[0, 0] = ncon([ro1b2b, H[(1, 1), (0, 2)]], ([1, 2, 3, 4], [1, 2, 3, 4])).real
                    E0s[((0, 0), (2, 1))] = E0a1as[0, 0][0]
                    E0s[((0, 0), (1, 2))] = E0a2as[0, 0][0]
                    E0s[((0, 0), (-1, 1))] = E1a2as[0, 0][0]
                    E0s[((1, 1), (3, 2))] = E0b1bs[0, 0][0]
                    E0s[((1, 1), (2, 3))] = E0b2bs[0, 0][0]
                    E0s[((1, 1), (0, 2))] = E1b2bs[0, 0][0]

                if abs(sim_config.model_params['J3']) > 0:
                    nrm0a3b = np.trace(np.reshape(ro0a3b[0], (4, 4))).real
                    nrm1a2b = np.trace(np.reshape(ro1a2b[0], (4, 4))).real
                    nrm1b2a = np.trace(np.reshape(ro1b2a[0], (4, 4))).real

                    nrm0a3bs[0, 0] = nrm0a3b
                    nrm1a2bs[0, 0] = nrm1a2b
                    nrm1b2as[0, 0] = nrm1b2a
                    ro0a3b = ro0a3b / nrm0a3b
                    ro1a2b = ro1a2b / nrm1a2b
                    ro1b2a = ro1b2a / nrm1b2a

                    E0a3bs[0, 0] = ncon([ro0a3b, H[(1, 1), (3, 3)]], ([1, 2, 3, 4], [1, 2, 3, 4])).real
                    E1a2bs[0, 0] = ncon([ro1a2b, H[(0, 0), (0, 2)]], ([1, 2, 3, 4], [1, 2, 3, 4])).real
                    E1b2as[0, 0] = ncon([ro1b2a, H[(0, 0), (2, 0)]], ([1, 2, 3, 4], [1, 2, 3, 4])).real
                    E0s[((1, 1), (3, 3))] = E0a3bs[0, 0][0]
                    E0s[((0, 0), (0, 2))] = E1a2bs[0, 0][0]
                    E0s[((0, 0), (2, 0))] = E1b2as[0, 0][0]

                if measure_obs:
                    for s in ['a', 'b']:
                        ro_1site = get_one_phys_site_dm(tensors.Cs,tensors.Ts,A,Ad,stk=s)
                        for obs_i, obs in enumerate([(sigmap+sigmam)/2, 1j*(sigmam-sigmap)/2, sigmaz/2]):
                            try:
                                obs_ev = ncon([ro_1site, obs], ([1, 2], [1, 2]))
                                norm = ncon([ro_1site, id2], ([1, 2], [1, 2]))
                                obs_evs[s][obs_i][0, 0] = obs_ev / norm
                            except:
                                obs_evs[s][obs_i][0, 0] = np.nan

    if abs(sim_config.model_params['J2']) > 0:
        if abs(sim_config.model_params['J3']) > 0:
            E = (E0a0bs.mean() + E0b1as.mean() + E0b2as.mean()
                 + E0a1as.mean() + E0a2as.mean() + E1a2as.mean() + E0b1bs.mean() + E0b2bs.mean() + E1b2bs.mean()
                 + E0a3bs.mean() + E1a2bs.mean() + E1b2as.mean()) / 12.
            nrm = (nrm0a0bs.mean() + nrm0b1as.mean() + nrm0b2as.mean()
                 + nrm0a1as.mean() + nrm0a2as.mean() + nrm1a2as.mean() + nrm0b1bs.mean() + nrm0b2bs.mean() + nrm1b2bs.mean()
                 + nrm0a3bs.mean() + nrm1a2bs.mean() + nrm1b2as.mean()) / 12.
        else:
            E = (E0a0bs.mean() + E0b1as.mean() + E0b2as.mean()
                 + E0a1as.mean() + E0a2as.mean() + E1a2as.mean() + E0b1bs.mean() + E0b2bs.mean() + E1b2bs.mean()) / 9.
            nrm = (nrm0a0bs.mean() + nrm0b1as.mean() + nrm0b2as.mean()
                 + nrm0a1as.mean() + nrm0a2as.mean() + nrm1a2as.mean() + nrm0b1bs.mean() + nrm0b2bs.mean() + nrm1b2bs.mean()) / 9.
    else:
        if abs(sim_config.model_params['J3']) > 0:
            E = (E0a0bs.mean() + E0b1as.mean() + E0b2as.mean() + E0a3bs.mean() + E1a2bs.mean() + E1b2as.mean()) / 6.
            nrm = (nrm0a0bs.mean() + nrm0b1as.mean() + nrm0b2as.mean() + nrm0a3bs.mean() + nrm1a2bs.mean() + nrm1b2as.mean()) / 6.
        else:
            E = (E0a0bs.mean() + E0b1as.mean() + E0b2as.mean()) / 3.
            nrm = (nrm0a0bs.mean() + nrm0b1as.mean() + nrm0b2as.mean()) / 3.

    return E, nrm, obs_evs, E0s


def compute_exci_norm(tensors):
    """Returns the norm of the excited state based on a one-site
    environment

    Averaged over sites in the unit cell
    """
    A = tensors.A
    nrms = TList(shape=A.size, pattern=A.pattern)
    nrms_gs = TList(shape=A.size, pattern=A.pattern)
    envBs = TList(shape=A.size, pattern=A.pattern)

    for i in A.x_major():
        with cur_loc(i):
            if not nrms.is_changed(0, 0):
                nrm, nrm_gs, envB = _compute_one_site_exci_norm(tensors)
                # Exci norm
                nrms[0, 0] = nrm
                # Ground state norm
                nrms_gs[0, 0] = nrm_gs
                # Environment (exci norm without center Bd)
                envBs[0, 0] = envB
    return nrms.mean(), nrms_gs.mean(), envBs, nrms_gs


def _compute_one_site_exci_norm(ts):
    """Returns the norm of the excited state for one site in the
    unit cell
    """

    def get_single_site_dm(C1, T1, C2, T2, C3, T3, C4, T4):
        return ncon((C2, T1, C1, T4, C4, T3, C3, T2), "dm_single_site")

    n_tensors = [
        ts.Cs[0][-1, -1],
        ts.Ts[0][0, -1],
        ts.Cs[1][1, -1],
        ts.Ts[1][1, 0],
        ts.Cs[2][1, 1],
        ts.Ts[2][0, 1],
        ts.Cs[3][-1, 1],
        ts.Ts[3][-1, 0],
    ]
    B_tensors = [
        ts.B_Cs[0][-1, -1],
        ts.B_Ts[0][0, -1],
        ts.B_Cs[1][1, -1],
        ts.B_Ts[1][1, 0],
        ts.B_Cs[2][1, 1],
        ts.B_Ts[2][0, 1],
        ts.B_Cs[3][-1, 1],
        ts.B_Ts[3][-1, 0],
    ]
    Bd_tensors = [
        ts.Bd_Cs[0][-1, -1],
        ts.Bd_Ts[0][0, -1],
        ts.Bd_Cs[1][1, -1],
        ts.Bd_Ts[1][1, 0],
        ts.Bd_Cs[2][1, 1],
        ts.Bd_Ts[2][0, 1],
        ts.Bd_Cs[3][-1, 1],
        ts.Bd_Ts[3][-1, 0],
    ]

    # Compute the ground state one-site reduced density matrix
    n_dm = get_single_site_dm(*n_tensors)
    nrm0 = ncon(
        (ts.A[0, 0], ts.Ad[0, 0], n_dm),
        ([1, 2, 3, 4, 5], [1, 6, 7, 8, 9], [2, 3, 4, 5, 6, 7, 8, 9]),
    )

    B_dm = EmptyT()
    for i in range(8):
        # Start with all regular (ground state) boundary tensors
        cur_tensors = n_tensors.copy()
        cur_tensors[i] = B_tensors[i]
        # Compute the one-site reduced density matrix and add it to the
        # total
        new_dm = get_single_site_dm(*cur_tensors)
        B_dm = B_dm + new_dm

    # The full norm can be split into two parts:
    #   - One B and Bd on the same center site, with regular boundary tensors
    #   - One Bd in the center and a B in the boundaries (many terms)
    nrm_exci = (
        ncon(
            (ts.B[0, 0], ts.Bd[0, 0], n_dm),
            ([1, 2, 3, 4, 5], [1, 6, 7, 8, 9], [2, 3, 4, 5, 6, 7, 8, 9]),
        )
        + ncon(
            (ts.A[0, 0], ts.Bd[0, 0], B_dm),
            ([1, 2, 3, 4, 5], [1, 6, 7, 8, 9], [2, 3, 4, 5, 6, 7, 8, 9]),
        )
    ) / nrm0

    # The row of the norm overlap matrix (i.e. the gradient of the norm) is the
    # reduced density matrix contracted with only the ket-layer of the center site
    nrmB_open = (
        ncon((ts.B[0, 0], n_dm), ([-1, 2, 3, 4, 5], [2, 3, 4, 5, -2, -3, -4, -5]))
        + ncon((ts.A[0, 0], B_dm), ([-1, 2, 3, 4, 5], [2, 3, 4, 5, -2, -3, -4, -5]))
    ) / nrm0

    try:
        print("B norm", nrm_exci.item(), " | Gs norm", nrm0.item(), level=1)
    except:
        pass
    return nrm_exci.real, nrm0, nrmB_open


def get_orth_basis(tensors):
    """Returns a basis of vectors orthogonal to the ground state

    Each of these vectors can be used as an input for the iPEPS
    excitation object
    """

    def get_single_site_dm(C1, T1, C2, T2, C3, T3, C4, T4):
        return ncon((C2, T1, C1, T4, C4, T3, C3, T2), "dm_single_site")

    basis = None
    A = tensors.A
    Ad = tensors.Ad
    nrms = TList(shape=A.size, pattern=A.pattern)
    for i in A.x_major():
        with cur_loc(i):
            if not nrms.is_changed(0, 0):
                n_tensors = [
                    tensors.Cs[0][-1, -1],
                    tensors.Ts[0][0, -1],
                    tensors.Cs[1][1, -1],
                    tensors.Ts[1][1, 0],
                    tensors.Cs[2][1, 1],
                    tensors.Ts[2][0, 1],
                    tensors.Cs[3][-1, 1],
                    tensors.Ts[3][-1, 0],
                ]
                # Compute the ground state one-site reduced density matrix
                n_dm = get_single_site_dm(*n_tensors)
                nrm0 = ncon(
                    (tensors.A[0, 0], tensors.Ad[0, 0], n_dm),
                    ([1, 2, 3, 4, 5], [1, 6, 7, 8, 9], [2, 3, 4, 5, 6, 7, 8, 9]),
                )
                nrms[0, 0] = nrm0
                env_0 = ncon(
                    (tensors.Ad[0, 0], n_dm),
                    ([-1, 6, 7, 8, 9], [-2, -3, -4, -5, 6, 7, 8, 9]),
                )
                env_0 = np.reshape(env_0, (1, -1))
                local_basis = linalg.null_space(onp.array(env_0))
                if basis is None:
                    basis = local_basis
                else:
                    basis = linalg.block_diag(basis, local_basis)
    if sim_config.gauge:
        basis = filter_null_modes(tensors, basis)
    # basis = _filter_null_modes(tensors, basis)
    return basis


def filter_null_modes(tensors, basis):
    def _apply_ops_h(A, B, ops):
        for i in A.x_major():
            with cur_loc(i):
                op_r = ops[0, 0]
                op_l = ops[-1, 0]
                phi = cmath.exp(1j * sim_config.px)
                B[0, 0] = phi * ncon((A[0, 0], op_r), ([-1, 1, -3, -4, -5], [1, -2]))
                B[0, 0] = B[0, 0] - ncon(
                    (A[0, 0], op_l), ([-1, -2, -3, 1, -5], [-4, 1])
                )
        return B

    def _apply_ops_v(A, B, ops):
        for i in A.x_major():
            with cur_loc(i):
                op_d = ops[0, 0]
                op_u = ops[0, -1]
                phi = cmath.exp(-1j * sim_config.py)
                B[0, 0] = phi * ncon((A[0, 0], op_u), ([-1, -2, 1, -4, -5], [-3, 1]))
                B[0, 0] = B[0, 0] - ncon(
                    (A[0, 0], op_d), ([-1, -2, -3, -4, 1], [1, -5])
                )
        return B

    ops_h = TList(pattern=tensors.A.pattern)
    ops_v = TList(pattern=tensors.A.pattern)
    D = sim_config.D
    for i in tensors.A.x_major():
        with cur_loc(i):
            ops_h[0, 0] = np.zeros((D, D))
            ops_v[0, 0] = np.zeros((D, D))

    key = random.PRNGKey(0)
    nulls = None
    for i in range(sim_config.D**2 * len(tensors.A)):
        key, subkey = random.split(key)
        v = random.normal(key, (ops_h.tot_numel(),))
        ops_h = ops_h.fill(v)
        new_vec = _apply_ops_h(tensors.A, tensors.B, ops_h).pack_data()
        new_vec = np.expand_dims(new_vec, 1)
        if i == 0:
            nulls = new_vec
        else:
            nulls = np.hstack((nulls, new_vec))
            nulls = linalg.orth(nulls)
        v = random.normal(key, (ops_v.tot_numel(),))
        ops_v = ops_v.fill(v)
        new_vec = _apply_ops_v(tensors.A, tensors.B, ops_v).pack_data()
        new_vec = np.expand_dims(new_vec, 1)
        if i == 0:
            nulls = new_vec
        else:
            nulls = np.hstack((nulls, new_vec))

    nulls = basis.T.conjugate() @ nulls
    basis = basis @ linalg.null_space(nulls.conjugate().T)
    return basis


def get_dms(ts, only_gs=False, only_nn=False):
    """Returns the two-site reduced density matrices

    This function relies on the Nested class, which contains
    tuples of different variants of site/boundary tensors.
    These variants contain either no B/Bd tensors, only a B
    tensor, only a Bd tensor or both a B and a Bd tensor.

    When the Nested tensors are contracted, all possible combinations
    that result again in one of these variants are computed and
    summed when there are multiple results in the same variant class.

    As a result, the different terms are summed on the fly during the
    contraction, which greatly reduces the computational cost.

    For example, the reduced density matrices contain 12*12=144 terms
    each (all possible locations of B and Bd tensors in the various
    boundaries), so that would make the energy evaluation 144 times
    as expensive as the ground state energy evaluation.
    Using this resummation, the total cost reduces to the maximal number
    of combinations in each contraction of pairs of tensors, 9, leading
    to a total computational cost of less than 9 times the ground state
    energy evaluation cost (the site tensors contain only two variants,
    so not every contraction contains 9 combinations).

    See the notes in nested.py for more details

    roh,rov are Nested tensors, with the following content:
        ro*[0]: ground state (no B/Bd tensors)
        ro*[1]: all terms with a single B tensor
        ro*[2]: all terms with a single Bd tensor
        ro*[3]: all terms with both a single B and Bd tensor

    All the nearest neighbor bonds with respect to super-site (0,0)
    as follows:

    """

    if only_gs:
        A = ts.A
        Ad = ts.Ad
        C1 = ts.Cs(0)
        C2 = ts.Cs(1)
        C3 = ts.Cs(2)
        C4 = ts.Cs(3)
        T1 = ts.Ts(0)
        T2 = ts.Ts(1)
        T3 = ts.Ts(2)
        T4 = ts.Ts(3)
    else:
        # The 'all_*' functions return Nested tensors, so for example
        # ts.all_Cs(0) contains (C1, B_C1, Bd_C1, BB_C1)
        A = ts.all_A
        Ad = ts.all_Ad
        C1 = ts.all_Cs(0)
        C2 = ts.all_Cs(1)
        C3 = ts.all_Cs(2)
        C4 = ts.all_Cs(3)
        T1 = ts.all_Ts(0)
        T2 = ts.all_Ts(1)
        T3 = ts.all_Ts(2)
        T4 = ts.all_Ts(3)

    # Tensors that are part of the diagonal reduced density matrix
    s_tensors = [
        C1[-1, -1],
        C2[1, -1],
        C3[1, 1],
        C4[-1, 1],
        T1[0, -1],
        T2[1, 0],
        T3[0, 1],
        T4[-1, 0],
        A[0, 0],
        Ad[0, 0],
    ]
    h_tensors = [
        C1[-1, -1],
        C2[2, -1],
        C3[2, 1],
        C4[-1, 1],
        T1[0, -1],
        T1[1, -1],
        T2[2, 0],
        T3[0, 1],
        T3[1, 1],
        T4[-1, 0],
        A[0, 0],
        A[1, 0],
        Ad[0, 0],
        Ad[1, 0],
    ]
    v_tensors = [
        C1[-1, -1],
        C2[1, -1],
        C3[1, 2],
        C4[-1, 2],
        T1[0, -1],
        T2[1, 0],
        T2[1, 1],
        T3[0, 2],
        T4[-1, 0],
        T4[-1, 1],
        A[0, 0],
        A[0, 1],
        Ad[0, 0],
        Ad[0, 1],
    ]
    p_tensors = [
        C1[-1, -1],
        C2[2, -1],
        C3[2, 2],
        C4[-1, 2],
        T1[0, -1],
        T1[1, -1],
        T2[2, 0],
        T2[2, 1],
        T3[0, 2],
        T3[1, 2],
        T4[-1, 0],
        T4[-1, 1],
        A[0, 0],
        A[1, 0],
        A[0, 1],
        A[1, 1],
        Ad[0, 0],
        Ad[1, 0],
        Ad[0, 1],
        Ad[1, 1],
    ]

    # (0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3
    # (a, b)
    # Regular variant
    ro0a0b = _get_dm_1x1(*s_tensors)
    ro0b1a = _get_dm_2x1(*h_tensors, stk=['b', 'a'])
    ro0b2a = _get_dm_1x2(*v_tensors, stk=['b', 'a'])
    if abs(sim_config.model_params['J2']) > 0:
        ro0a1a = _get_dm_2x1(*h_tensors, stk=['a', 'a'])
        ro0a2a = _get_dm_1x2(*v_tensors, stk=['a', 'a'])
        ro1a2a = _get_dm_2x2(*p_tensors, stk=['n', 'a', 'a', 'n'])
        ro0b1b = _get_dm_2x1(*h_tensors, stk=['b', 'b'])
        ro0b2b = _get_dm_1x2(*v_tensors, stk=['b', 'b'])
        ro1b2b = _get_dm_2x2(*p_tensors, stk=['n', 'b', 'b', 'n'])

        if abs(sim_config.model_params['J3']) > 0:
            ro0a3b = _get_dm_2x2(*p_tensors, stk=['b', 'n', 'n', 'a'])
            ro1a2b = _get_dm_2x2(*p_tensors, stk=['n', 'a', 'b', 'n'])
            ro1b2a = _get_dm_2x2(*p_tensors, stk=['n', 'b', 'a', 'n'])
            return ro0a0b, ro0b1a, ro0b2a, ro0a1a, ro0a2a, ro1a2a, ro0b1b, ro0b2b, ro1b2b, ro0a3b, ro1a2b, ro1b2a
        else:
            return ro0a0b, ro0b1a, ro0b2a, ro0a1a, ro0a2a, ro1a2a, ro0b1b, ro0b2b, ro1b2b
    else:
        if abs(sim_config.model_params['J3']) > 0:
            ro0a3b = _get_dm_2x2(*p_tensors, stk=['b', 'n', 'n', 'a'])
            ro1a2b = _get_dm_2x2(*p_tensors, stk=['n', 'a', 'b', 'n'])
            ro1b2a = _get_dm_2x2(*p_tensors, stk=['n', 'b', 'a', 'n'])
            return ro0a0b, ro0b1a, ro0b2a, ro0a3b, ro1a2b, ro1b2a
        else:
            return ro0a0b, ro0b1a, ro0b2a


def _get_dm_1x1(C1, C2, C3, C4, T1, T2, T3, T4, A, Ad):
    """Regular variant
    A(0,0) (a, b)
    """
    # Upper left
    upper_half = ncon([C1, T1, T4, A, C2], "dm_tri_single_upper")

    lower_half = ncon([C3, T2, T3, Ad, C4], "dm_tri_single_lower")

    # Contract upper and lower halves
    ro2 = ncon([upper_half, lower_half], "dm_tri_single")
    # print(roa)
    ro2 = ro2.reshape((2, 2, 2, 2))

    return ro2


def _get_dm_2x2(C1, C2, C3, C4, T1l, T1r, T2u, T2d, T3l, T3r, T4u, T4d, Aul, Aur, Adl, Adr, Adul, Adur, Addl, Addr,
                stk=None):
    """Regular variant

    A_lu (0,0) -- A_ru (1,0)
     |             |
    A_ld (0,1) -- A_rd (1,1)

    stk: sites to be kept on each triangle-blocks, whose order
         is given by
         stk = [stk[0,0], stk[1,0], stk[0,1], stk[1,1]]
    E.g., stk = ['b', 'n', 'n', 'a']
    """
    px = sim_config.px
    py = sim_config.py

    D = sim_config.D
    # Reshape onsite tensor
    if set(stk).issubset({'a', 'b', 'n'}):
        if stk[0] in ['a', 'b']:
            Aul = Aul.reshape([2, 2, D, D, D, D])
            Adul = Adul.reshape([2, 2, D, D, D, D])
        if stk[1] in ['a', 'b']:
            Aur = Aur.reshape([2, 2, D, D, D, D])
            Adur = Adur.reshape([2, 2, D, D, D, D])
        if stk[2] in ['a', 'b']:
            Adl = Adl.reshape([2, 2, D, D, D, D])
            Addl = Addl.reshape([2, 2, D, D, D, D])
        if stk[3] in ['a', 'b']:
            Adr = Adr.reshape([2, 2, D, D, D, D])
            Addr = Addr.reshape([2, 2, D, D, D, D])
    else:
        raise Exception(f"Invalid stk set: {stk}; expect elements of ('a', 'b', 'n')")

    # Upper left
    patch_upper_left = ncon([C1, T1l, T4u, Aul, Adul], f"dm_grp2_upper_left_{stk[0]}")  # contract

    # Upper right
    patch_upper_right = ncon([T1r.shift(px), C2.shift(px), T2u.shift(px), Aur.shift(px), Adur.shift(px)],
                             f"dm_grp2_upper_right_{stk[1]}")
    # Contract for upper half
    if stk[1] == 'n':
        upper_half = ncon([patch_upper_left, patch_upper_right], "dm_upper_ro1x1y")
    else:
        upper_half = ncon([patch_upper_left, patch_upper_right], "dm_upper_rod")

    # Lower left
    patch_lower_left = ncon([C4.shift(py), T3l.shift(py), T4d.shift(py), Adl.shift(py), Addl.shift(py)],
                            f"dm_grp2_lower_left_{stk[2]}")

    # Lower right
    patch_lower_right = ncon(
        [C3.shift(px + py), T2d.shift(px + py), T3r.shift(px + py), Adr.shift(px + py), Addr.shift(px + py)],
        f"dm_grp2_lower_right_{stk[3]}")

    # Contract for lower half
    if stk[2] == 'n':
        lower_half = ncon([patch_lower_left, patch_lower_right], "dm_lower_rod")
    else:
        lower_half = ncon([patch_lower_left, patch_lower_right], "dm_lower_ro1x1y")

    # Contract upper and lower halves
    ro2x2 = ncon([upper_half, lower_half], "dm_ro1x1y")  # dm_ro1x1y == dm_rod

    return ro2x2


def _get_dm_2x1(C1, C2, C3, C4, T1l, T1r, T2, T3l, T3r, T4, Al, Ar, Adl, Adr, stk=None):
    """Regular variant

    A_lu (0,0) -- A_ru (1,0)
    """
    px = sim_config.px

    D = sim_config.D
    # Reshape onsite tensor
    if set(stk).issubset({'a', 'b', 'n'}):
        if stk[0] in ['a', 'b']:
            Al = Al.reshape([2, 2, D, D, D, D])
            Adl = Adl.reshape([2, 2, D, D, D, D])
        if stk[1] in ['a', 'b']:
            Ar = Ar.reshape([2, 2, D, D, D, D])
            Adr = Adr.reshape([2, 2, D, D, D, D])
    else:
        raise Exception(f"Invalid stk set: {stk}; expect elements of ('a', 'b', 'n')")

    # Upper left
    patch_upper_left = ncon([C1, T1l, T4, Al, Adl], f"dm_grp2_upper_left_{stk[0]}")
    # Contract for left half
    left_half = ncon([patch_upper_left, C4, T3l], "dm_grp_left")

    # Upper right
    patch_upper_right = ncon([T1r.shift(px), C2.shift(px), T2.shift(px), Ar.shift(px), Adr.shift(px)],
                             f"dm_grp2_upper_right_{stk[1]}")

    # Contract for right half
    right_half = ncon([patch_upper_right, C3.shift(px), T3r.shift(px)], "dm_grp_right")

    # Contract upper and lower halves
    ro2x1 = ncon([left_half, right_half], "dm_grp_roh")
    return ro2x1


def _get_dm_1x2(C1, C2, C3, C4, T1, T2u, T2d, T3, T4u, T4d, Au, Ad, Adu, Add, stk=None):
    """Regular variant
    A_u (0,0)
     |
    A_d (0,1)
    """
    py = sim_config.py

    D = sim_config.D
    # Reshape onsite tensor
    if set(stk).issubset({'a', 'b', 'n'}):
        if stk[0] in ['a', 'b']:
            Au = Au.reshape([2, 2, D, D, D, D])
            Adu = Adu.reshape([2, 2, D, D, D, D])
        if stk[1] in ['a', 'b']:
            Ad = Ad.reshape([2, 2, D, D, D, D])
            Add = Add.reshape([2, 2, D, D, D, D])
    else:
        raise Exception(f"Invalid stk set: {stk}; expect elements of ('a', 'b', 'n')")

    # Upper left
    patch_upper_left = ncon([C1, T1, T4u, Au, Adu], f"dm_grp2_upper_left_{stk[0]}")  # contract
    # Contract for upper half
    upper_half = ncon([patch_upper_left, C2, T2u], "dm_grp_upper")

    # Lower left
    patch_lower_left = ncon([C4.shift(py), T3.shift(py), T4d.shift(py), Ad.shift(py), Add.shift(py)],
                            f"dm_grp2_lower_left_{stk[1]}")

    # Contract for lower half
    lower_half = ncon([patch_lower_left, C3.shift(py), T2d.shift(py)], "dm_grp_lower")

    # Contract upper and lower halves
    ro1x2 = ncon([upper_half, lower_half], "dm_grp_rov")
    return ro1x2


def get_one_site_dm(Cs, Ts, A, Ad):
    # Tensors that are part of 1-site reduced density matrix
    C1 = Cs[0][-1, -1]
    C2 = Cs[1][1, -1]
    C3 = Cs[2][1, 1]
    C4 = Cs[3][-1, 1]
    T1 = Ts[0][0, -1]
    T2 = Ts[1][1, 0]
    T3 = Ts[2][0, 1]
    T4 = Ts[3][-1, 0]

    ro1_no_op = ncon((C2, T1, C1, T4, C4, T3, C3, T2), "dm_single_site")
    ro1 = ncon((ro1_no_op, A[0, 0], Ad[0, 0]), ([1,2,3,4,5,6,7,8], [-1,1,2,3,4], [-2,5,6,7,8]))
    return ro1


def get_one_phys_site_dm(Cs, Ts, A, Ad, stk=None):
    # Tensors that are part of 1-site reduced density matrix
    C1 = Cs[0][-1, -1]
    C2 = Cs[1][1, -1]
    C3 = Cs[2][1, 1]
    C4 = Cs[3][-1, 1]
    T1 = Ts[0][0, -1]
    T2 = Ts[1][1, 0]
    T3 = Ts[2][0, 1]
    T4 = Ts[3][-1, 0]

    ro1_no_op = ncon((C2, T1, C1, T4, C4, T3, C3, T2), "dm_single_site")
    ro1x1 = ncon((ro1_no_op, A[0, 0], Ad[0, 0]), ([1, 2, 3, 4, 5, 6, 7, 8], [-1, 1, 2, 3, 4], [-2, 5, 6, 7, 8]))

    if stk == 'a':
        ro1site = ncon((ro1x1.reshape(2, 2, 2, 2), id2), ([-1, 1, -2, 2], [1, 2]))
    elif stk == 'b':
        ro1site = ncon((ro1x1.reshape(2, 2, 2, 2), id2), ([1, -1, 2, -2], [1, 2]))
    else:
        ro1site = ncon((ro1x1.reshape(2, 2, 2, 2), id2, id2), ([1, 2, 3, 4], [1, 3], [2, 4]))

    return ro1site