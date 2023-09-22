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


def get_all_energy(H, tensors):
    """Returns only energy and norm of the iPEPS"""
    E, nrm, _ = get_obs(H, tensors, measure_obs=False)
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

    Etrias = TList(shape=A.size, pattern=A.pattern)  # a0-b0-c0 triangular terms
    Etribs = TList(shape=A.size, pattern=A.pattern)  # b0-c1-a3 triangular terms
    Etrics = TList(shape=A.size, pattern=A.pattern)  # c0-a3-b2 triangular terms
    Etriannns = TList(shape=A.size, pattern=A.pattern)  # a0-a1-a3 triangular terms
    Etribnnns = TList(shape=A.size, pattern=A.pattern)  # b0-b1-b3 triangular terms
    Etricnnns = TList(shape=A.size, pattern=A.pattern)  # c0-c1-c3 triangular terms
    # Etrias_exci = TList(shape=A.size, pattern=A.pattern)
    # Etribs_exci = TList(shape=A.size, pattern=A.pattern)
    # Etrics_exci = TList(shape=A.size, pattern=A.pattern)

    nrmtrias = TList(shape=A.size, pattern=A.pattern)  # a0-b0-c0 triangular terms
    nrmtribs = TList(shape=A.size, pattern=A.pattern)  # b0-c1-a3 triangular terms
    nrmtrics = TList(shape=A.size, pattern=A.pattern)  # c1-a4-b3 triangular terms
    nrmtriannns = TList(shape=A.size, pattern=A.pattern)  # a0-a1-a3 triangular terms
    nrmtribnnns = TList(shape=A.size, pattern=A.pattern)  # b0-b1-b3 triangular terms
    nrmtricnnns = TList(shape=A.size, pattern=A.pattern)  # c0-c1-c3 triangular terms
    obs_evs = [TList(shape=A.size, pattern=A.pattern) for _ in [sigmaz, sigmap, sigmam]]

    for i in A.x_major():
        with cur_loc(i):
            if not Etrias.is_changed(0, 0):
                # (1) Construct all the 2-body density matrices individually for the energy evaluation
                # roh, rov, rod, ron2x1y, ron1x2y, ro1x1y = get_dms(tensors)
                #
                # nrmh = np.trace(np.reshape(roh[0], (4, 4))).real
                # nrmv = np.trace(np.reshape(rov[0], (4, 4))).real
                # nrmd = np.trace(np.reshape(rod[0], (4, 4))).real
                # nrmn2x1y = np.trace(np.reshape(ron2x1y[0], (4, 4))).real
                # nrmn1x2y = np.trace(np.reshape(ron1x2y[0], (4, 4))).real
                # nrm1x1y = np.trace(np.reshape(ro1x1y[0], (4, 4))).real
                #
                # nrmhs[0, 0] = nrmh
                # nrmvs[0, 0] = nrmv
                # nrmds[1, 0] = nrmd
                # nrmn2x1ys[2, 0] = nrmn2x1y
                # nrmn1x2ys[1, 0] = nrmn1x2y
                # nrm1x1ys[0, 0] = nrm1x1y
                #
                # roh = roh / nrmh
                # rov = rov / nrmv
                # rod = rod / nrmd
                # ron2x1y = ron2x1y / nrmn2x1y
                # ron1x2y = ron1x2y / nrmn1x2y
                # ro1x1y = ro1x1y / nrm1x1y
                #
                # Ehs[0, 0] = ncon([roh, H[((0, 0), (1, 0))]], ([1, 2, 3, 4], [1, 2, 3, 4])).real
                # Evs[0, 0] = ncon([rov, H[((0, 0), (0, 1))]], ([1, 2, 3, 4], [1, 2, 3, 4])).real
                # Eds[1, 0] = ncon([rod, H[((0, 0), (-1, 1))]], ([1, 2, 3, 4], [1, 2, 3, 4])).real
                # En2x1ys[2, 0] = ncon([ron2x1y, H[((0, 0), (-2, 1))]], ([1, 2, 3, 4], [1, 2, 3, 4])).real
                # En1x2ys[1, 0] = ncon([ron1x2y, H[((0, 0), (-1, 2))]], ([1, 2, 3, 4], [1, 2, 3, 4])).real
                # E1x1ys[0, 0] = ncon([ro1x1y, H[((0, 0), (1, 1))]], ([1, 2, 3, 4], [1, 2, 3, 4])).real

                # (2) Construct 3-body density matrices for the energy evaluation (2-body still needed for bonds evaluation)
                rotria, rotrib, rotric, rotria_nnn, rotrib_nnn, rotric_nnn = get_dms_tri(tensors)
                nrma = np.trace(np.reshape(rotria[0], (6, 6))).real
                nrmb = np.trace(np.reshape(rotrib[0], (6, 6))).real
                nrmc = np.trace(np.reshape(rotric[0], (6, 6))).real
                nrmannn = np.trace(np.reshape(rotria_nnn[0], (6, 6))).real
                nrmbnnn = np.trace(np.reshape(rotrib_nnn[0], (6, 6))).real
                nrmcnnn = np.trace(np.reshape(rotric_nnn[0], (6, 6))).real

                rotria = rotria / nrma
                rotrib = rotrib / nrmb
                rotric = rotric / nrmc
                rotria_nnn = rotria_nnn / nrmannn
                rotrib_nnn = rotrib_nnn / nrmbnnn
                rotric_nnn = rotric_nnn / nrmcnnn

                Etrias[0, 0] = ncon([rotria, H[(0, 0), (1, 0)]], ([1,2,3,4,5,6], [1,2,3,4,5,6])).real
                Etribs[0, 0] = ncon([rotrib, H[(0, 0), (1, 0)]], ([1,2,3,4,5,6], [1,2,3,4,5,6])).real
                Etrics[0, 0] = ncon([rotric, H[(0, 0), (1, 0)]], ([1,2,3,4,5,6], [1,2,3,4,5,6])).real
                Etriannns[0, 0] = ncon([rotria_nnn, H[(0, 0), (0, 1)]], ([1,2,3,4,5,6], [1,2,3,4,5,6])).real
                Etribnnns[0, 0] = ncon([rotrib_nnn, H[(0, 0), (0, 1)]], ([1,2,3,4,5,6], [1,2,3,4,5,6])).real
                Etricnnns[0, 0] = ncon([rotric_nnn, H[(0, 0), (0, 1)]], ([1,2,3,4,5,6], [1,2,3,4,5,6])).real

                # if measure_obs:
                #     ro_one = get_one_site_dm(tensors.Cs,tensors.Ts,A,Ad)
                #     for obs_i, obs in enumerate([(sigmap+sigmam)/2, 1j*(sigmam-sigmap)/2, sigmaz/2]):
                #         try:
                #             obs_ev = ncon([ro_one, obs], ([1, 2], [1, 2]))
                #             norm = ncon([ro_one, id2], ([1, 2], [1, 2]))
                #             obs_evs[obs_i][0, 0] = obs_ev / norm
                #         except:
                #             obs_evs[obs_i][0, 0] = np.nan
                #     # for obs_i, obs in enumerate(tensors.observables):
                #     #     if obs.size == 1:
                #     #         try:
                #     #             obs_ev = ncon([ro_one, obs.operator], ([1,2],[1,2]))
                #     #             # print(f"Obs {(obs_i,i)} {obs.__repr__()}: {obs_ev.item()}", level=2)
                #     #             print(f"Obs {(obs_i,i)} {obs.__repr__()}: {obs_ev.astype(float)}", level=2)
                #     #             obs_evs[obs_i][0,0] = obs_ev.astype(float)
                #     #         except:
                #     #             obs_evs[obs_i][0,0] = np.nan
                #     #     elif obs.size == 2:
                #     #         try:
                #     #             obs_ev_h = ncon([roh, obs.operator], ([1,2,3,4],[1,2,3,4]))
                #     #             obs_ev_v = ncon([rov, obs.operator], ([1,2,3,4],[1,2,3,4]))
                #     #             print(f"Obs {(obs_i,i)} {obs.__repr__()}: {obs_ev_h.item()}, {obs_ev_v.item()}", level=2)
                #     #             obs_evs[obs_i][0,0] = (obs_ev_h.item(), obs_ev_v.item())
                #     #         except:
                #     #             obs_evs[obs_i][0,0] = (np.nan, np.nan)
                # print(Ehs[0, 1], Evs[0, 0], Eds[1, 1], level=2)
    # try:
    #     print(Ehs.mean(), Evs.mean(), Eds.mean(), level=2)
    # except:
    #     print(Ehs.mean(), Evs.mean(), Eds.mean(), level=2)
    # print(Ehs.mean(), Evs.mean(), Eds.mean(), level=2)
    E = Etrias.mean() + Etribs.mean() + Etrics.mean() + Etriannns.mean() + Etribnnns.mean() + Etricnnns.mean()

    # nrm = 0.5 * (nrmhs.mean() + nrmvs.mean())
    # nrm = (nrmhs.mean() + nrmvs.mean() + nrmds.mean()) / 3.
    nrm = (nrmtrias.mean() + nrmtribs.mean() + nrmtrics.mean() + nrmtriannns.mean() + nrmtribnnns.mean() + nrmtricnnns.mean()) / 6.
    return E, nrm, obs_evs


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

def get_dms_tri(ts, only_gs=False):
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


    # Regular variant
    roa = _get_dm_tria(*s_tensors)
    rob = _get_dm_trib(*p_tensors)
    # rop = _get_dm_p(*p_tensors)
    roc = _get_dm_tric(*p_tensors)

    roa2 = _get_dm_tria_nnn(*p_tensors)
    rob2 = _get_dm_trib_nnn(*p_tensors)
    roc2 = _get_dm_tric_nnn(*p_tensors)

    return roa, rob, roc, roa2, rob2, roc2

def _get_dm_tria(C1, C2, C3, C4, T1, T2, T3, T4, A, Ad):
    """Regular variant

    A_lu (0,0) -- A_ru (1,0)
     |             |
    A_ld (0,1) -- A_rd (1,1)
    """
    px = sim_config.px
    py = sim_config.py
    # distortion of momentum
    # [[\sqrt{3}/2,  1/2]
    #  [\sqrt{3}/2, -1/2]]
    # Upper left
    upper_half = ncon([C1, T1, T4, A, C2], "dm_tri_single_upper")

    lower_half = ncon([C3, T2, T3, Ad, C4], "dm_tri_single_lower")

    # Contract upper and lower halves
    roa = ncon([upper_half, lower_half], "dm_tri_single")

    roa = np.reshape(roa, [2, 2, 2, 2, 2, 2])

    return roa


def _get_dm_tri(C1, C2, C3, C4, T1l, T1r, T2u, T2d, T3l, T3r, T4u, T4d, Aul, Aur, Adl, Adr, Adul, Adur, Addl, Addr, stk=None):
    """Regular variant

    A_lu (0,0) -- A_ru (1,0)
     |             |
    A_ld (0,1) -- A_rd (1,1)

    stk: sites to be kept on each triangle-blocks, whose order
         is given by
         stk = [stk[0,0], stk[1,0], stk[0,1], stk[1,1]]
    E.g., stk = ['b', 'c', 'n', 'a']
    """
    px = sim_config.px
    py = sim_config.py

    D = sim_config.D
    # Reshape onsite tensor
    if stk == {'a', 'b', 'c', 'n'}:
        if stk[0] in ['a', 'b', 'c']:
            Aul = Aul.reshape([2, 2, 2, D, D, D, D])
            Adul = Adul.reshape([2, 2, 2, D, D, D, D])
        if stk[1] in ['a', 'b', 'c']:
            Aur = Aur.reshape([2, 2, 2, D, D, D, D])
            Adur = Adur.reshape([2, 2, 2, D, D, D, D])
        if stk[2] in ['a', 'b', 'c']:
            Adl = Adl.reshape([2, 2, 2, D, D, D, D])
            Addl = Addl.reshape([2, 2, 2, D, D, D, D])
        if stk[3] in ['a', 'b', 'c']:
            Adr = Adr.reshape([2, 2, 2, D, D, D, D])
            Addr = Addr.reshape([2, 2, 2, D, D, D, D])
    else:
        raise Exception(f"Invalid stk set: {stk}; expect an arangement of ('a', 'b', 'c', 'n')")

    # Upper left
    patch_upper_left = ncon([C1, T1l, T4u, Aul, Adul], f"dm_tri_upper_left_{stk[0]}")  # contract

    # Upper right
    patch_upper_right = ncon([T1r.shift(px), C2.shift(px), T2u.shift(px), Aur.shift(px), Adur.shift(px)], f"dm_tri_upper_right_{stk[1]}")
    # Contract for upper half
    if stk[2] == 'n':
        upper_half = ncon([patch_upper_left, patch_upper_right], "dm_tri_upper")
    else:
        upper_half = ncon([patch_upper_left, patch_upper_right], "dm_upper_rod")

    # Lower left
    patch_lower_left = ncon([C4.shift(py), T3l.shift(py), T4d.shift(py), Adl.shift(py), Addl.shift(py)], f"dm_tri_lower_left_{stk[2]}")

    # Lower right
    patch_lower_right = ncon([C3.shift(px+py), T2d.shift(px+py), T3r.shift(px+py), Adr.shift(px+py), Addr.shift(px+py)], f"dm_lower_right_{stk[3]}")

    # Contract for lower half
    if stk[2] == 'n':
        lower_half = ncon([patch_lower_left, patch_lower_right], "dm_lower_rod")
    else:
        lower_half = ncon([patch_lower_left, patch_lower_right], "dm_tri_lower")

    # Contract upper and lower halves
    if stk[2] == 'n':
        rotri = ncon([upper_half, lower_half], "dm_tri_l")
    else:
        rotri = ncon([upper_half, lower_half], "dm_tri_u")

    return rotri


def _get_dm_1x1y(C1, C2, C3, C4, T1l, T1r, T2u, T2d, T3l, T3r, T4u, T4d, Aul, Aur, Adl, Adr, Adul, Adur, Addl, Addr):
    """Regular variant

    A_lu (0,0) -- A_ru (1,0)
     |             |
    A_ld (0,1) -- A_rd (1,1)
    """
    px = sim_config.px
    py = sim_config.py
    # distortion of momentum
    # [[\sqrt{3}/2,  1/2]
    #  [\sqrt{3}/2, -1/2]]
    # Upper left
    patch_upper_left = ncon([C1, T1l, T4u, Aul, Adul], "dm_upper_left")
    # Upper right
    patch_upper_right = ncon([T1r.shift(px), C2.shift(px), T2u.shift(px), Aur.shift(px), Adur.shift(px)], "dm_upper_right_traced")
    # Contract for upper half
    upper_half = ncon([patch_upper_left, patch_upper_right], "dm_upper_ro1x1y")

    # Lower left
    patch_lower_left = ncon([C4.shift(py), T3l.shift(py), T4d.shift(py), Adl.shift(py), Addl.shift(py)], "dm_lower_left_traced")

    # Lower right
    patch_lower_right = ncon([C3.shift(px+py), T2d.shift(px+py), T3r.shift(px+py), Adr.shift(px+py), Addr.shift(px+py)], "dm_lower_right")
    # Contract for lower half
    lower_half = ncon([patch_lower_left, patch_lower_right], "dm_lower_ro1x1y")

    # Contract upper and lower halves
    rod = ncon([upper_half, lower_half], "dm_ro1x1y")
    return rod


def _get_dm_n2x1y(C1, C2, C3, C4, T1l, T1m, T1r, T2u, T2d, T3l, T3m, T3r, T4u, T4d, Aul, Aum, Aur, Adl, Adm, Adr, Adul, Adum, Adur, Addl, Addm, Addr):
    """Regular variant

    A_lu (0,0) -- A_mu (1,0) -- A_ru (2,0)
     |             |             |
    A_ld (0,1) -- A_md (1,1) -- A_rd (2,1)
    """
    px = sim_config.px
    py = sim_config.py
    # distortion of momentum
    # [[\sqrt{3}/2,  1/2]
    #  [\sqrt{3}/2, -1/2]]
    # Upper left
    patch_upper_left = ncon([C1, T1l, T4u, Aul, Adul], "dm_upper_left_traced")  # contract
    # Upper left + upper middle
    patch_upper_left2 = ncon([patch_upper_left, T1m.shift(px), Aum.shift(px), Adum.shift(px)], "dm_upper_left_upper_middle_traced")
    # Upper right
    patch_upper_right = ncon([T1r.shift(2*px), C2.shift(2*px), T2u.shift(2*px), Aur.shift(2*px), Adur.shift(2*px)], "dm_upper_right")
    # Contract for upper half
    upper_half = ncon([patch_upper_left2, patch_upper_right], "dm_upper_ron2x1y")

    # Lower left
    patch_lower_left = ncon([C4.shift(py), T3l.shift(py), T4d.shift(py), Adl.shift(py), Addl.shift(py)], "dm_lower_left")
    # Lower right
    patch_lower_right = ncon([C3.shift(2*px+py), T2d.shift(2*px+py), T3r.shift(2*px+py), Adr.shift(2*px+py), Addr.shift(2*px+py)], "dm_lower_right_traced")
    # Lower right + lower middle
    patch_lower_right2 = ncon([patch_lower_right, T3m.shift(px+py), Adm.shift(px+py), Addm.shift(px+py)], "dm_lower_right_lower_middle_traced")
    # Contract for lower half
    lower_half = ncon([patch_lower_right2, patch_lower_left], "dm_lower_ron2x1y")

    # Contract upper and lower halves
    rod = ncon([upper_half, lower_half], "dm_ron2x1y")
    return rod


def _get_dm_n1x2y(C1, C2, C3, C4, T1l, T1r, T2u, T2m, T2d, T3l, T3r, T4u, T4m, T4d, Aul, Aur, Aml, Amr, Adl, Adr, Adul, Adur, Adml, Admr, Addl, Addr):
    """Regular variant

    A_lu (0,0) -- A_ru (1,0)
     |             |
    A_lm (0,1) -- A_rm (1,1)
     |             |
    A_ld (0,2) -- A_rd (1,2)

    """
    px = sim_config.px
    py = sim_config.py
    # distortion of momentum
    # [[\sqrt{3}/2,  1/2]
    #  [\sqrt{3}/2, -1/2]]
    # Upper left
    patch_upper_left = ncon([C1, T1l, T4u, Aul, Adul], "dm_upper_left_traced")
    # Upper left + middle left
    patch_upper_left2 = ncon([patch_upper_left, T4m.shift(py), Aml.shift(py), Adml.shift(py)], "dm_upper_left_left_middle_traced")
    # Lower left
    patch_lower_left = ncon([C4.shift(2*py), T3l.shift(2*py), T4d.shift(2*py), Adl.shift(2*py), Addl.shift(2*py)], "dm_lower_left")
    # Contract for left half
    left_half = ncon([patch_upper_left2, patch_lower_left], "dm_left_ron1x2y")


    # Upper right
    patch_upper_right = ncon([T1r.shift(px), C2.shift(px), T2u.shift(px), Aur.shift(px), Adur.shift(px)], "dm_upper_right")
    # Lower right
    patch_lower_right = ncon([C3.shift(px+2*py), T2d.shift(px+2*py), T3r.shift(px+2*py), Adr.shift(px+2*py), Addr.shift(px+2*py)], "dm_lower_right_traced")
    # Lower right + lower middle
    patch_lower_right2 = ncon([patch_lower_right, T2m.shift(px+py), Amr.shift(px+py), Admr.shift(px+py)], "dm_lower_right_right_middle_traced")
    # Contract for lower half
    right_half = ncon([patch_lower_right2, patch_upper_right], "dm_right_ron1x2y")

    # Contract upper and lower halves
    rod = ncon([left_half, right_half], "dm_ron1x2y")
    return rod


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
