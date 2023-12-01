""" Main excited-state executable script

    Note:
        The simulations are intended to be used by calling the package 
        directly via :code:`python -m adpeps ...`, as described in 
        :ref:`notes/start`
"""

import jax
import jax.numpy as np
import numpy as onp
from jax import grad, jit, random, value_and_grad, vmap
from jax.scipy.optimize import minimize
from jax.test_util import check_grads
from scipy import optimize
from scipy.linalg import eig, eigh, block_diag
from yaml import dump, safe_load

import adpeps.ipeps.config as sim_config
from adpeps.ipeps.evaluation import filter_null_modes
# from adpeps.ipeps.ipeps import iPEPS, iPEPS_exci
from adpeps.ipeps.ipeps_trgl_grp import iPEPS, iPEPS_exci
from adpeps.ipeps.make_momentum_path import make_momentum_path
from adpeps.utils import io
from adpeps.utils.printing import print
from adpeps.utils.tlist import TList, cur_loc, set_pattern

from adpeps.tensor.contractions import ncon
from adpeps.ipeps.evaluation_trgl_grp import get_one_site_dm

def run(config_file: str, momentum_ix: int):
    """Start the simulation

    Args:
        config_file: filename of the configuration file
        momentum_ix: index of the point in momentum space
    """

    print(config_file)
    with open(config_file) as f:
        cfg = safe_load(f)

    # Show options
    print(dump(cfg))

    sim_config.from_dict(cfg)
    base_file = io.get_exci_base_file()
    if not base_file.exists():
        print(
            f"Base file {base_file} not found. Prepare the simulation first by \
                running with option '-i'"
        )
        return

    sim = iPEPSExciSimulation(config_file, momentum_ix)
    output_folder = io.get_exci_folder()
    output_folder.mkdir(parents=True, exist_ok=True)
    kxs, kys = make_momentum_path(sim_config.momentum_path)
    sim_config.px = kxs[momentum_ix]
    sim_config.py = kys[momentum_ix]
    output_file = io.get_exci_file(momentum_ix)
    print(f"Output: {output_file}", level=2)
    basis_size = sim.basis_size
    res_dtype = np.complex128
    H = onp.zeros((basis_size, basis_size), dtype=res_dtype)
    N = onp.zeros((basis_size, basis_size), dtype=res_dtype)

    for m in range(basis_size):
        grad_H, grad_N = sim(m)
        H[:, m] = grad_H
        N[:, m] = grad_N
        onp.savez(output_file, H=H, N=N)

    print(H)
    print(N)
    onp.savez(output_file, H=H, N=N)
    print("Done")
    print(f"Saved to {output_file}")


def prepare(config_file):
    with open(config_file) as f:
        cfg = safe_load(f)
    sim_config.from_dict(cfg)
    base_file = io.get_exci_base_file()
    print(base_file)
    peps = iPEPS()

    gs_file = io.get_gs_file()
    loaded_sim = np.load(gs_file, allow_pickle=True)
    peps = loaded_sim["peps"].item()

    sim_config.ctm_max_iter = 30
    sim_config.ctm_conv_tol = 1e-12

    # Converge GS boundary tensors
    peps.converge_boundaries()

    # Convert to excitations iPEPS
    peps.__class__ = iPEPS_exci

    # Normalize the ground-state tensors such that the state has norm 1
    peps.normalize_gs()

    # Shift the Hamiltonian by the ground-state energy
    # The excited state energy is then relative to the ground state
    peps.substract_gs_energy()

    # Prepare an orthonormal basis with respect to the ground state
    print("Preparing orthonormal basis")
    basis = peps.compute_orth_basis()

    print(f"Saving base to {base_file}")
    np.savez(base_file, peps=peps, basis=basis)


def evaluate_single(config_file, momentum_ix):
    def _compute_ev_red_basis(H, N, P, n):
        P = P[:, :n]
        N2 = P.T.conjugate() @ N @ P
        H2 = P.T.conjugate() @ H @ P
        N2 = 0.5 * (N2 + N2.T.conjugate())
        H2 = 0.5 * (H2 + H2.T.conjugate())
        ev, _ = eig(H2, N2)
        return sorted(ev.real)

    with open(config_file) as f:
        cfg = safe_load(f)

    sim_config.from_dict(cfg)
    kxs, kys = make_momentum_path(sim_config.momentum_path)
    sim_config.px = kxs[momentum_ix]
    sim_config.py = kys[momentum_ix]
    base_file = io.get_exci_base_file()
    base_sim = np.load(base_file, allow_pickle=True)
    output_file = io.get_exci_file(momentum_ix)
    print(output_file)
    dat = np.load(output_file)
    H, N = dat["H"], dat["N"]
    basis = base_sim["basis"]
    peps = base_sim["peps"].item()

    # basis = basis.T @ filter_null_modes(peps.tensors, basis)
    # print(basis.shape)
    # print(N.shape)
    # N = basis.T @ N @ basis
    # H = basis.T @ H @ basis
    # H = H.conjugate()

    H = 0.5 * (H + H.T.conjugate())
    N = 0.5 * (N + N.T.conjugate())
    ev_N, P = np.linalg.eig(N)
    idx = ev_N.real.argsort()[::-1]
    ev_N = ev_N[idx]
    selected = (ev_N / ev_N.max()) > 1e-3
    P = P[:, idx]
    P = P[:, selected]
    N2 = P.T.conjugate() @ N @ P
    H2 = P.T.conjugate() @ H @ P
    N2 = 0.5 * (N2 + N2.T.conjugate())
    H2 = 0.5 * (H2 + H2.T.conjugate())
    ev, vectors = eig(H2, N2)
    ixs = np.argsort(ev)
    ev = ev[ixs]
    vectors = vectors[:, ixs]

    return sorted(ev.real)


def evaluate_spectral_weight(config_file, momentum_ix, tol_norm=1e-3, n_basis=None):
    def _compute_ev_red_basis(H, N, P, n):
        P = P[:, :n]
        N2 = P.T.conjugate() @ N @ P
        H2 = P.T.conjugate() @ H @ P
        N2 = 0.5 * (N2 + N2.T.conjugate())
        H2 = 0.5 * (H2 + H2.T.conjugate())
        ev, _ = eig(H2, N2)
        return sorted(ev.real)

    with open(config_file) as f:
        cfg = safe_load(f)

    sim_config.from_dict(cfg)
    kxs, kys = make_momentum_path(sim_config.momentum_path)
    sim_config.px = kxs[momentum_ix]
    sim_config.py = kys[momentum_ix]
    base_file = io.get_exci_base_file()
    base_sim = np.load(base_file, allow_pickle=True)
    output_file = io.get_exci_file(momentum_ix)
    print(output_file)
    dat = np.load(output_file)
    H, N = dat["H"], dat["N"]
    basis = base_sim["basis"]
    peps = base_sim["peps"].item()

    # basis = basis.T @ filter_null_modes(peps.tensors, basis)
    # print(basis.shape)
    # print(N.shape)
    # N = basis.T @ N @ basis
    # H = basis.T @ H @ basis
    # H = H.conjugate()

    H = 0.5 * (H + H.T.conjugate())
    N = 0.5 * (N + N.T.conjugate())
    ev_N, P = np.linalg.eig(N)
    idx = ev_N.real.argsort()[::-1]
    ev_N = ev_N[idx]
    # selected = (ev_N / ev_N.max()) > 1e-3
    if n_basis is not None:
        selected = np.arange(n_basis)
    else:
        selected = (ev_N / ev_N.max()) > tol_norm
    P = P[:, idx]
    P = P[:, selected]
    N2 = P.T.conjugate() @ N @ P
    H2 = P.T.conjugate() @ H @ P
    N2 = 0.5 * (N2 + N2.T.conjugate())
    H2 = 0.5 * (H2 + H2.T.conjugate())
    ev, vectors = eig(H2, N2)
    ixs = np.argsort(ev)
    ev = ev[ixs]
    vectors = vectors[:, ixs]

    sx = np.array([[0, 0.5], [0.5, 0]])
    sy = np.array([[0, -0.5j], [0.5j, 0]])
    sz = np.array([[0.5, 0], [0, -0.5]])
    idp2 = np.array([[1, 0], [0, 1]])
    ops = [sx, sy, sz]

    A = peps.tensors.A

    # subtract ground state expectation value
    obs_gs = np.array(peps.evaluate_obs())
    num_sites = 3
    ops_exci = [TList(shape=A.size, pattern=A.pattern) for _ in range(num_sites*len(ops))]
    for i in A.x_major():
        with cur_loc(i):
            if not ops_exci[0].is_changed(0, 0):
                for obs_i in range(num_sites*len(ops)):
                    ops_exci[obs_i][0, 0] = ops[obs_i % num_sites] - 0 * obs_gs[obs_i] * idp2
    t = peps.tensors

    def get_single_site_dm(C1, T1, C2, T2, C3, T3, C4, T4):
        return ncon((C2, T1, C1, T4, C4, T3, C3, T2), "dm_single_site")

    env_0s = TList(shape=A.size, pattern=A.pattern)

    nrms = TList(shape=A.size, pattern=A.pattern)
    for i in A.x_major():
        with cur_loc(i):
            if not nrms.is_changed(0, 0):
                n_tensors = [
                    t.Cs[0][-1, -1],
                    t.Ts[0][0, -1],
                    t.Cs[1][1, -1],
                    t.Ts[1][1, 0],
                    t.Cs[2][1, 1],
                    t.Ts[2][0, 1],
                    t.Cs[3][-1, 1],
                    t.Ts[3][-1, 0],
                ]
                # Compute the ground state one-site reduced density matrix
                n_dm = get_single_site_dm(*n_tensors)
                nrm0 = ncon(
                    (t.A[0, 0], t.Ad[0, 0], n_dm),
                    ([1, 2, 3, 4, 5], [1, 6, 7, 8, 9], [2, 3, 4, 5, 6, 7, 8, 9]),
                )
                nrms[0, 0] = nrm0
                env_0 = ncon(
                    (peps.tensors.Ad[0, 0], n_dm),
                    ([-1, 6, 7, 8, 9], [-2, -3, -4, -5, 6, 7, 8, 9]),
                )
                env_0s[0, 0] = env_0

    gs_with_ops = []
    gs0s = []
    for i_op in range(len(ops)):
        gs_with_op = None
        gs0 = None
        nrms = TList(shape=A.size, pattern=A.pattern)
        nrms1 = TList(shape=A.size, pattern=A.pattern)
        op_exci = None
        for i in A.x_major():
            with cur_loc(i):
                if not nrms1.is_changed(0, 0):
                    nrms1.mark_changed(i)
                    op_1 = ops_exci[i_op][0, 0]
                    op_exci = np.einsum('ia,jb,kc->ijkabc', op_1, idp2, idp2)
                    op_2 = np.exp(1j*(sim_config.px+2*sim_config.py)/3) * ops_exci[i_op+num_sites][0, 0]
                    op_exci += np.einsum('ia,jb,kc->ijkabc', idp2, op_2, idp2)
                    op_3 = np.exp(1j*(2*sim_config.px+sim_config.py)/3) * ops_exci[i_op+num_sites*2][0, 0]
                    op_exci += np.einsum('ia,jb,kc->ijkabc', idp2, idp2, op_3)
        op_exci = np.reshape(op_exci, (8, 8))
        for i in A.x_major():
            with cur_loc(i):
                if not nrms.is_changed(0, 0):
                    nrms.mark_changed(i)
                    A_op = ncon((A[0, 0], op_exci), ([1, -2, -3, -4, -5], [-1, 1]))
                    A0 = A[0, 0]
                    # nrm_gs = ncon((env_0s[0, 0], env_0s[0, 0]), ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]))
                    # A_op = A_op - 0 * ncon((A_op, env_0s[0, 0]), ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])) * env_0s[0, 0] / nrm_gs
                    # A0 = A[0, 0] - 0 * np.einsum('uijkl,uijkl', A[0, 0], env_0s[0, 0]) * env_0s[0, 0] / nrm_gs
                    gs0 = A0
        gs_with_op = np.reshape(A_op, (-1))
        gs_with_ops.append(gs_with_op)
        gs0 = np.reshape(gs0, (-1))
        gs0s.append(gs0)

    basis2 = basis @ N @ P @ vectors
    spectral_weight = []
    rho = vectors.T.conj() @ P.T.conj() @ N @ P @ vectors @ vectors.T.conj() @ P.T.conj() @ N.T.conj() @ P @ vectors
    # rho = vectors.T.conj() @ P.T.conj() @ N @ N.T.conj() @ P @ vectors
    norm = np.sum(np.diag(rho))
    for gs_with_op in gs_with_ops:
        sw = basis2.T @ gs_with_op.conj() / np.sqrt(norm)
        spectral_weight.append(np.abs(sw)**2)

    return spectral_weight, ev.real


def run_sq_static(config_file):
    with open(config_file) as f:
        cfg = safe_load(f)

    sim_config.from_dict(cfg)

    from pathlib import Path
    output_file = Path(io.get_exci_folder(), "sq_static.npz")
    print(output_file)
    # if not output_file.exists() or not sim_config.resume:
    if not output_file.exists():
        sx = np.array([[0, 0.5], [0.5, 0]])
        sy = np.array([[0, -0.5j], [0.5j, 0]])
        sz = np.array([[0.5, 0], [0, -0.5]])
        idp = np.array([[1, 0], [0, 1]])
        ops = [sx, sy, sz]

        with open(config_file) as f:
            cfg = safe_load(f)
        sim_config.from_dict(cfg)

        base_file = io.get_exci_base_file()
        base_sim = np.load(base_file, allow_pickle=True)
        peps = base_sim["peps"].item()
        # substract ground state expectation value of spin operators
        A = peps.tensors.A
        obs_gs = np.array(peps.evaluate_obs())
        num_sites = 3
        ops_exci = [TList(shape=A.size, pattern=A.pattern) for _ in range(num_sites * len(ops))]
        for i in A.x_major():
            with cur_loc(i):
                if not ops_exci[0].is_changed(0, 0):
                    for obs_i in range(num_sites * len(ops)):
                        ops_exci[obs_i][0, 0] = ops[obs_i % num_sites] - 1 * obs_gs[obs_i] * idp
        # gs0s = []
        gs0 = A[0, 0]
        gs0 = np.reshape(gs0, (-1))
        kxs, kys = make_momentum_path(sim_config.momentum_path)
        print(f"Output: {output_file}", level=2)
        res_dtype = np.complex128
        N = onp.zeros((len(ops), len(kxs)), dtype=res_dtype)
        N2 = onp.zeros((len(ops), len(kxs)), dtype=res_dtype)
        for m in range(len(kxs)):
            sim_config.px = kxs[m]
            sim_config.py = kys[m]
            print(f"momentum_ix={m+1}, kx={sim_config.px/np.pi:.5}pi, ky={sim_config.py/np.pi:.5}pi")
            for obs_i in range(len(ops)):
                gs_with_op = None
                # gs0 = None
                nrms = TList(shape=A.size, pattern=A.pattern)
                nrms1 = TList(shape=A.size, pattern=A.pattern)
                op_exci = None
                for i in A.x_major():
                    with cur_loc(i):
                        if not nrms1.is_changed(0, 0):
                            nrms1.mark_changed(i)
                            op_1 = ops_exci[obs_i][0, 0]
                            op_exci = np.einsum('ia,jb,kc->ijkabc', op_1, idp, idp)
                            op_2 = np.exp(1j * (sim_config.px + 2 * sim_config.py) / 3) * ops_exci[obs_i + num_sites][0, 0]
                            op_exci += np.einsum('ia,jb,kc->ijkabc', idp, op_2, idp)
                            op_3 = np.exp(1j * (2 * sim_config.px + sim_config.py) / 3) * ops_exci[obs_i + num_sites * 2][0, 0]
                            op_exci += np.einsum('ia,jb,kc->ijkabc', idp, idp, op_3)
                op_exci = np.reshape(op_exci, (8, 8))
                for i in A.x_major():
                    with cur_loc(i):
                        if not nrms.is_changed(0, 0):
                            nrms.mark_changed(i)
                            gs_with_op = ncon((A[0, 0], op_exci), ([1, -2, -3, -4, -5], [-1, 1]))
                gs_with_op = np.reshape(gs_with_op, (-1))
                sA = gs_with_op

                res = peps.run(np.array(sA))
                s_disc = gs0.T.conjugate() @ res[1].pack_data()
                N[obs_i, m] = (sA.T.conjugate() @ res[1].pack_data()).real
                N2[obs_i, m] = s_disc

        # print(repr(N))
        # print(repr(N2))
        onp.savez(output_file,  N=N)
        print("Done")
        print(f"Saved to {output_file}")
    else:
        print(f"Read static structure factors from {output_file}.")
        dat = np.load(output_file)
        N = dat["N"]
    print(repr(N))
    return N


def evaluate(config_file, momentum_ix):
    # Default option (-1): evaluate all momenta
    if momentum_ix != -1:
        return evaluate_single(config_file, momentum_ix)

    with open(config_file) as f:
        cfg = safe_load(f)

    # Show options
    print(dump(cfg))

    sim_config.from_dict(cfg)
    kxs, kys, plot_info = make_momentum_path(
        sim_config.momentum_path, with_plot_info=True
    )
    tols_norm = [1e-3]*len(kxs)
    num_basis = [None]*len(kxs)
    # num_basis = [60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
    #              60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
    #              60, 60, 70, 70, 70, 70, 70]
    plot_spectrum = True
    is_plot = False
    is_savefig = True
    import matplotlib.pyplot as plt
    if not plot_spectrum:
        evs = []
        for ix in range(len(kxs)):
            try:
                ev = evaluate_single(config_file, ix)
            except:
                ev = [np.nan]
            evs.append(ev[0])
        plt.plot(evs, "--+")
        plt.xticks(**plot_info["xticks"])
        plt.title(f"Dispersion {sim_config.model} D={sim_config.D}")
        plt.xlabel("k")
        plt.ylabel("$\omega$")
        plt.show()
    else:
        # DYNAMICAL SPIN STRUCTURE FACTOR SPECTRUM
        evs = []
        evs_full = []
        obs = []
        for ix in range(len(kxs)):
            try:
                # ev = evaluate_single(config_file, ix)
                sqw, ev = evaluate_spectral_weight(config_file, ix, tol_norm=tols_norm[ix], n_basis=num_basis[ix])
            except:
                ev = [np.nan]
                sqw = [np.nan]
            evs_full.append(ev)
            evs.append(ev[0])
            obs.append(sqw)
        tot_sws = [[np.sum(obs[ix][0]) for ix in range(len(kxs))],
                   [np.sum(obs[ix][1]) for ix in range(len(kxs))],
                   [np.sum(obs[ix][2]) for ix in range(len(kxs))]]
        print(repr(np.array(tot_sws)))
        # print(repr(np.array(sqs_stat)))
        sqs_stat = run_sq_static(config_file)
        print(repr(np.sum(sqs_stat, axis=0)))
        print(evs)
        filename = "dyn_struct_factor"
        foldername = io.get_exci_folder()
        from pathlib import Path
        obs_file = Path(foldername, filename).with_suffix(".npz")
        fig_name = Path(foldername, "sqw_perp").with_suffix(".pdf")

        if not obs_file.exists() or not sim_config.resume:
            def intensity_func(q, w, eta, ev, sw, amp=100000):
                # return amp*np.sum(np.array([np.exp(-1/eta*(w-ev[ia])**2)*sw[ia] for ia in range(len(sw))]))
                return amp*np.sum(np.array([1/np.pi*eta/((w-ev[ia])**2+eta**2)*sw[ia] for ia in range(len(sw))]))

            eta0 = 0.01
            max_freq = np.nanmax(np.array([np.nanmax(np.array(evs_full[xk])) for xk in range(len(kxs))]))
            freq = np.linspace(0, max_freq, 200)
            XK, FREQ = np.meshgrid(np.arange(len(kxs)), freq)
            DSSF_SPEC = np.zeros((*np.shape(XK), 3))
            for i in range(np.shape(XK)[0]):
                for j in range(np.shape(XK)[1]):
                    for s in range(3):
                        # DSSF_SPEC[i, j] = intensity_func(kxs[j], freq[i], 0.01, evs[j], obs[j][-1])
                        DSSF_SPEC = DSSF_SPEC.at[i, j, s].set(intensity_func(kxs[j], freq[i], eta0, evs_full[j], obs[j][s]))
            np.savez(obs_file, spectrum=DSSF_SPEC, omega=freq, elowest=evs)
        else:
            data = np.load(obs_file, allow_pickle=True)
            DSSF_SPEC = data["spectrum"]
            freq = data["omega"]
            evs = data["elowest"] # inhomogeneous
            XK, FREQ = np.meshgrid(np.arange(DSSF_SPEC.shape[1]), freq)

        # plt.pcolormesh(XK, FREQ, DSSF_SPEC[:, :, 0])
        if is_plot:
            n_vec = np.array([1, 0, 1])
            n_vec = n_vec / np.linalg.norm(n_vec)
            DSSF_SPEC_TRANSVERSE = DSSF_SPEC @ (n_vec**2)
            # plt.pcolormesh(XK, FREQ, DSSF_SPEC_TRANSVERSE)
            # plt.plot(np.arange(len(kxs)), evs, color='white', ls='--', label=r"$\text{min}_{\alpha} \omega_\alpha(k)$")
            import matplotlib.colors as colors
            DSSF_SPEC_TRANSVERSE = DSSF_SPEC_TRANSVERSE / np.max(DSSF_SPEC_TRANSVERSE)
            plt.pcolormesh(XK, FREQ, DSSF_SPEC_TRANSVERSE, cmap="turbo", rasterized=True, linewidth=0,
                           norm=colors.LogNorm(vmin=DSSF_SPEC_TRANSVERSE.min(), vmax=DSSF_SPEC_TRANSVERSE.max()))

            plt.xticks(**plot_info["xticks"])
            # plt.title(rf"$S^{{\perp}}$ TLHAFM D={sim_config.D}")
            # plt.title(rf"$S^{{\perp}}$ XXZ $\Delta_z$={sim_config.model_params['Delta']} D={sim_config.D}")
            plt.title(rf"$S^{{\perp}}$ D={sim_config.D}")
            plt.xlabel("k")
            plt.ylabel("$\omega$")
        if is_savefig:
            plt.savefig(fig_name)
        else:
            plt.show()


class iPEPSExciSimulation:
    """Simulation class for the excited-state simulation

    Call an instance of this class directly to start the simulation
    """

    def __init__(self, config_file, momentum_ix):
        self.config_file = config_file
        self.momentum_ix = momentum_ix

    @property
    def basis_size(self):
        with open(self.config_file) as f:
            cfg = safe_load(f)
        sim_config.from_dict(cfg)
        base_file = io.get_exci_base_file()
        base_sim = np.load(base_file, allow_pickle=True)
        basis = base_sim["basis"]
        return basis.shape[1]

    def __call__(self, ix, v=None):
        print(f"Starting simulation of basis vector {ix+1}/{self.basis_size}")
        with open(self.config_file) as f:
            cfg = safe_load(f)
        sim_config.from_dict(cfg)

        base_file = io.get_exci_base_file()
        base_sim = np.load(base_file, allow_pickle=True)
        basis = np.complex_(base_sim["basis"])
        peps = base_sim["peps"].item()
        if v is None:
            v = basis[:, ix]
            # if sim_config.unit_cell_phase:
            #     # Add the phase factor within the unit cell
            #     D = sim_config.D
            #     kxs, kys = make_momentum_path(sim_config.momentum_path)
            #     sim_config.px = kxs[self.momentum_ix]
            #     sim_config.py = kys[self.momentum_ix]
            #     v = np.reshape(v, (2, 2, 2, D, D, D, D))
            #     idp = np.array([[1, 0], [0, 1]])
            #     phase1 = 1j*(2*sim_config.px+sim_config.py)/3
            #     phase2 = 1j*(sim_config.px+2*sim_config.py)/3
            #     v = ncon((v, idp * np.exp(phase1), idp * np.exp(phase2)), ([-1, 1, 2, -4, -5, -6, -7], [-2, 1], [-3, 2]))
            #     v = np.reshape(v, (-1))
        res, grad_H = value_and_grad(peps.run, has_aux=True)(v)
        grad_H = grad_H.conj()
        print("Res", res, level=2)
        grad_N = res[1].pack_data()
        print("Grad H", grad_H, level=2)
        print("Grad N", grad_N, level=2)
        print(f"========== \nFinished basis vector {ix+1}/{self.basis_size} \n")
        return basis.T @ jax.lax.stop_gradient(grad_H), basis.T @ jax.lax.stop_gradient(
            grad_N
        )

    def check_grads(self, A=None):
        with open(self.config_file) as f:
            cfg = safe_load(f)
        sim_config.from_dict(cfg)

        base_file = io.get_exci_base_file()
        base_sim = np.load(base_file, allow_pickle=True)
        basis = np.complex_(base_sim["basis"])
        peps = base_sim["peps"].item()
        print("Checking gradient")
        # peps.fill(A)
        check_grads(peps.run_gc, (A,), order=1, modes="rev")
        print("Done check")
