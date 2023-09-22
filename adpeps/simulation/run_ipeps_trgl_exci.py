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
from adpeps.ipeps.ipeps_xxz_trgl import iPEPS, iPEPS_exci
from adpeps.ipeps.make_momentum_path import make_momentum_path
from adpeps.utils import io
from adpeps.utils.printing import print
from adpeps.utils.tlist import TList, cur_loc, set_pattern


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


def evaluate_spectral_weight(config_file, momentum_ix):
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

    sx = np.array([[0, 0.5], [0.5, 0]])
    sy = np.array([[0, -0.5], [0.5, 0]])
    sz = np.array([[0.5, 0], [0, -0.5]])
    ops = [sx, sy, sz]

    A = peps.tensors.A
    gs_with_ops = []
    for op in ops:
        gs = None
        nrms = TList(shape=A.size, pattern=A.pattern)
        for i in A.x_major():
            with cur_loc(i):
                if not nrms.is_changed(0, 0):
                    nrms.mark_changed(i)
                    if gs is None:
                        gs = op @ A[0, 0]
                    else:
                        gs = np.concatenate((gs, A[0, 0]), axis=0)
        gs = np.reshape(gs, (-1))
        gs_with_ops.append(gs)

    basis2 = basis @ P @ N2
    spectral_weight = []
    for gs_with_op in gs_with_ops:
        sw = abs(basis2.T @ gs_with_op)**2
        spectral_weight.append(sw[ixs])
    return spectral_weight, ev.real


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

    plot_spectrum = False
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
                sqw, ev = evaluate_spectral_weight(config_file, ix)
            except:
                ev = [np.nan]
                sqw = [np.nan]
            evs_full.append(ev)
            evs.append(ev[0])
            obs.append(sqw)

        filename = "dyn_struct_factor"
        foldername = io.get_exci_folder()
        from pathlib import Path
        obs_file = Path(foldername, filename)

        if not obs_file.exists() or not cfg.resume:
            def intensity_func(q, w, eta, ev, sw, amp=100000):
                # return amp*np.sum(np.array([np.exp(-1/eta*(w-ev[ia])**2)*sw[ia] for ia in range(len(sw))]))
                return amp*np.sum(np.array([1/np.pi*eta/((w-ev[ia])**2+eta**2)*sw[ia] for ia in range(len(sw))]))

            eta0 = 0.02
            freq = np.arange(0, max([max(evs_full[xk]) for xk in range(len(kxs))])/2, 0.02)
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
        n_vec = np.array([1, 1, 0])
        n_vec = n_vec / np.linalg.norm(n_vec)
        DSSF_SPEC_TRANSVERSE = DSSF_SPEC @ (n_vec**2)
        plt.pcolormesh(XK, FREQ, DSSF_SPEC_TRANSVERSE)
        plt.plot(np.arange(len(kxs)), evs, color='white', ls='--', label=r"$\text{min}_{\alpha} \omega_\alpha(k)$")

        plt.xticks(**plot_info["xticks"])
        # plt.title(rf"$S^{{\perp}}$ TLHAFM D={sim_config.D}")
        plt.title(rf"$S^{{\perp}}$ XXZ $\Delta_z$={sim_config.model_params['Delta']} D={sim_config.D}")
        plt.xlabel("k")
        plt.ylabel("$\omega$")
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
