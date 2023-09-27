""" 2D Heisenberg model """

import jax.numpy as np

import adpeps.ipeps.config as sim_config
from adpeps.utils.tlist import set_pattern
from adpeps.utils.tlist import TList

from .common import sigmam, sigmap, sigmaz
from .hamiltonian import Hamiltonian


# from jax.scipy.linalg import expm
import jax.scipy as sc

name = "spin-1/2 XXZ model on distorted triangular lattice"


def setup():
    """Returns the Hamiltonian"""
    H = make_hamiltonian(**sim_config.model_params)

    obs = None
    return H, obs


def make_hamiltonian(J=1, Jzigzag=1, delta=1, deltaxy=0, deltazigzag=0, swap=None, B_ext=0):
    """Heisenberg model"""

    # if anisotropy == "sy":
    #     H = (
    #         tprod(sigmaz, sigmaz) / 4
    #         + (1 + Delta) * (tprod(sigmap, sigmam)
    #                         + tprod(sigmam, sigmap)) / 4
    #         + (1 - Delta) * (tprod(sigmap, sigmap)
    #                         + tprod(sigmam, sigmam)) / 4
    #     )
    # else:
    #     H = (
    #         Delta * tprod(sigmaz, sigmaz) / 4
    #         + (tprod(sigmap, sigmam) + tprod(sigmam, sigmap)) / 2
    #     )

    Je_vec = np.array([1, 1 + delta, 1 - deltaxy])
    Jo_vec = np.array([1 - deltaxy, 1 + delta, 1])
    Jp_vec = np.array([1 - deltazigzag, 1, 1 - deltazigzag])

    if swap == "xy":
        inds = np.array([1, 0, 2])
        Je_vec = Je_vec[inds]
        Jo_vec = Jo_vec[inds]
        Jp_vec = Jp_vec[inds]
    elif swap == "yzx":
        inds = np.array([1, 2, 0])
        Je_vec = Je_vec[inds]
        Jo_vec = Jo_vec[inds]
        Jp_vec = Jp_vec[inds]

    Heven = J * (
            Je_vec[2] * tprod(sigmaz, sigmaz) / 4
            + (Je_vec[0] + Je_vec[1]) * (tprod(sigmap, sigmam)
                            + tprod(sigmam, sigmap)) / 4
            + (Je_vec[0] - Je_vec[1]) * (tprod(sigmap, sigmap)
                            + tprod(sigmam, sigmam)) / 4
    )

    Hodd = J * (
            Jo_vec[2] * tprod(sigmaz, sigmaz) / 4
            + (Jo_vec[0] + Jo_vec[1]) * (tprod(sigmap, sigmam)
                            + tprod(sigmam, sigmap)) / 4
            + (Jo_vec[0] - Jo_vec[1]) * (tprod(sigmap, sigmap)
                            + tprod(sigmam, sigmam)) / 4
    )

    Hzigzag = Jzigzag * (
            Jp_vec[2] * tprod(sigmaz, sigmaz) / 4
            + (Jp_vec[0] + Jp_vec[1]) * (tprod(sigmap, sigmam)
                            + tprod(sigmam, sigmap)) / 4
            + (Jp_vec[0] - Jp_vec[1]) * (tprod(sigmap, sigmap)
                            + tprod(sigmam, sigmam)) / 4
    )

    # H = (
    #     tprod(sigmaz, sigmaz) / 4
    #     + (tprod(sigmap, sigmam) + tprod(sigmam, sigmap)) / 2
    # )
    # H_ext_ydir = -1j * B_ext * (sigmap - sigmam) / 2
    # the pattern of Hamiltonian is different from the state!
    pattern = np.array([[0, 1], [0, 1]])
    H = Hamiltonian(pattern=pattern)

    H.fill((1, 0), Hzigzag, tag="H_nn_h")  # inter chain horizontal
    H.fill((-1, 1), Hzigzag, tag="H_nn_d")  # inter chain diagonal

    # even chain
    H[((0, 0), (0, 1))] = Heven
    H[((0, 1), (0, 2))] = Heven
    # H[((0, 1), (0, 2))] = Heven
    H[((1, 0), (1, 1))] = Hodd
    H[((1, 1), (1, 2))] = Hodd
    # H[((1, 1), (1, 2))] = Hodd

    return H


def make_obs(spin=0.5):
    return 0.5 * (sigmap + sigmam), 0.5j * (sigmam - sigmap), 0.5 * sigmaz


def tprod(a, b):
    return np.outer(a, b).reshape([2, 2, 2, 2], order="F").transpose([0, 2, 1, 3])
