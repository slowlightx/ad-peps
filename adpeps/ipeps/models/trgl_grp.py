""" 2D Heisenberg model """

import jax.numpy as np

import adpeps.ipeps.config as sim_config
from adpeps.utils.tlist import set_pattern
from adpeps.utils.tlist import TList

from .common import sigmam, sigmap, sigmaz, id2
from .hamiltonian import Hamiltonian


# from jax.scipy.linalg import expm
import jax.scipy as sc

name = "spin-1/2 XXZ model on triangular lattice"


def setup():
    """Returns the Hamiltonian"""
    H = make_hamiltonian(**sim_config.model_params)

    obs = None
    return H, obs


def make_hamiltonian(J=1, Delta=1, J2=0, anisotropy=None, B_ext=0):
    """Heisenberg model"""

    if anisotropy == "sy":
        H2 = (
            tprod(sigmaz, sigmaz) / 4
            + (1 + Delta) * (tprod(sigmap, sigmam)
                            + tprod(sigmam, sigmap)) / 4
            + (1 - Delta) * (tprod(sigmap, sigmap)
                            + tprod(sigmam, sigmam)) / 4
        )
    else:
        H2 = (
            Delta * tprod(sigmaz, sigmaz) / 4
            + (tprod(sigmap, sigmam) + tprod(sigmam, sigmap)) / 2
        )
    # H = (
    #     tprod(sigmaz, sigmaz) / 4
    #     + (tprod(sigmap, sigmam) + tprod(sigmam, sigmap)) / 2
    # )
    # H_ext_ydir = -1j * B_ext * (sigmap - sigmam) / 2

    H3 = np.einsum('ijab,kc->ijkabc', H2, id2) + np.einsum('ikac,jb->ijkabc', H2, id2) + np.einsum('jkbc,ia->ijkabc', H2, id2)
    H_tri_nn = J * H3
    H_tri_nnn = J2 * H3
    # H_1x0y, H_0x1y, H_n1x1y, H_n2x1y, H_n1x2y, H_1x1y = J*H, J*H, J*H, J2*H, J2*H, J2*H

    # pattern = np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0]])
    pattern = np.array([[0]])
    H = Hamiltonian(pattern=pattern)

    H.fill((1, 1), H_tri_nn, tag="H_tria_nn")  # H_tria_nn
    H.fill((2, 1), H_tri_nn, tag="H_trib_nn")  # H_trib_nn
    H.fill((1, 2), H_tri_nn, tag="H_tric_nn")  # H_tric_nn

    if abs(J2) > 0:
        H.fill((3, 0), H_tri_nnn, tag="H_tria_nnn")  # H_tria_nnn
        H.fill((4, 0), H_tri_nnn, tag="H_trib_nnn")  # H_trib_nnn
        H.fill((3, 1), H_tri_nnn, tag="H_tric_nnn")  # H_tric_nnn

    # H.fill((0, 1), H2, tag="H2")  # 2-site

    return H


def make_obs(spin=0.5):
    return 0.5 * (sigmap + sigmam), 0.5j * (sigmam - sigmap), 0.5 * sigmaz


def tprod(a, b):
    return np.outer(a, b).reshape([2, 2, 2, 2], order="F").transpose([0, 2, 1, 3])


def tprod3(a, b, c):
    return np.einsum('ia,jb,kc->ijkabc', a, b, c)