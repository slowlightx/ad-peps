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


def make_hamiltonian(J1=1, J2=0, J3=0, hx=0, hy=0, hz=0):
    """Kitaev model with external magnetic field
       Math: H = J_x S_x\cdot S_x + J_y S_y\cdot S_y + J_z S_z\cdot S_z - \vec{B} \cdot \vec{S}
    """

    Hxx = J1 * (tprod(sigmap, sigmam) + tprod(sigmam, sigmap)
                + tprod(sigmap, sigmap) + tprod(sigmam, sigmam)) / 4
    Hyy = J1 * (tprod(sigmap, sigmam) + tprod(sigmam, sigmap)
                - tprod(sigmap, sigmap) - tprod(sigmam, sigmam)) / 4
    Hzz = J1 * tprod(sigmaz, sigmaz) / 4

    H_ext = hx * (tprod(sigmap + sigmam, id2) + tprod(id2, sigmap + sigmam)) / 2 \
            - 1j * hy * (tprod(sigmap - sigmam, id2) + tprod(id2, sigmap - sigmam)) / 2 \
            + hz * (tprod(sigmaz, id2) + tprod(id2, sigmaz)) / 2

    pattern = np.array([[0]])  # within grouped sites
    H = Hamiltonian(pattern=pattern)

    # NN
    H.fill((1, 0), Hxx, tag="Sx--Sx bond")
    H.fill((0, 1), Hyy, tag="Sy--Sy bond")
    H.fill((1, 1), Hzz + H_ext, tag="Sz--Sz bond and field terms")
    # H[((0, 0), (1, 0))] = Hxx
    # H[((0, 0), (0, 1))] = Hyy
    # H[((0, 0), (1, 1))] = Hzz + H_ext

    # NNN
    # if abs(J2) > 0:
    #     H[((0, 0), (2, 1))] = H_2nn
    #     H[((0, 0), (1, 2))] = H_2nn
    #     H[((0, 0), (-1, 1))] = H_2nn
    #     H[((1, 1), (3, 2))] = H_2nn
    #     H[((1, 1), (2, 3))] = H_2nn
    #     H[((1, 1), (0, 2))] = H_2nn
    # NNNN
    # if abs(J3) > 0:
    #     H[((0, 0), (0, 2))] = H_3nn
    #     H[((0, 0), (2, 0))] = H_3nn
    #     H[((1, 1), (3, 3))] = H_3nn

    return H


def make_obs(spin=0.5):
    return 0.5 * (sigmap + sigmam), 0.5j * (sigmam - sigmap), 0.5 * sigmaz


def tprod(a, b):
    return np.outer(a, b).reshape([2, 2, 2, 2], order="F").transpose([0, 2, 1, 3])
