""" 2D Heisenberg model """

import jax.numpy as np

import adpeps.ipeps.config as sim_config
from adpeps.utils.tlist import set_pattern
from adpeps.utils.tlist import TList

from .common import sigmam, sigmap, sigmaz
from .hamiltonian import Hamiltonian


# from jax.scipy.linalg import expm
import jax.scipy as sc

name = "spin-1/2 XXZ model on triangular lattice"


def setup():
    """Returns the Hamiltonian"""
    H = make_hamiltonian(**sim_config.model_params)

    obs = None
    return H, obs


def make_hamiltonian(J=1, Delta=1, J2=0, sl_rot=None, anisotropy=None, q=1./3, B_ext=0):
    """Heisenberg model"""
    Rot_op = sc.linalg.expm((2*np.pi*q)*(sigmap - sigmam))
    if anisotropy == "sy":
        H = J * (
            tprod(sigmaz, sigmaz) / 4
            + (1 + Delta) * (tprod(sigmap, sigmam)
                            + tprod(sigmam, sigmap)) / 4
            + (1 - Delta) * (tprod(sigmap, sigmap)
                            + tprod(sigmam, sigmam)) / 4
        )
    else:
        H = J * (
            Delta * tprod(sigmaz, sigmaz) / 4
            + (tprod(sigmap, sigmam) + tprod(sigmam, sigmap)) / 2
        )
    # H = (
    #     tprod(sigmaz, sigmaz) / 4
    #     + (tprod(sigmap, sigmam) + tprod(sigmam, sigmap)) / 2
    # )
    # H_ext_ydir = -1j * B_ext * (sigmap - sigmam) / 2

    if sl_rot == "unitary":
        H_nn_h = J*np.einsum('ixjy,xa,yb->iajb', H, Rot_op, Rot_op)
        H_nn_v = J*np.einsum('ixjy,xa,yb->iajb', H, Rot_op, Rot_op)
        H_nn_diag = J*np.einsum('ixjy,xa,yb->iajb', H, Rot_op@Rot_op, Rot_op@Rot_op)
        pattern = np.array([[0]])
    else:
        H_nn_h, H_nn_v, H_nn_diag, H_nnn_h, H_nnn_v, H_nnn_diag = J*H, J*H, J*H, J2*H, J2*H, J2*H
        pattern = np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0]])
    # H = Hamiltonian(pattern=pattern)
    #
    # H.fill((1, 0), H_nn_h, tag="H_nn_h")  # H_nn_h
    # H.fill((0, 1), H_nn_v, tag="H_nn_v")  # H_nn_v
    # H.fill((1, 1), H_nn_diag, tag="H_nn_diag")  # H_nn_diag

    return H


def make_obs(spin=0.5):
    return 0.5 * (sigmap + sigmam), 0.5j * (sigmam - sigmap), 0.5 * sigmaz


def tprod(a, b):
    return np.outer(a, b).reshape([2, 2, 2, 2], order="F").transpose([0, 2, 1, 3])
