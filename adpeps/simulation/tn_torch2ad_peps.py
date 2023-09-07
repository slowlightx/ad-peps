"""
    iPEPS module for optimization with CTM

    For an example of how to run a simulation see :mod:`adpeps.simulation.run_ipeps_gs`

    The module is initialized from one of the specific 
    model files, which return the initial boundary and 
    site tensors

    The list of parameters is set to the elements of the 
    individual site tensors

    Conventions for indices:

        - Site tensors::

            A: [phys, right, top, left, bottom]

        - Boundary tensors::

            C1: [right, bottom]
            C2: [left,  bottom]
            C3: [top,   left]
            C4: [right, top]
            T1: [right, left, ket, bra]
            T2: [top,   bottom, ket, bra]
            T3: [right, left, ket, bra]
            T4: [top,   bottom, ket, bra]


    Order of boundary tensors::

        C1 - T1 - C2
        |    |    |
        T4 - A  - T2
        |    |    |
        C4 - T3 - C3
"""

import jax
import jax.numpy as np
# from jax import random

import numpy as onp
import adpeps.ipeps.config as sim_config
from yaml import dump, safe_load
# from adpeps.ipeps import models

from adpeps.ipeps.ipeps import iPEPS

from adpeps.utils import io

from math import sqrt, pi
import torch
import scipy
from scipy.stats import ortho_group

import json


def read_ipeps_trgl_tntorch(jsonfile, aux_seq=[3,0,1,2]):
    """
    :param jsonfile:
    :param aux_seq:
    :return: sites

    Translate between different conventions for virtual index labelling in tn-torch and ad-peps :
    tn-torch:                 ad-peps:
       1 0                       2 0
       |/                        |/
    2--A--4                   3--A--1
       |                         |
       3                         4
    Permute original sites by (0,4,1,2,3).
    """

    sites = {}
    with open(jsonfile) as j:
        raw_state = json.load(j)
        # import pdb; pdb.set_trace()

        WARN_REAL_TO_COMPLEX = False
        asq = [x + 1 for x in aux_seq]

        # Loop over non-equivalent tensor,site pairs in the unit cell
        for ts in raw_state["map"]:
            coord = (ts["x"], ts["y"])

            # find the corresponding tensor (and its elements)
            # identified by "siteId" in the "sites" list
            t = None
            for s in raw_state["sites"]:
                if s["siteId"] == ts["siteId"]:
                    t = s
            if t == None:
                raise Exception("Tensor with siteId: " + ts["sideId"] + " NOT FOUND in \"sites\"")

                # depending on the "format", read the bare tensor
            # if "format" in t.keys():
            #     if t["format"] == "1D":
            #         X = torch.from_numpy(read_bare_json_tensor_np(t))
            # else:
            #     # default
            #     X = torch.from_numpy(read_bare_json_tensor_np_legacy(t))
            # sites[coord] = X.permute((0, *asq))
            X = read_bare_json_tensor_np_legacy(t)
            sites[coord] = np.array(onp.transpose(X, (0, *asq)))

        return sites

def read_ipeps_trgl_tntorch_1site(jsonfile, aux_seq=[3,0,1,2]):
    """
    :param jsonfile:
    :param aux_seq:
    :return: sites

    Translate between different conventions for virtual index labelling in tn-torch and ad-peps :
    tn-torch:                 ad-peps:
       1 0                       2 0
       |/                        |/
    2--A--4                   3--A--1
       |                         |
       3                         4
    Permute original sites by (0,4,1,2,3).
    """
    sp = torch.zeros((2, 2))
    sm = torch.zeros((2, 2))
    sp[0, 1] = 1
    sm[1, 0] = 1
    rot_op = torch.linalg.matrix_exp((-pi/3)*(sp-sm)).numpy()
    print(torch.linalg.matrix_exp((-pi/3)*(sp-sm)))
    print(torch.linalg.matrix_exp((-pi/2)*(sp-sm)))
    rot_op = np.array([[1/2, -sqrt(3)/2], [sqrt(3)/2, 1/2]])
    print(rot_op@rot_op@rot_op)
    sites = {}
    with open(jsonfile) as j:
        raw_state = json.load(j)
        # import pdb; pdb.set_trace()
        assert len(raw_state["map"]) == 1, f'Unmatched number of sites! Expected 1, got {len(raw_state["map"])}.'

        WARN_REAL_TO_COMPLEX = False
        asq = [x + 1 for x in aux_seq]

        # Loop over non-equivalent tensor,site pairs in the unit cell
        for ts in raw_state["map"]:
            coord = (ts["x"], ts["y"])
            # find the corresponding tensor (and its elements)
            # identified by "siteId" in the "sites" list
            t = None
            for s in raw_state["sites"]:
                if s["siteId"] == ts["siteId"]:
                    t = s
            if t == None:
                raise Exception("Tensor with siteId: " + ts["sideId"] + " NOT FOUND in \"sites\"")

                # depending on the "format", read the bare tensor
            # if "format" in t.keys():
            #     if t["format"] == "1D":
            #         X = torch.from_numpy(read_bare_json_tensor_np(t))
            # else:
            #     # default
            #     X = torch.from_numpy(read_bare_json_tensor_np_legacy(t))
            # sites[coord] = X.permute((0, *asq))
            X = read_bare_json_tensor_np_legacy(t)
            sites[coord] = np.array(onp.transpose(X, (0, *asq)))
            # q_, new_X = gauge_fix(X, X.shape[1])
            # print(new_X)
        site0 = sites[(0, 0)]
        # sites[(1, 0)] = rot_op@site0
        # sites[(2, 0)] = rot_op@(rot_op@site0)
        # sites[(1, 0)] = np.einsum('xa,xijkl->aijkl', rot_op, site0)
        # sites[(2, 0)] = np.einsum('xa,xijkl->aijkl', rot_op@rot_op, site0)
        sites[(1, 0)] = np.einsum('ax,xijkl->aijkl', rot_op.T, site0)
        sites[(2, 0)] = np.einsum('ax,xijkl->aijkl', rot_op.T@rot_op.T, site0)
        return sites


# def constraint_func(q_vec):
#     if torch.is_tensor(q_vec):
#         q_mat = q_vec.reshape(2,2)
#     else:
#         q_mat = torch.from_numpy(q_vec.reshape(2,2))
#     return torch.norm(torch.inverse(q_mat)@q_mat-torch.eye(2), p=2)
#
#
# def imag_res(q, site):
#     q = torch.from_numpy(q)
#     q= torch.complex(q,q*0)
#     q_inv = torch.inverse(q).contiguous()
#     print(q.dtype)
#     new_site = torch.einsum('axjyl,ix,yk->aijkl', site, q, q_inv)
#     return sum([val.imag for val in new_site])
#
#
# def gauge_fix(t, ad):
#     q0 = torch.from_numpy(ortho_group.rvs(ad))
#     q0 = torch.complex(q0, q0*0)
#     # q0 = ortho_group.rvs(ad)
#     ortho_cons = scipy.optimize.NonlinearConstraint(constraint_func, -1e-16, 1e-16)
#     t = torch.from_numpy(t)
#     optim_res = scipy.optimize.minimize(imag_res, q0, args=(t), method='trust-constr', constraints=ortho_cons,
#                                         options={"maxiter": 5000, "xtol": 1e-16, "gtol": 1e-16})
#     qsol = optim_res.x
#     qsol_inv = torch.inverse(qsol).contiguous()
#     new_t = torch.einsum('axjyl,ix,yk->aijkl', qsol, qsol_inv, t)
#     return optim_res.x, new_t


# def read_bare_json_tensor_np(json_obj):
#     dtype_str = json_obj["dtype"].lower()
#     assert dtype_str in ["float64", "complex128"], "Invalid dtype" + dtype_str
#     dims = json_obj["dims"]
#     raw_data = json_obj["data"]
#
#     # convert raw_data list[str] into list[dtype]
#     if "complex" in dtype_str:
#         raw_data = np.asarray(raw_data, dtype=np.complex128)
#     else:
#         raw_data = np.asarray(raw_data, dtype=np.float64)
#
#     return raw_data.reshape(dims)


def read_bare_json_tensor_np_legacy(json_obj):
    t = json_obj
    # 0) find dtype, else assume float64
    dtype_str = json_obj["dtype"].lower() if "dtype" in t.keys() else "float64"
    assert dtype_str in ["float64", "complex128"], "Invalid dtype" + dtype_str

    # 1) find the dimensions of indices
    if "dims" in t.keys():
        dims = t["dims"]
    else:
        # assume all auxiliary indices have the same dimension
        dims = [t["physDim"], t["auxDim"], t["auxDim"], t["auxDim"], t["auxDim"]]

    X = onp.zeros(dims, dtype=dtype_str)

    # 1) fill the tensor with elements from the list "entries"
    # which list the non-zero tensor elements in the following
    # notation. Dimensions are indexed starting from 0
    #
    # index (integer) of physDim, left, up, right, down, (float) Re, Im
    #                             (or generic auxilliary inds ...)
    if dtype_str == "complex128":
        for entry in t["entries"]:
            l = entry.split()
            X[tuple(int(i) for i in l[:-2])] = float(l[-2]) + float(l[-1]) * 1.0j
    else:
        for entry in t["entries"]:
            l = entry.split()
            k = 1 if len(l) == len(dims) + 1 else 2
            X[tuple(int(i) for i in l[:-k])] += float(l[-k])
    return X


dir = r"/Users/slowlight/PycharmProjects/tn/TensorNetwork"

inputfile = rf"{dir}/triangular/states/trglC_j11.0_D2_1SITE_C4X4_state.json"
inputfile = rf"{dir}/temp/test_trgl_1site_d2_state.json"
# outputfile = rf"{dir}/ad-peps/simulations/gs/xxz_trgl_D2_X31.npz"
outputfile = rf"{dir}/ad-peps/simulations/gs/xxz_trgl_D2_X31_raw.npz"


# config_file = rf"{dir}/ad-peps/examples/xxz_trgl_D2.yaml"
# print(config_file)
# with open(config_file) as f:
#     cfg = safe_load(f)
#
# # Show options
# print(dump(cfg))
#
# # Load the configuration file into the sim_config object
# sim_config.from_dict(cfg)

# 3SITE
pattern = [
    [0,1,2],
    [2,0,1],
    [1,2,0],
  ]


# 1SITE
# sites = read_ipeps_trgl_tntorch(inputfile, aux_seq=[1,2,3,0])
sites = read_ipeps_trgl_tntorch_1site(inputfile, aux_seq=[3,0,1,2])
coords2sublat = {(0, 0): 0, (1, 0): 1, (2, 0): 2}

new_sites = {}
for i in sites.keys():
    new_sites[coords2sublat[i]] = sites[i]

np.savez(outputfile, sites=new_sites)

# adpeps_state = iPEPS()
# import pdb; pdb.set_trace()
#
# adpeps_state.fill(new_sites)
#
# np.savez(outputfile, peps=adpeps_state)
