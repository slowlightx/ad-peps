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

import numpy as onp
from yaml import dump, safe_load

from adpeps.utils import io
from pathlib import Path

from math import sqrt, pi
import torch

import json
import argparse


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
        site0 = sites[(0, 0)]
        # sites[(1, 0)] = np.einsum('xa,xijkl->aijkl', rot_op, site0)
        # sites[(2, 0)] = np.einsum('xa,xijkl->aijkl', rot_op@rot_op, site0)
        sites[(1, 0)] = np.einsum('ax,xijkl->aijkl', rot_op, site0)
        sites[(2, 0)] = np.einsum('ax,xijkl->aijkl', rot_op@rot_op, site0)
        return sites


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

parser = argparse.ArgumentParser()
parser.add_argument("--j1", type=float, default=1., help="nearest neighbour interaction (strong plaquettes)")
parser.add_argument('--j2', type=float, default=0.0, help='next-nearest neighbor interaction')
parser.add_argument('--bond_dim', type=int, default=2, help='bond dimension')
parser.add_argument('--chi', type=int, default=40, help='environmental bond dimension')
parser.add_argument('--sitetype', type=str, default='1SITE', help='site type of input state', choices=['1SITE', '2SITE', '3SITE'])
parser.add_argument('--model', type=str, default='trgl', help='model type')
parser.add_argument('--instate', type=str, default=None, help='site type')

args, unknown_args = parser.parse_known_args()

if args.j2 == 0.0:
    j2 = 0
else:
    j2 = args.j2
D = args.bond_dim
chi = args.chi
sitetype = args.sitetype

if args.instate is not None:
    inputfile = io.localize_data_file(args.instate)
else:
    inputfile = io.localize_data_file(f"raw/J1J2_uncompressed_1site_D{D}/trglC_j11.0_j2{j2}_1SITE_C4X4_state.json")

# For workstation
filename = fr"cplx_J2_{j2}_{args.model}_D{D}_X{chi}_raw".replace('.', 'd')  # cluster
config_filename = fr"J2_{j2}_{args.model}_D{D}".replace('.', 'd')
config_filename = fr"cplx_trgl_D2".replace('.', 'd')

filename = Path("gs", filename)

outputfile = io.localize_data_file(filename).with_suffix(".npz")
config_file = io.localize_config_file(config_filename).with_suffix(".yaml")

with open(config_file) as f:
    cfg = safe_load(f)

# Show options
print(dump(cfg))

# 3SITE
# pattern = [
#     [0,1,2],
#     [2,0,1],
#     [1,2,0],
#   ]
pattern = cfg['pattern']
print(pattern)
if sitetype == '1SITE':
    sites = read_ipeps_trgl_tntorch_1site(inputfile, aux_seq=[3,0,1,2])
else:
    sites = read_ipeps_trgl_tntorch(inputfile, aux_seq=[3,0,1,2])

if max(pattern) == 2:
    coords2sublat = {(0, 0): 0, (1, 0): 2, (2, 0): 1}
elif max(pattern) == 1:
    coords2sublat = {(0, 0): 0, (1, 0): 1}
else:
    coords2sublat = {(0, 0): 0, (1, 0): 2, (2, 0): 1}

new_sites = {}
for i in sites.keys():
    new_sites[coords2sublat[i]] = sites[i]
np.savez(outputfile, sites=new_sites)
