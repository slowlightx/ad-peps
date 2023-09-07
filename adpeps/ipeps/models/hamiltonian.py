import jax.numpy as np
from adpeps.utils.tlist import TList
from adpeps.utils.empty_tensor import EmptyT


class LocalHamiltonian(dict):
    pass


class Hamiltonian:
    def __init__(self, H_base=None, pattern=None, unit_cell=None):
        self.shape = None
        self.unit_cell = unit_cell
        self._H = TList(pattern=pattern)
        for i in range(len(self._H)):
            self._H._data[i] = LocalHamiltonian()
        if H_base is not None:
            self.fill_H_base(H_base)

    def l_ix(self, *ij):
        i, j = ij
        return np.ravel_multi_index(self.mod_ij(i, j), self.unit_cell)

    def mod_ij(self, *ij):
        i, j = ij
        return (i % self.unit_cell[0], j % self.unit_cell[1])

    def fill_H_base(self, H_base):
        """Fills the Hamiltonian with one term that is
        used for both horizontal (2,1)-shape and
        vertical (1,2)-shape
        """
        self.fill((1, 0), H_base)
        self.fill((0, 1), H_base)

    def _conv_ix(self, ix):
        loc = TList._loc

        if len(loc) == 1:
            shift_i, shift_j = np.unravel_index(loc[0], self.unit_cell, order="F")
        else:
            shift_i, shift_j = loc
        i = ix[0] + shift_i
        j = ix[1] + shift_j
        linear_ix = self.l_ix(i, j)

        return linear_ix

    def __setitem__(self, ix, value):
        # loc = TList._loc
        # ix1 = (ix[0][0] + loc[0], ix[0][1] + loc[1])
        # ix2 = (ix[1][0] + loc[0], ix[1][1] + loc[1])
        # l_ix1 = self.l_ix(*ix1)
        # l_ix2 = self.l_ix(*ix2)
        # l_ix1 = self._conv_ix(ix[0])
        # l_ix2 = self._conv_ix(ix[1])
        # self._H[l_ix1][l_ix2] = value
        dx = ix[1][0] - ix[0][0]
        dy = ix[1][1] - ix[0][1]
        # if not hasattr(self, 'fermionic'):
        #     self.fermionic = value.fermionic
        # print(f"Setting {ix[0]} with shape {(dx,dy)}")
        self._H[ix[0]][(dx, dy)] = value

    def __getitem__(self, ix):
        # loc = TList._loc
        # ix1 = (ix[0][0] + loc[0], ix[0][1] + loc[1])
        # ix2 = (ix[1][0] + loc[0], ix[1][1] + loc[1])
        # l_ix1 = self.l_ix(*ix1)
        # l_ix2 = self.l_ix(*ix2)
        # l_ix1 = self._conv_ix(ix[0])
        # l_ix2 = self._conv_ix(ix[1])
        # return self._H[l_ix1][l_ix2]
        dx = ix[1][0] - ix[0][0]
        dy = ix[1][1] - ix[0][1]
        # print(f"Getting {ix[0]} with shape {(dx,dy)}")
        # return self._H[ix[0]].get((dx, dy), EmptyT())
        # print(ix)
        # print(self._H[ix[0]].keys())
        return self._H[ix[0]].get((dx, dy))

    def fill(self, shape, H_term, tag=None):
        for h in self._H:
            H = H_term.copy()
            h[shape] = H
            self.shape = H.shape

    def __str__(self):
        strrep = ""
        unit_cell = self._H.size
        for j in range(unit_cell[1]):
            for i in range(unit_cell[0]):
                strrep += str((i, j)) + " -- "
                h = self[(i, j), (i + 1, j)]
                if h != EmptyT():
                    hname = "<h>"
                else:
                    hname = " - "
                strrep += hname + " -- "
            strrep += "\n"
            strrep += "  |\n  |\n"
            h = self[(i, j), (i, j + 1)]
            if h != EmptyT():
                hname = "<v>"
            else:
                hname = " - "
            strrep += "  " + hname + "\n  |\n  |\n"

        return strrep
