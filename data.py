import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from aviary.roost.model import DescriptorNetwork
from pymatgen.core import Composition
from torch import Tensor, LongTensor

elem_embs = pd.read_json(
    "https://raw.githubusercontent.com/CompRhys/aviary/refs/heads/main/aviary/embeddings/element/onehot112.json"
)


cache = {}


def comp2graph(composition):
    if composition in cache:
        return (torch.clone(x) for x in cache[composition])
    comp_dict = Composition(composition).get_el_amt_dict()
    elements = list(comp_dict)

    weights = list(comp_dict.values())
    weights = np.atleast_2d(weights).T / np.sum(weights)

    try:
        elem_fea = np.vstack([elem_embs[elements]]).T
    except AssertionError as exc:
        raise AssertionError(
            f"{composition} contains element types not in embedding"
        ) from exc
    except ValueError as exc:
        raise ValueError(
            f"{composition} composition cannot be parsed into elements"
        ) from exc

    n_elems = len(elements)
    self_idx = []
    nbr_idx = []
    for elem_idx in range(n_elems):
        self_idx += [elem_idx] * n_elems
        nbr_idx += list(range(n_elems))

    # convert all data to tensors
    elem_weights = torch.tensor(weights).float()
    elem_fea = torch.tensor(elem_fea).float()
    self_idx = torch.tensor(self_idx).long()
    nbr_idx = torch.tensor(nbr_idx).long()
    ans = (elem_weights, elem_fea, self_idx, nbr_idx)
    cache[composition] = ans
    return ans


# https://github.com/CompRhys/aviary/blob/181e2b2b2d679a12f6dbb430853d92508e8d71f2/aviary/roost/data.py#L140C1-L212C6
def collate_batch(samples):
    # define the lists
    batch_elem_weights = []
    batch_elem_fea = []
    batch_self_idx = []
    batch_nbr_idx = []
    crystal_elem_idx = []

    cry_base_idx = 0
    for idx, inputs in enumerate(samples):
        elem_weights, elem_fea, self_idx, nbr_idx = inputs

        n_sites = elem_fea.shape[0]  # number of atoms for this crystal

        # batch the features together
        batch_elem_weights.append(elem_weights)
        batch_elem_fea.append(elem_fea)

        # mappings from bonds to atoms
        batch_self_idx.append(self_idx + cry_base_idx)
        batch_nbr_idx.append(nbr_idx + cry_base_idx)

        # mapping from atoms to crystals
        crystal_elem_idx.append(torch.ones(n_sites, dtype=torch.long) * idx)

        # increment the id counter
        cry_base_idx += n_sites

    return (
        torch.cat(batch_elem_weights, dim=0),
        torch.cat(batch_elem_fea, dim=0),
        torch.cat(batch_self_idx, dim=0),
        torch.cat(batch_nbr_idx, dim=0),
        torch.cat(crystal_elem_idx),
    )
