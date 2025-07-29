import numpy as np
import mdtraj as md
import torch
from rdkit import Chem

import sys
sys.path.append('..')
from utils.bio_utils import get_atype


def make_batch(state0file, xyz, bs, _format="pdb"):
    if _format == "pdb":
        state0 = md.load(state0file)
        top = state0.topology
        xyz = 10 * xyz
        atype = get_atype(top)  # (N,)
        # to tensor
        atype = torch.from_numpy(atype).long()
        x0 = torch.from_numpy(xyz).float()  # (N, 3)
        x0 = x0 - x0.mean(dim=0)    # CoM
    else:
        raise NotImplementedError(f'File format {_format} cannnot be recognized.')

    # batch id
    abid = torch.tensor([[i] * atype.shape[0] for i in range(bs)], dtype=torch.long).flatten()
    # mask
    mask = torch.tensor([[1] * atype.shape[0] for _ in range(bs)], dtype=torch.bool).flatten()
    # edge mask
    edge_mask = torch.tensor([[0] * atype.shape[0] for _ in range(bs)], dtype=torch.long).flatten()

    batch = {
        "atype": atype.repeat(bs),
        "x0": x0.repeat(bs, 1),
        "abid": abid,
        "mask": mask,
        "edge_mask": edge_mask
    }

    return batch


def make_batch_complex(protein_state0_path, ligand_state0_path, bs):
    # preprocess protein
    prot_state0 = md.load(protein_state0_path)
    top = prot_state0.toppology
    Xp = 10 * prot_state0.xyz[0]
    Zp = get_atype(top)
    Np = Xp.shape[0]

    # preprocess ligand
    lig_state0 = Chem.SDMolSupplier(ligand_state0_path, removeHs=False, sanitize=False)[0]
    Nl = lig_state0.GetNumAtoms()
    lig_conf = lig_state0.GetConformer()
    Xl = np.array([lig_conf.GetAtomPosition(i) for i in range(Nl)], dtype=float)
    Zl = np.array([atom.GetAtomicNum() - 1 for atom in lig_state0.GetAtoms()], dtype=np.compat.long)    # H - 0

    atype = torch.from_numpy(np.concatenate([Zp, Zl], dtype=np.compat.long)).long()
    x0 = torch.from_numpy(np.concatenate([Xp, Xl], dtype=float)).float()

    # batch id
    abid = torch.tensor([[i] * atype.shape[0] for i in range(bs)], dtype=torch.long).flatten()
    # mask
    mask = torch.tensor([[1] * atype.shape[0] for _ in range(bs)], dtype=torch.bool).flatten()
    # edge mask
    edge_mask = torch.tensor([[0] * Np + [1] * Nl for _ in range(bs)], dtype=torch.long).flatten()

    batch = {
        "atype": atype.repeat(bs),
        "x0": x0.repeat(bs, 1),
        "abid": abid,
        "mask": mask,
        "edge_mask": edge_mask
    }

    return batch
