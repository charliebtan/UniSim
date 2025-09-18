import numpy as np
import sys
import os
import pickle
from argparse import ArgumentParser

import mdtraj as md
import torch
from tqdm import tqdm

sys.path.append('..')
from utils import load_file
from utils.bio_utils import *
from utils.tica_utils import run_tica, reweigh_by_free_energy
from data.subgraph import graph_cut, SG_THRESH
from data.mmap_dataset import create_mmap
from utils.constants import *


### preprocess openmm simulation output to curate Peptide train/valid data
def preprocess_atlas(split_path, delta=10, _type="train"):
    """
    :param split_path: split summary file, jsonl format
    :param delta: maximum time interval between training pairs, default 10(x10)ps
    """
    # save_dir = os.path.split(split_path)[0]
    # split = os.path.split(split_path)[-1].split('.')[0]     # train/valid
    items = load_file(split_path)

    np.random.seed(42)

    for item in tqdm(items):
        traj = md.load(item["traj_xtc_path"], top=item["state0_path"])
        top = traj.topology
        xyz = 10 * traj.xyz     # (T, N, 3), Angstrom
        T = xyz.shape[0]

        """
        # reweighing by free energy
        if os.path.exists(tica_model_path := os.path.join(os.path.split(item["state0_path"])[0], "tica_model.pic")):
            with open(tica_model_path, "rb") as f:
                tica_model = pickle.load(f)
        else:
            tica_model = run_tica(traj, lagtime=1, dim=2)   # lagtime=100ps
            with open(tica_model_path, "wb") as f:
                pickle.dump(tica_model, f)
        weight = reweigh_by_free_energy(traj, tica_model)
        weight /= weight.sum()
        """

        atype = get_atype(top)  # (N,)

        traj_npz = np.load(item["traj_npz_path"])
        # kJ/mole/nm => MJ/mole/nm
        forces = traj_npz["forces"] * 0.001  # (T, N, 3)
        # kJ/mole, min-max normalization
        potentials = norm_energy(traj_npz["energies"])

        # data split
        valid_length = list(range(T - delta))
        train_len = int(0.8 * len(valid_length))
        train_idx = np.random.choice(valid_length[:train_len], 500, replace=False)
        valid_idx = np.random.choice(valid_length[train_len:], 100, replace=False)

        idx = train_idx if _type == "train" else valid_idx

        for i in idx:
            # get protein pairs
            x0, x1 = xyz[i], xyz[i + delta]
            n_subgraph = int(atype.shape[0] / SG_THRESH) + 1
            if n_subgraph <= 1:
                x0_center = x0.mean(axis=0)
                x0 = x0 - x0_center
                x1 = x1 - x0_center
                force0, force1 = forces[i], forces[i + delta]
                pot0, pot1 = potentials[i], potentials[i + delta]
                data = {
                    "atype": atype.tolist(),
                    "delta": delta * 10,
                    "x0": x0.tolist(),
                    "x1": x1.tolist(),
                    "force0": force0.tolist(),
                    "force1": force1.tolist(),
                    "pot0": pot0.tolist(),
                    "pot1": pot1.tolist(),
                    "env": 3
                }
                yield f'{item["pdb"]}_{i}', data, [atype.shape[0]]
            else:
                atom_indices = list(range(atype.shape[0]))
                centers = [np.random.choice(atom_indices[i * SG_THRESH: (i + 1) * SG_THRESH]) for i in range(n_subgraph)]
                for center in centers:
                    max_indices, mask = graph_cut(x0, center=center)
                    x0_slice = x0[max_indices]
                    x1_slice = x1[max_indices]
                    # make sure in a conservative field by CoM
                    x0_center = x0_slice.mean(axis=0)
                    x0_slice = x0_slice - x0_center
                    x1_slice = x1_slice - x0_center
                    force0, force1 = forces[i][max_indices], forces[i + delta][max_indices]
                    pot0, pot1 = potentials[i], potentials[i + delta]
                    data = {
                        "atype": atype[max_indices].tolist(),
                        "mask": mask.tolist(),
                        "delta": delta * 10,
                        "x0": x0_slice.tolist(),
                        "x1": x1_slice.tolist(),
                        "force0": force0.tolist(),
                        "force1": force1.tolist(),
                        "pot0": pot0.tolist(),
                        "pot1": pot1.tolist(),
                        "env": 3
                    }
                    env_atom_num = max_indices.shape[0]
                    grad_atom_num = mask.sum()
                    adj_atom_num = grad_atom_num + int(0.5 * (env_atom_num - grad_atom_num))
                    yield f'{item["pdb"]}_{i}-{center}', data, [adj_atom_num]


def parse():
    arg_parser = ArgumentParser(description='curate ATLAS dataset')
    arg_parser.add_argument('--split', type=str, required=True, help='dataset split file')
    arg_parser.add_argument('--delta', type=int, default=10, help='time interval between training pairs, unit: (x10)ps')
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse()
    ### curate Peptide data
    for _type in ["train", "valid"]:
        create_mmap(
            preprocess_atlas(args.split, delta=args.delta, _type=_type),
            os.path.join(os.path.split(args.split)[0], _type)
        )