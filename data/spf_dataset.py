import numpy as np
import torch
import sys
import os
import pickle
from argparse import ArgumentParser

sys.path.append('..')
from data.mmap_dataset import create_mmap
from utils.constants import *


def preprocess_spf(data_path, _type="train"):
    """
        :param data_path: .npz file
        """
    np.random.seed(42)

    data = np.load(data_path)

    atype = data['Z'] - 1
    xyz = data['R'] # Angstrom
    E = data['E']
    F = data['F']
    N = data['N']

    # eV => MJ/mole
    E = E * eV_to_kJ_per_mol * 0.001
    # eV/Angstrom => MJ/mole/nm
    F = F * eV_to_kJ_per_mol * 0.001 * 10

    T = xyz.shape[0]

    # data split
    indices = list(range(T))
    train_len = int(0.8 * T)

    idx = indices[:train_len] if _type == "train" else indices[train_len:]

    for i in idx:
        num_atoms = N[i]
        # get protein pairs
        x0 = xyz[i][:num_atoms]
        x0 = x0 - x0.mean(axis=0)
        data = {
            "atype": atype[i, :num_atoms].tolist(),
            "x0": x0.tolist(),
            "force0": F[i, :num_atoms].tolist(),
            "pot0": E[i].tolist(),
            "env": 7
        }

        yield f'{i}', data, [num_atoms]


def parse():
    arg_parser = ArgumentParser(description='curate dataset')
    arg_parser.add_argument('--data_path', type=str, required=True, help='Solvated Protein Fragments datafile (.npz)')
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse()
    save_dir = os.path.split(args.data_path)[0]
    for _type in ["train", "valid"]:
        create_mmap(
            preprocess_spf(args.data_path, _type=_type),
            os.path.join(save_dir, _type)
        )
