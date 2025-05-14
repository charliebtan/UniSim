import numpy as np
import torch
import sys
import os
import pickle
from argparse import ArgumentParser

sys.path.append('..')
from data.mmap_dataset import create_mmap
from utils.constants import *


def preprocess_md17(data_dir, delta=100, _type="train"):
    """
    :param data_dir: MD17 data directory, containing .npz files
    :param delta: time interval between training pairs, default: 100ps
    """
    np.random.seed(42)

    for npzfile in os.listdir(data_dir):
        if not npzfile.endswith('.npz'):
            continue
        data = np.load(os.path.join(data_dir, npzfile))
        atype = data['z'] - 1
        xyz = data['R']             # Angstrom
        E = data['E'].squeeze()     # kcal/mole
        F = data['F']               # kcal/mole/Angstrom
        # min-max normalization
        E = norm_energy(E)
        # kcal/mole/Angstrom => MJ/mole/nm
        F *= kcal_to_kJ * 0.001 * 10

        T = xyz.shape[0]
        unit = 2_000    # frame spacing=0.5fs => 1ps

        # data split
        if T < delta * unit:
            continue
        indices = list(range(T - delta * unit))
        train_len = int(0.8 * len(indices))
        train_idx = np.random.choice(indices[:train_len], 5000, replace=False)
        valid_idx = np.random.choice(indices[train_len:], 500, replace=False)
        idx = train_idx if _type == "train" else valid_idx

        for i in idx:
            # get protein pairs
            x0, x1 = xyz[i], xyz[i + delta * unit]
            x0_center = x0.mean(axis=0)
            x0 = x0 - x0_center
            x1 = x1 - x0_center
            force0, force1 = F[i], F[i + delta * unit]
            pot0, pot1 = E[i], E[i + delta * unit]
            data = {
                "atype": atype.tolist(),
                "delta": delta,
                "x0": x0.tolist(),
                "x1": x1.tolist(),
                "force0": force0.tolist(),
                "force1": force1.tolist(),
                "pot0": pot0,
                "pot1": pot1,
                "env": 1
            }

            yield f'{npzfile[:-4]}_{i}', data, [atype.shape[0]]


def parse():
    arg_parser = ArgumentParser(description='curate dataset')
    arg_parser.add_argument('--data_dir', type=str, required=True, help='MD17 data directory')
    arg_parser.add_argument('--delta', type=int, default=100, help='time interval between training pairs, unit: ps')
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse()
    for _type in ["train", "valid"]:
        create_mmap(
            preprocess_md17(args.data_dir, delta=args.delta, _type=_type),
            os.path.join(args.data_dir, _type)
        )
