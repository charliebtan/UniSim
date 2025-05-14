import h5py
import numpy as np
import torch
import sys
from argparse import ArgumentParser
import os

sys.path.append('..')
from data.mmap_dataset import create_mmap
from utils.constants import *
from utils.bio_utils import ATOM_TYPE


# List of keys to point to requested data
data_keys = ['wb97x_dz.energy', 'wb97x_dz.forces']  # Original ANI-1x data (https://doi.org/10.1063/1.5023802)
# data_keys = ['wb97x_tz.energy','wb97x_tz.forces'] # CHNO portion of the data set used in AIM-Net (https://doi.org/10.1126/sciadv.aav6490)
# data_keys = ['ccsd(t)_cbs.energy'] # The coupled cluster ANI-1ccx data set (https://doi.org/10.1038/s41467-019-10827-4)
# data_keys = ['wb97x_dz.dipoles'] # A subset of this data was used for training the ACA charge model (https://doi.org/10.1021/acs.jpclett.8b01939)


def iter_data_buckets(h5filename, keys=None):
    """ Iterate over buckets of data in ANI HDF5 file.
    Yields dicts with atomic numbers (shape [Na,]) coordinated (shape [Nc, Na, 3])
    and other available properties specified by `keys` list, w/o NaN values.
    """
    keys = set(keys)
    keys.discard('atomic_numbers')
    keys.discard('coordinates')
    with h5py.File(h5filename, 'r') as f:
        for grp in f.values():
            Nc = grp['coordinates'].shape[0]
            mask = np.ones(Nc, dtype=bool)
            data = dict((k, grp[k][()]) for k in keys)
            for k in keys:
                v = data[k].reshape(Nc, -1)
                mask = mask & ~np.isnan(v).any(axis=1)
            if not np.sum(mask):
                continue
            d = dict((k, data[k][mask]) for k in keys)
            d['atomic_numbers'] = grp['atomic_numbers'][()]
            d['coordinates'] = grp['coordinates'][()][mask]
            yield d


def preprocess_ani1x(data_path, _type="train"):
    data = list(iter_data_buckets(data_path, keys=data_keys))

    indices = list(range(len(data)))
    train_len = int(0.8 * len(data))
    idx = indices[:train_len] if _type == "train" else indices[train_len:]

    for i in idx:
        df = data[i]
        X = df['coordinates']           # (M, N, 3), Angstrom
        Z = df['atomic_numbers'] - 1    # (N,)
        E = df['wb97x_dz.energy']       # (M,), Hartree
        F = df['wb97x_dz.forces']       # (M, N, 3), Hartree/Angstrom

        ref_energy = sum([atom_ref_energy[ATOM_TYPE[a]] for a in Z])
        # eV => Hartree
        ref_energy /= Hartree_to_eV
        # subtract reference energy
        E -= ref_energy

        E *= Hartree_to_kJ_per_mol * 0.001          # Hartree => J/mol
        F *= (Hartree_to_kJ_per_mol * 0.001) * 10   # Hartree/A => J/mol/nm

        for m in range(X.shape[0]):
            x = X[m] - X[m].mean(axis=0)
            dp = {
                "atype": Z.tolist(),
                "x0": x.tolist(),
                "pot0": E[m].tolist(),
                "force0": F[m].tolist(),
                "env": 2
            }
            yield f'{i}-{m}', dp, [Z.shape[0]]


def parse():
    arg_parser = ArgumentParser(description='curate dataset')
    arg_parser.add_argument('--data_path', type=str, required=True, help='path to ANI-1x dataset file (.h5)')
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse()
    # Path to the ANI-1x data set
    data_path = args.data_path

    for _type in ["train", "valid"]:
        create_mmap(
            preprocess_ani1x(data_path, _type=_type),
            os.path.join(os.path.split(data_path)[0], _type)
        )
