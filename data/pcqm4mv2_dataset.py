from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from argparse import ArgumentParser
import os

import sys
sys.path.append('..')
from data.mmap_dataset import create_mmap


def optimize_H(mol):
    mol_with_H = Chem.AddHs(mol, addCoords=True)
    conf_with_H = mol_with_H.GetConformer()
    atom_type = np.array([atom.GetAtomicNum() - 1 for atom in mol_with_H.GetAtoms()])
    coords = np.array([conf_with_H.GetAtomPosition(i) for i in range(mol_with_H.GetNumAtoms())])
    return atom_type, coords


def preprocess_pcqm4mv2(data_path, _type="train"):
    """
    :param data_path: path to pcqm4m-v2-train.sdf
    """
    suppl = Chem.SDMolSupplier(data_path)

    np.random.seed(42)
    indexes = list(range(len(suppl)))
    np.random.shuffle(indexes)

    train_len = int(0.8 * len(indexes))
    valid_indexes = indexes[:train_len] if _type == "train" else indexes[train_len:]

    for idx in valid_indexes:
        mol = suppl[idx]
        res = optimize_H(mol)
        if not res:
            continue
        atom_type, coords = res
        coords = coords - coords.mean(axis=0)
        dp = {
            "atype": atom_type.tolist(),
            "x0": coords.tolist(),
            "env": 4
        }

        yield idx, dp, [atom_type.shape[0]]


def parse():
    arg_parser = ArgumentParser(description='curate dataset')
    arg_parser.add_argument('--data_path', type=str, required=True, help='path to pcqm4m-v2-train.sdf')
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse()
    # Path to the ANI-1x data set
    data_path = args.data_path


    for _type in ["train", "valid"]:
        create_mmap(
            preprocess_pcqm4mv2(data_path, _type=_type),
            os.path.join(os.path.split(data_path)[0], _type)
            )
