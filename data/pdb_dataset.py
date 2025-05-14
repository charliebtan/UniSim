import numpy as np
import sys
import os
import gzip
import shutil
from argparse import ArgumentParser

import mdtraj as md
from tqdm import tqdm

sys.path.append('..')
from utils.bio_utils import *
from data.subgraph import graph_cut, SG_THRESH
from data.mmap_dataset import create_mmap


### preprocess openmm simulation output to curate Peptide train/valid data
def preprocess_pdb(raw_dir, split_path, tmp_dir=None, _type="train"):
    """
    :param split_path: split summary file, txt format
    """
    with open(split_path, "r") as fin:
        items = [line.strip().split('\t') for line in fin.readlines()]

    for item in tqdm(items):
        pdb_file_name = item[0]
        pdb, chain_name = pdb_file_name[5:9], pdb_file_name[0]
        pdb_path = os.path.join(raw_dir, pdb[1:3], pdb_file_name[2:])

        # uncompress the file to the tmp file
        tmp_file = os.path.join(tmp_dir, f'{pdb}.pdb')
        with gzip.open(pdb_path, 'rb') as fin:
            with open(tmp_file, 'wb') as fout:
                shutil.copyfileobj(fin, fout)
        try:
            traj = md.load(tmp_file)
            chain_id = None
            for chain in traj.topology.chains:
                if chain.chain_id == chain_name:
                    chain_id = chain.index
                    break
            chain_index = traj.topology.select(f'chainid {chain_id}')
            top = traj.topology.subset(chain_index)
            xyz = 10 * traj.xyz[0][chain_index]     # (N, 3), Angstrom
            atype = get_atype(top)  # (N,)
            n_subgraph = int(xyz.shape[0] / SG_THRESH) + 1
            if n_subgraph <= 1:
                xyz = xyz - xyz.mean(axis=0)
                data = {
                    "atype": atype.tolist(),
                    "x0": xyz.tolist(),
                    "env": 5
                }
                yield f'{pdb}_{chain_name}', data, [atype.shape[0]]
            else:
                atom_indices = list(range(xyz.shape[0]))
                centers = [np.random.choice(atom_indices[i*SG_THRESH: (i+1)*SG_THRESH]) for i in range(n_subgraph)]
                for center in centers:
                    max_indices, mask = graph_cut(xyz, center=center)
                    xyz_slice = xyz[max_indices]
                    xyz_slice = xyz_slice - xyz_slice.mean(axis=0)
                    data = {
                        "atype": atype[max_indices].tolist(),
                        "mask": mask.tolist(),
                        "x0": xyz_slice.tolist(),
                        "env": 5
                    }
                    env_atom_num = max_indices.shape[0]
                    grad_atom_num = mask.sum()
                    adj_atom_num = grad_atom_num + int(0.5 * (env_atom_num - grad_atom_num))
                    yield f'{pdb}_{chain_name}_{center}', data, [adj_atom_num]
            os.remove(tmp_file)
        except BaseException as e:
            os.remove(tmp_file)
            print(f'[!] Errno: {e}')
            continue


def parse():
    arg_parser = ArgumentParser(description='curate ATLAS dataset')
    arg_parser.add_argument('--raw_dir', type=str, required=True, help='raw data directory of PDB')
    arg_parser.add_argument('--tmp_dir', type=str, default=None, help='directory for saving tmp files')
    arg_parser.add_argument('--split_dir', type=str, required=True, help='directory of dataset splits')
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse()
    ### curate Peptide data
    splits = ["train", "valid", "test"]
    for split in splits:
        create_mmap(
            preprocess_pdb(args.raw_dir, os.path.join(args.split_dir, f'{split}.txt'), tmp_dir=args.tmp_dir, _type=split),
            os.path.join(os.path.split(args.split_dir)[0], split)
        )
