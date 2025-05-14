import glob
import json
import os
import pickle
import random
import sys
from argparse import ArgumentParser

import mdtraj as md
import torch
from tqdm import tqdm

sys.path.append('..')
from utils import load_file
from utils.bio_utils import *
from utils.constants import *
from utils.geometry import kabsch_numpy
from utils.tica_utils import run_tica, reweigh_by_free_energy
from data.mmap_dataset import create_mmap


### split pep data
def split_pep(sim_dir):
    save_dir = os.path.split(sim_dir)[0]
    splits = ["train", "test"]
    all_items = []

    for _dir in os.listdir(sim_dir):
        _dir = os.path.join(sim_dir, _dir)
        traj_npz_path = glob.glob(_dir + '/*.npz')[0]
        name = os.path.split(traj_npz_path)[-1]
        pdb, chain = name[:4], name[5]
        state0_path = os.path.join(_dir, "state0.pdb")
        all_items.append({
            "pdb": pdb,
            "traj_npz_path": traj_npz_path,
            "state0_path": state0_path
        })

    np.random.seed(42)
    np.random.shuffle(all_items)

    # train:test=4:1
    train_len = int(0.8 * len(all_items))

    split_items = [
        all_items[:train_len],
        all_items[train_len:]
    ]

    split_paths = []

    for split, items in zip(splits, split_items):
        split_path = os.path.join(save_dir, f"{split}.jsonl")
        split_paths.append(split_path)
        with open(split_path, 'w') as fout:
            for item in items:
                item_str = json.dumps(item)
                fout.write(f'{item_str}\n')

    return split_paths


### preprocess openmm simulation output to curate Peptide train/valid data
def preprocess_pep(split_path, delta=100, _type="train"):
    """
    :param split_path: split summary file, jsonl format
    :param delta: minimum time interval between training pairs, default 100ps
    """
    # save_dir = os.path.split(split_path)[0]
    # split = os.path.split(split_path)[-1].split('.')[0]     # train/valid
    items = load_file(split_path)

    np.random.seed(42)

    for item in tqdm(items):
        state0 = md.load(item["state0_path"])
        if not os.path.exists(item["state0_path"]):
            continue
        top = state0.topology
        traj_npz = np.load(item["traj_npz_path"])

        atype = get_atype(top)  # (N,)
        # Angstrom
        xyz = 10 * traj_npz["positions"]           # (T, N, 3)
        # kJ/mole/nm => MJ/mole/nm
        forces = traj_npz["forces"] * 0.001        # (T, N, 3)
        # min-max normalization
        potentials = norm_energy(traj_npz["energies"][:, 0])
        T = xyz.shape[0]

        # data split
        valid_length = list(range(T - delta))
        train_len = int(0.8 * len(valid_length))
        train_idx = np.random.choice(valid_length[:train_len], 5000, replace=False)
        valid_idx = np.random.choice(valid_length[train_len:], 500, replace=False)

        idx = train_idx if _type == "train" else valid_idx

        for i in idx:
            # get protein pairs
            x0, x1 = xyz[i], xyz[i + delta]
            x0_center = x0.mean(axis=0)
            x0 = x0 - x0_center
            x1 = x1 - x0_center
            # kabsch alignment
            # x1, R, _ = kabsch_numpy(x1, x0)
            force0, force1 = forces[i], forces[i + delta]
            pot0, pot1 = potentials[i], potentials[i + delta]
            data = {
                "atype": atype.tolist(),
                "delta": delta,
                "x0": x0.tolist(),
                "x1": x1.tolist(),
                "force0": force0.tolist(),
                "force1": force1.tolist(),
                "pot0": pot0,
                "pot1": pot1,
                "env": 0
            }

            yield f'{item["pdb"]}_{i}', data, [atype.shape[0]]


def save_xtc(split_path):
    items = load_file(split_path)
    for item in tqdm(items):
        pdb = item["pdb"]
        save_dir = os.path.split(item["state0_path"])[0]
        state0 = md.load(item["state0_path"])
        top = state0.topology
        traj_npz = np.load(item["traj_npz_path"])
        xyz = traj_npz["positions"]  # (T, N, 3), nm
        md.Trajectory(
            xyz,
            top
        ).save_xtc(os.path.join(save_dir, f"{pdb}-sim.xtc"))


def parse():
    arg_parser = ArgumentParser(description='curate dataset')
    arg_parser.add_argument('--sim_dir', type=str, required=True, help='openmm simulation directory')
    arg_parser.add_argument('--delta', type=int, default=10, help='time interval between training pairs, unit: ps')
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse()
    ### curate Peptide data
    split_paths = split_pep(args.sim_dir)
    for split_path in split_paths:
        for _type in ["train", "valid"]:
            create_mmap(
                preprocess_pep(split_path, delta=args.delta, _type=_type),
                os.path.join(os.path.split(split_path)[0], f'{_type}_dt{args.delta}_raw')
            )
        save_xtc(split_path)
