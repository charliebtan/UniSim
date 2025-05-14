import numpy as np
import sys
import os
import pickle
from argparse import ArgumentParser
from openmm.app import PDBFile

import mdtraj as md
import torch
from tqdm import tqdm
import json

sys.path.append('..')
from utils import load_file
from utils.bio_utils import *
from simulation import *


### preprocess openmm simulation output to curate Peptide train/valid data
def fix_energies_and_forces(split_path):
    """
    :param split_path: split summary file, jsonl format
    """
    items = load_file(split_path)
    fix_items = []

    for item in tqdm(items):
        E, F = [], []

        traj = md.load(item["traj_xtc_path"], top=item["state0_path"])
        xyz = traj.xyz

        parameters = get_default_parameters()
        parameters["force-field"] = "amber14-implicit"
        simulation = get_simulation_environment_from_pdb(item["state0_path"], parameters=parameters)

        for idx in range(xyz.shape[0]):
            E.append(get_potential(simulation, xyz[idx]))
            F.append(get_force(simulation, xyz[idx]))

        E = np.array(E, dtype=np.float32)
        F = np.array(F, dtype=np.float32)

        traj_npz_path = os.path.join(os.path.split(item["state0_path"])[0], 'ef_implicit.npz')
        np.savez(traj_npz_path, energies=E, forces=F)

        item["traj_npz_path"] = traj_npz_path
        fix_items.append(item)

    with open(f'{os.path.splitext(split_path)[0]}_fix.jsonl', 'w') as fout:
        for item in fix_items:
            item_str = json.dumps(item)
            fout.write(f'{item_str}\n')


def parse():
    arg_parser = ArgumentParser(description='curate ATLAS dataset')
    arg_parser.add_argument('--split', type=str, required=True, help='dataset split file')
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse()
    ### curate Peptide data
    fix_energies_and_forces(args.split)
