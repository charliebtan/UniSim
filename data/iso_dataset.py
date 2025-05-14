import numpy as np
import torch
import sys
import os
import shutil
import pickle
import glob
import subprocess
from ase import io
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from argparse import ArgumentParser

sys.path.append('..')
from data.mmap_dataset import create_mmap
from utils.constants import *
from utils.dft_utils import *
from utils.bio_utils import ATOM_TYPE


def calc_dft(atoms, prefix, outdir, pseudo_dir):
    input_file = os.path.join(outdir, f'{prefix}.pw.x.in')
    output_file = os.path.join(outdir, f'{prefix}.pw.x.out')
    ase_atoms_to_pw_input(input_file, atoms, prefix=prefix, outdir=outdir, pseudo_dir=pseudo_dir)
    subprocess.run(f'mpirun --mca btl ^openib -np 4 pw.x < {input_file} > {output_file}', shell=True)
    output_xml = os.path.join(outdir, f'{prefix}.xml')
    e_tot, dft_force = parse_pw_output(output_xml)
    # remove tmp files
    os.remove(input_file)
    os.remove(output_file)
    os.remove(output_xml)
    shutil.rmtree(os.path.join(outdir, f'{prefix}.save'))
    return e_tot, dft_force


def preprocess_iso(data_dir, pseudo_dir, delta=1, _type="train"):
    """
    :param data_dir: ISO17 data directory
    :param pseudo_dir: pseudo-potential directory for DFT calculation
    """
    np.random.seed(42)

    xyz_files = glob.glob(os.path.join(data_dir, '*.xyz'))

    for xyz_file in tqdm(xyz_files):
        iso_idx = os.path.split(xyz_file)[-1][:-4]
        energy_dat = os.path.join(data_dir, f'{iso_idx}.energy.dat')
        E = np.loadtxt(energy_dat)
        # min-max normalization
        E = norm_energy(E)

        atoms = io.read(xyz_file, index=':')
        atype = np.array([ATOM_TYPE.index(atom) for atom in atoms[0].get_chemical_symbols()], dtype=np.compat.long)
        xyz = np.array([atom.get_positions() for atom in atoms], dtype=float)

        dft_force_npz = os.path.join(data_dir, f'{iso_idx}_{_type}.npz')
        if os.path.exists(dft_force_npz):
            dft_data = np.load(dft_force_npz)
            indices, forces = dft_data["indices"], dft_data["forces"]
            dft_force_dict = {index: force for index, force in zip(indices, forces)}
        else:
            dft_force_dict = {}
        # prev_dm = None
        # for atom in atoms:
        #     pyscf_mol = ase_atoms_to_pyscf(atom)
        #     dft_energy, dft_force, prev_dm = dft_rks(pyscf_mol, prev_dm=prev_dm)
        #     dft_forces.append(dft_force)
        # dft_forces = np.array(dft_forces, dtype=float)
        outdir = "./tmp"
        os.makedirs(outdir, exist_ok=True)

        T = xyz.shape[0]
        unit = 1_000    # frame spacing = 1fs => 1ps

        # data split
        indices = list(range(T - delta * unit))
        train_len = int(0.8 * len(indices))
        train_idx = np.random.choice(indices[:train_len], 500, replace=False)
        valid_idx = np.random.choice(indices[train_len:], 100, replace=False)

        idx = train_idx if _type == "train" else valid_idx

        for i in tqdm(idx):
            # get protein pairs
            x0, x1 = xyz[i], xyz[i + delta * unit]
            x0_center = x0.mean(axis=0)
            x0 = x0 - x0_center
            x1 = x1 - x0_center
            pot0, pot1 = E[i], E[i + delta * unit]

            skip = False
            for u in [i, i + delta * unit]:
                if dft_force_dict.get(u) is None:
                    # Calculate forces using DFT
                    prefix = f'{iso_idx}_{u}'
                    try:
                        _, dft_force = calc_dft(atoms[u], prefix, outdir, pseudo_dir)
                        # Hartree/au => MJ/mol/nm
                        dft_force *= (Hartree_to_kJ_per_mol * 0.001) / (Bohr_to_Angstrom * 0.1)
                        dft_force_dict[u] = dft_force
                    except BaseException as e:
                        print(f"Error occurred for index {i}: {e}, skip.")
                        skip = True
            if skip:
                continue

            force0, force1 = dft_force_dict[i], dft_force_dict[i + delta * unit]
            data = {
                "atype": atype.tolist(),
                "delta": delta,
                "x0": x0.tolist(),
                "x1": x1.tolist(),
                "force0": force0.tolist(),
                "force1": force1.tolist(),
                "pot0": pot0,
                "pot1": pot1,
                "env": 8
            }

            yield f'{iso_idx}_{i}', data, [atype.shape[0]]

        selected_idx, dft_forces = [], []
        for k, v in dft_force_dict.items():
            selected_idx.append(k)
            dft_forces.append(v)
        selected_idx = np.array(selected_idx, dtype=np.compat.long)
        dft_forces = np.array(dft_forces, dtype=float)
        np.savez(dft_force_npz, indices=selected_idx, forces=dft_forces)


def parse():
    arg_parser = ArgumentParser(description='curate dataset')
    arg_parser.add_argument('--data_dir', type=str, required=True, help='ISO data directory')
    arg_parser.add_argument('--pseudo_dir', type=str, required=True, help='pseudo-potential directory')
    arg_parser.add_argument('--delta', type=int, default=1, help='time interval between training pairs, unit: ps')
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse()
    for _type in ["train", "valid"]:
        create_mmap(
            preprocess_iso(args.data_dir, args.pseudo_dir, delta=args.delta, _type=_type),
            os.path.join(args.data_dir, _type)
        )
