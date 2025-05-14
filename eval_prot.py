import torch
import numpy as np
import mdtraj as md
import pandas as pd
from argparse import ArgumentParser
import os
import pickle
import glob
from utils.tica_utils import *
from utils.backbone_utils import *
from plots import *


def load_traj(trajfile, top):
    _, ext = os.path.splitext(trajfile)
    if ext in ['.pdb']:
        traj = md.load(trajfile)
    elif ext in ['.xtc']:
        traj = md.load(trajfile, top=top)
    elif ext in ['.npz']:
        positions = np.load(trajfile)["positions"]
        traj = md.Trajectory(
            positions,
            md.load(top).topology
        )
    elif ext in ['.npy']:
        positions = np.load(trajfile)
        if positions.ndim == 4:
            positions = positions[0]
        traj = md.Trajectory(
            positions,
            md.load(top).topology
        )
    else:
        raise NotImplementedError
    return traj


def traj_analysis(traj_model_path, traj_ref_path, top=None, use_distances=True):
    traj_model = load_traj(traj_model_path, top=top)
    # traj_model = md.load(traj_model_path, top=f'{traj_model_path[:-4]}.pdb')
    traj_ref = load_traj(traj_ref_path, top=top)

    # TICA can be loaded if constructed before
    ref_dir = os.path.split(traj_ref_path)[0]
    if os.path.exists(tica_model_path := os.path.join(ref_dir, "tica_model.pic")):
        with open(tica_model_path, "rb") as f:
            tica_model = pickle.load(f)
    else:
        # lagtime: ATLAS 10, PepMD 100
        tica_model = run_tica(traj_ref, lagtime=10, dim=4)
        with open(tica_model_path, "wb") as f:
            pickle.dump(tica_model, f)

    # compute tica
    features = tica_features(traj_ref, use_distances=use_distances)
    feat_model = tica_features(traj_model, use_distances=use_distances)
    tics_ref = tica_model.transform(features)
    tics_model = tica_model.transform(feat_model)

    # compute phi and psi
    phi_ref, psi_ref = compute_phi_psi(traj_ref)
    phi_model, psi_model = compute_phi_psi(traj_model)
    ramachandran_js = compute_joint_js_distance(phi_ref, psi_ref, phi_model, psi_model)

    try:
        # compute JS distance of PwG, Rg and TIC01
        pwd_ref = compute_pairwise_distances(traj_ref)
        pwd_model = compute_pairwise_distances(traj_model)

        rg_ref = compute_radius_of_gyration(traj_ref)
        rg_model = compute_radius_of_gyration(traj_model)

        pwd_js = compute_js_distance(pwd_ref, pwd_model)
        rg_js = compute_js_distance(rg_ref, rg_model)
    except BaseException as e:
        print(f"[!] Errno: {e}")
        pwd_js = 0
        rg_js = 0
    tic_js = compute_js_distance(tics_ref[:, :2], tics_model[:, :2])
    tic2d_js = compute_joint_js_distance(tics_ref[:, 0], tics_ref[:, 1], tics_model[:, 0], tics_model[:, 1])

    print(
        f"JS distance: Ram {ramachandran_js:.4f} PwD {pwd_js:.4f}, Rg {rg_js:.4f}, TIC {tic_js:.4f}, TIC2D {tic2d_js:.4f}")

    # compute Val-CA
    val_ca, _ = compute_validity(traj_model)
    print(f"Validity CA: {val_ca:.4f}")

    # compute RMSE contact
    try:
        contact_ref, res_dist_ref = compute_residue_matrix(traj_ref)
        contact_model, res_dist_model = compute_residue_matrix(traj_model)
        n_residues = contact_ref.shape[0]
        rmse_contact = np.sqrt(2 / (n_residues * (n_residues - 1)) * np.sum((contact_ref - contact_model) ** 2))
    except BaseException as e:
        print(f"[!] Errno: {e}")
        rmse_contact = 0
    print(f"RMSE contact: {rmse_contact:.4f}")

    # compute ESS per second
    try:
        ess_model = ESS(tics_model, axis=0)
        ess_ref = ESS(tics_ref, axis=0)
        # classical MD inference time
        ref_stats = pd.read_csv(os.path.join(os.path.split(traj_ref_path)[0], "stats.txt"))
        elapsed_time_ref = ref_stats[ref_stats.columns[-1]].iloc[-1]    # unit: s
        # UniSim inference time
        model_stats_path = glob.glob(f"{os.path.dirname(os.path.dirname(traj_model_path))}/all*.csv")
        model_stats = pd.read_csv(model_stats_path[0])
        pdb_name = os.path.dirname(traj_model_path)[-4:]
        elapsed_time_model = model_stats[model_stats["PDB"] == pdb_name]["TIME"].values[0]    # unit: s
        ess_per_second_ref = ess_ref / elapsed_time_ref
        ess_per_second_model = ess_model / elapsed_time_model
    except:
        ess_per_second_model = ess_per_second_ref = -1.0
    print(f"ESS per second: {ess_per_second_model:.4f} (model) / {ess_per_second_ref:.4f} (ref)")

    return ramachandran_js, pwd_js, rg_js, tic_js, tic2d_js, val_ca, rmse_contact, ess_per_second_model, ess_per_second_ref


def parse():
    arg_parser = ArgumentParser(description='simulation')
    arg_parser.add_argument('--top', type=str, default=None, help='topology file path (.pdb)')
    arg_parser.add_argument('--ref', type=str, required=True, help='reference trajectory file path')
    arg_parser.add_argument('--model', type=str, required=True, help='model generated trajectory file path')
    arg_parser.add_argument('--use_distances', default=True, action='store_true', help='[Optional] using pairwise distances as projected features')
    return arg_parser.parse_args()


def main(args):
    traj_analysis(args.model, args.ref, top=args.top, use_distances=args.use_distances)


if __name__ == "__main__":
    main(parse())
