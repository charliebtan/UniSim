import torch
import numpy as np
import mdtraj as md
from argparse import ArgumentParser
import os
import pickle
from utils.tica_utils import *
from utils.backbone_utils import *
from plots import *


def traj_analysis(traj_model_path, traj_ref_path, mol, name, plot=False):
    # Angstrom => nm
    traj_model = np.load(traj_model_path)['positions'] / 10
    traj_ref = np.load(traj_ref_path)['R'] / 10

    z = np.load(traj_ref_path)['z']
    non_H_mask = z != 1
    traj_model = traj_model[:, non_H_mask]
    traj_ref = traj_ref[:, non_H_mask]

    # TICA can be loaded if constructed before
    ref_dir = os.path.split(traj_ref_path)[0]
    if os.path.exists(tica_model_path := os.path.join(ref_dir, f"{mol}_tica_model.pic")):
        with open(tica_model_path, "rb") as f:
            tica_model = pickle.load(f)
    else:
        tica_model = run_tica(traj_ref, lagtime=100, dim=4, pos_only=True)
        with open(tica_model_path, "wb") as f:
            pickle.dump(tica_model, f)

    # compute tica
    features = distances(traj_ref)
    feat_model = distances(traj_model)
    tics_ref = tica_model.transform(features)
    tics_model = tica_model.transform(feat_model)
    tic_js = compute_js_distance(tics_ref[:, :2], tics_model[:, :2])
    tic2d_js = compute_joint_js_distance(tics_ref[:, 0], tics_ref[:, 1], tics_model[:, 0], tics_model[:, 1])

    print(f"JS distance: TIC {tic_js:.4f}, TIC2D {tic2d_js:.4f}")

    # plot
    if plot:
        # TIC & TIC-2D
        tic_kde_path = f"outputs/{mol}/tic_kde.npz"
        if os.path.exists(tic_kde_path):
            tic_kde = np.load(tic_kde_path)
            tic_x, tic_y, tic_z = tic_kde['x'], tic_kde['y'], tic_kde['z']
            plot_tic2d_contour(tics_ref[:, 0], tics_ref[:, 1], tics_model[:, 0], tics_model[:, 1],
                               save_path=f"outputs/{mol}/tic2d_contour_{name}.pdf", xlabel='TIC 0', ylabel='TIC 1',
                               name=name, kde=(tic_x, tic_y, tic_z))
        else:
            tic_x, tic_y, tic_z = plot_tic2d_contour(tics_ref[:, 0], tics_ref[:, 1], tics_model[:, 0], tics_model[:, 1],
                                                     save_path=f"outputs/{mol}/tic2d_contour_{name}.pdf", xlabel='TIC 0',
                                                     ylabel='TIC 1', name=name)
            np.savez(tic_kde_path, x=tic_x, y=tic_y, z=tic_z)

        # free energy projected on TIC01
        tics_list = [tics_ref, tics_model]
        method_list = ["MD", name]
        plot_free_energy_all([tics[:, 0] for tics in tics_list], method_list, xlabel='TIC 0',
                             save_path=f"outputs/{mol}/fe_tic0_{name}.pdf")
        plot_free_energy_all([tics[:, 1] for tics in tics_list], method_list, xlabel='TIC 1',
                             save_path=f"outputs/{mol}/fe_tic1_{name}.pdf")

        return tic_js, tic2d_js


def parse():
    arg_parser = ArgumentParser(description='evaluate on molecules')
    arg_parser.add_argument('--ref', type=str, required=True, help='reference trajectory file path')
    arg_parser.add_argument('--model', type=str, required=True, help='model generated trajectory file path')
    arg_parser.add_argument('--mol', type=str, required=True, help='test molecule name')
    arg_parser.add_argument('--name', type=str, default='UniSim', help='model name')
    arg_parser.add_argument('--plot', action='store_true', help='if need plot (slow!)')
    return arg_parser.parse_args()


def main(args):
    traj_analysis(args.model, args.ref, mol=args.mol, name=args.name, plot=args.plot)


if __name__ == "__main__":
    main(parse())
