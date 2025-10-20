import torch
from torch.utils.data import DataLoader
from scipy.special import softmax, logsumexp
import openmm.unit as unit
from openmm.app import PDBFile
import time
import glob
import math
import shutil
import json
import os
import yaml
import pandas as pd

from config import infer_config, dict_to_namespace
from data import *
from utils import *
from utils.random_seed import setup_seed, SEED
from simulation import (
    get_default_parameters,
    get_simulation_environment_from_pdb,
    spring_constraint_energy_minimization
)

from simulation.hacks import minimize_with_scipy

### set backend == "pytorch"
os.environ["GEOMSTATS_BACKEND"] = "pytorch"

setup_seed(SEED)
torch.set_default_dtype(torch.float32)


def create_save_dir(args):
    if args.save_dir is None:
        save_dir = '.'.join(args.ckpt.split('.')[:-1]) + '_results_' + str(args.max_iter_energy_minimization)
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def to_device(data, device):
    if isinstance(data, dict):
        for key in data:
            data[key] = to_device(data[key], device)
    elif isinstance(data, list) or isinstance(data, tuple):
        res = [to_device(item, device) for item in data]
        data = type(data)(res)
    elif hasattr(data, 'to'):
        data = data.to(device)
    return data


def main(args):

    save_dir = create_save_dir(args)

    # load model
    print("Loading checkpoints...")
    model = torch.load(args.ckpt, map_location='cpu')
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    model.to(device)
    model.eval()
    print(f"Model loaded.")

    if args.mode == "single":
        pdbs = [args.name]
        test_set = [args.test_set]
    elif args.mode == "all":
        items = load_file(args.test_set)
        pdbs = [item["pdb"] for item in items]
        test_set = [item["state0_path"] for item in items]
    else:
        raise NotImplementedError(f"test mode {args.mode} not implemented.")
    
    cols = ['PDB', 'TIME']
    res = []

    for pdb_loop_index, (pdb_name, pdb_path) in enumerate(zip(pdbs, test_set)):
        if pdb_loop_index != int(args.index):
            continue
        out_dir = os.path.join(save_dir, pdb_name)
        os.makedirs(out_dir, exist_ok=True)
        topology = md.load(pdb_path).topology

        if args.use_energy_minim:
            param = get_default_parameters()
            param["force-field"] = args.force_field
            sim = get_simulation_environment_from_pdb(pdb_path, parameters=param)

        print(f"Start inference for PDB {pdb_name}.")
        start = time.time()

        total_count = 0

        energy_before = sim.context.getState(getEnergy=True).getPotentialEnergy()
        print(f"Energy before minimization: {energy_before} kJ/mol")
        minimization_steps = minimize_with_scipy(sim, maxiter=1000)
        energy_after = sim.context.getState(getEnergy=True).getPotentialEnergy()
        print(f"Energy after minimization: {energy_after} kJ/mol")
        print(f"Minimization steps: {minimization_steps}")

        total_count += minimization_steps

        state = sim.context.getState(getPositions=True)
        xyz_min = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

        # make test batch
        batch = make_batch(pdb_path, xyz_min, args.batch_size)
        batch = to_device(batch, device)

        positions = []

        with torch.no_grad():
            for _ in tqdm(range(args.inf_step)):
                x = model.sde(batch, sde_step=args.sde_step, temp=args.temperature, guidance=args.guidance)
                # save positions, Angstrom => nm
                x_numpy = x.cpu().numpy() / 10
                positions.append(x_numpy)
                # use energy minimization
                if args.use_energy_minim and args.max_iter_energy_minimization > 0:
                    device = x.device
                    x, minimization_steps = spring_constraint_energy_minimization(sim, x_numpy, args.max_iter_energy_minimization)
                    x = torch.from_numpy(x).to(device) * 10  # convert back to tensor
                    total_count += minimization_steps
                    print(f"Minimization steps in inference: {minimization_steps}, total count: {total_count}")
                # update batch
                batch["x0"] = x  # nm => Angstrom

                if total_count > args.energy_eval_budget:
                    print(f"Total minimization steps in inference: {total_count}")
                    break

        positions = np.array(positions, dtype=float)    # (T, N, 3)

        md.Trajectory(
            positions,
            topology
        ).save_xtc(os.path.join(out_dir, f'{pdb_name}_model_ode{args.sde_step}_inf{args.inf_step}_guidance{args.guidance}.xtc'))

        end = time.time()
        elapsed_time = end - start

        print(f"[*] Inference for PDB {pdb_name} finished, total elapsed time: {elapsed_time}.")

        res.append([pdb_name, elapsed_time])
    
    df = pd.DataFrame(res, columns=cols)
    mode = 'all' if args.mode == 'all' else args.name
    df.to_csv(os.path.join(save_dir, f'{mode}_ode{args.sde_step}_inf{args.inf_step}_guidance{args.guidance}.csv'), index=False)


if __name__ == "__main__":
    args = infer_config()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = dict_to_namespace(config)
    config.index = args.index
    main(config)
