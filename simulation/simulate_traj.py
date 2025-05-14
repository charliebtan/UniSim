from argparse import ArgumentParser
import openmm as mm
from tqdm import tqdm
import os
import json
import numpy as np
import yaml

import sys

sys.path.append('..')
from simulation.npz_reporter import NPZReporter, RegularSpacing
from simulation.md_utils import *


def load_file(fpath):
    with open(fpath, 'r') as fin:
        lines = fin.read().strip().split('\n')
    items = [json.loads(s) for s in lines]
    return items


def simulate_trajectory(pdb_path, save_path, parameters):
    print(f"Simulation parameters: {parameters}")

    summary_path = os.path.join(save_path, 'stats.txt')
    pdb_name = os.path.split(pdb_path)[1].split('.')[0]

    model = get_openmm_model(pdb_path)
    model.addHydrogens()
    model.deleteWater()

    protein_atoms = len(model.positions)
    print("Pre-processed protein has %d atoms." % protein_atoms)
    # Write state0 file
    mm.app.pdbfile.PDBFile.writeFile(model.topology, model.positions, open(os.path.join(save_path, 'state0.pdb'), "w"))

    simulation = get_simulation_environment_from_model(model, parameters)

    simulation.context.setPositions(model.positions)

    tolerance = float(parameters["min-tol"])
    print("Performing ENERGY MINIMIZATION to tolerance %2.2f kJ/mol" % tolerance)
    simulation.minimizeEnergy(tolerance=tolerance)
    print("Completed ENERGY MINIMIZATION")

    temperature = parameters["temperature"]
    print("Initializing VELOCITIES to %s" % temperature)
    simulation.context.setVelocitiesToTemperature(temperature)

    # frame spacing=1ps
    # simfile = os.path.join(save_path, f'{pdb_name}-sim.pdb')
    # simulation.reporters.append(PDBReporter(simfile, spacing))
    spacing = parameters["spacing"]
    # save NPZ file (energies, positions, velocities, forces)
    trajnpzfile = os.path.join(save_path, f'{pdb_name}-traj-arrays.npz')
    simulation.reporters.append(
        NPZReporter(trajnpzfile, RegularSpacing(spacing), atom_indices=range(protein_atoms))
    )
    simulation.reporters.append(mm.app.StateDataReporter(summary_path, spacing, step=True, elapsedTime=True,
                                                         potentialEnergy=True))
    with open(os.path.join(save_path, "simulation_env.yaml"), 'w') as yaml_file:
        yaml.dump(parameters, yaml_file, default_flow_style=False)

    sampling = parameters["sampling"]
    print(f"Begin SAMPLING for {sampling} steps.")
    simulation.step(sampling)
    print("Completed SAMPLING")

    del simulation


def parse():
    parser = ArgumentParser(description='simulation')
    parser.add_argument('--summary', type=str, required=True, help='Path to summary file')
    parser.add_argument('--force-field', type=str, default="amber14-implicit",
                        choices=["amber99-implicit", "amber14-implicit", "amber14-explicit", "amber14-only"],
                        help='(preset) Force field, "amber99-implicit", "amber14-implicit", '
                             'or "amber14-explicit". [default: amber14-implicit]')
    parser.add_argument('--integrator', type=str, default="LangevinMiddleIntegrator",
                        choices=["LangevinMiddleIntegrator", "LangevinIntegrator"])
    parser.add_argument('--waterbox-pad', type=float, default=1.0, help='Waterbox padding width in nm [default: 1.0]')
    parser.add_argument('--temperature', type=int, default=300, help='simulation temperature [default: 300K]')
    parser.add_argument('--timestep', type=float, default=1.0,
                        help='Integration time step in femtoseconds [default: 1.0]')
    parser.add_argument('--friction', type=float, default=0.5, help='Langevin friction in 1.0/ps [default: 0.5]')
    parser.add_argument('--sampling', type=int, default=100_000_000,
                        help='Number of total integration steps [default: 100_000_000].')
    parser.add_argument('--spacing', type=int, default=1_000, help='frame spacing in femtoseconds [default: 1000]')
    parser.add_argument('--min-tol', type=float, default=2.0,
                        help='Energy minimization tolerance in kJ/mol [default: 2.0].')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='whether to use CUDA to accelerate simulation, -1 for cpu and {>0} for GPU index')
    return parser.parse_args()


def main(args):
    param_keys = ["force-field", "integrator", "waterbox-pad", "temperature", "timestep", "friction",
                  "sampling", "spacing", "min-tol", "gpu"]
    parameters = {key: getattr(args, key.replace('-', '_')) for key in param_keys}
    save_dir = os.path.join(os.path.split(args.summary)[0], 'sim_new')
    os.makedirs(save_dir, exist_ok=True)
    items = load_file(args.summary)
    for item in tqdm(items):
        pdb = item['pdb']
        print(f"[+] Start MD simulations on pdb: {pdb}.")
        item_dir = os.path.join(save_dir, pdb)
        if os.path.exists(item_dir):
            continue
        os.makedirs(item_dir, exist_ok=True)
        simulate_trajectory(item['pdb_path'], item_dir, parameters=parameters)


if __name__ == "__main__":
    main(parse())
