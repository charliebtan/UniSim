import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.special import softmax, logsumexp
from openmm.app import PDBFile
import time
import glob
import math
import shutil
import json
import os
import yaml

from config import infer_config, dict_to_namespace
from data import *
from utils import *
from utils.random_seed import setup_seed, SEED

### set backend == "pytorch"
os.environ["GEOMSTATS_BACKEND"] = "pytorch"

setup_seed(SEED)
torch.set_default_dtype(torch.float32)


def create_save_dir(args):
    if args.save_dir is None:
        save_dir = '.'.join(args.ckpt.split('.')[:-1]) + '_results'
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
        names = [args.name]
        test_set = [args.test_set]
    elif args.mode == "all":
        items = load_file(args.test_set)
        names = [item["name"] for item in items]
        test_set = [item["traj_path"] for item in items]
    else:
        raise NotImplementedError(f"test mode {args.mode} not implemented.")

    for name, traj_path in zip(names, test_set):
        out_dir = os.path.join(save_dir, name)
        os.makedirs(out_dir, exist_ok=True)

        # make test batch
        batch = make_batch(traj_path, args.batch_size, _format="npz")
        batch = to_device(batch, device)

        positions = []

        print(f"Start inference for PDB {name}.")
        start = time.time()

        with torch.no_grad():
            for _ in tqdm(range(args.inf_step)):
                x = model.sde(batch, sde_step=args.sde_step, temp=args.temperature, guidance=args.guidance)
                positions.append(x.cpu().numpy())
                # update batch
                batch["x0"] = x  # nm => Angstrom

        positions = np.array(positions, dtype=float)    # (T, N, 3)

        np.savez(os.path.join(out_dir, f'{name}_model_ode{args.sde_step}_inf{args.inf_step}_guidance{args.guidance}.npz'),
                 z=batch["atype"].detach().cpu().numpy() + 1, positions=positions)

        end = time.time()
        elapsed_time = end - start

        print(f"[*] Inference for molecule {name} finished, total elapsed time: {elapsed_time}.")


if __name__ == "__main__":
    args = infer_config()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = dict_to_namespace(config)
    main(config)
