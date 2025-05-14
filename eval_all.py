import yaml
import os
import pandas as pd
from config import infer_config, dict_to_namespace
from utils.io import load_file
from eval_prot import *


def main(args):
    test_pdbs = load_file(args.test_set)
    gen_dir = args.gen_dir
    postfix = f'model_ode{args.sde_step}_inf{args.inf_step}_guidance{args.guidance}'
    save_path = os.path.join(gen_dir, f'{postfix}.csv')
    cols = ['PDB', 'RAM', 'PwD', 'Rg', 'TIC', 'TIC2D', 'Val', 'CONTACT', 'ESS_M', 'ESS_R']
    res_list = []
    for pdb in test_pdbs:
        name, top, ref = pdb['pdb'], pdb['state0_path'], pdb['traj_path']
        model = os.path.join(gen_dir, name, f'{name}_{postfix}.xtc')
        res = traj_analysis(model, ref, top=top, use_distances=args.use_distances)
        res = [name] + list(res)
        res_list.append(res)
    df = pd.DataFrame(res_list, columns=cols)
    df.to_csv(save_path, index=False)
    # output to terminal
    for column in cols[1:]:
        mean = df[column].mean()
        std = df[column].std()
        print(f"{column}: {mean:.4f} / {std:.4f}")


if __name__ == "__main__":
    args = infer_config()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = dict_to_namespace(config)
    main(config)
