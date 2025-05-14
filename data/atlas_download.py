import pandas as pd
import numpy
import requests
import os
import json
import zipfile
from tqdm import tqdm
from collections import Counter
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

from cluster import seq_cluster

import sys
sys.path.append('..')
from utils.bio_utils import get_seq

arg_parser = ArgumentParser(description='download full pdb data')
arg_parser.add_argument('--save_dir', type=str, required=True, help='Saving directory for raw PDB files and output json file')
arg_parser.add_argument('--test_size', type=int, default=16, help='test set size')
arg_parser.add_argument('--n_cpu', type=int, default=8, help='Number of cpu to use')
args = arg_parser.parse_args()

raw_dir = os.path.join(args.save_dir, 'raw')


def download_zip(url):
    os.makedirs(raw_dir, exist_ok=True)
    try:
        local_filename = os.path.join(raw_dir, url.split("/")[-1])
        response = requests.get(url, stream=True, timeout=10)
        print(f"[+] Begin downloading file {local_filename}.")

        if response.status_code == 200:
            with open(local_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"[*] {local_filename} downloaded.")
        else:
            print(f"Download fails: {url}, status code: {response.status_code}")
    except requests.RequestException as e:
        print(f"Download fails: {url}, errno: {e}")


def download(max_workers=8):
    df = pd.read_csv("ATLAS/ATLAS  Search.csv")
    pdbs = df["PDB"]
    pdbs = [f"{pdb[:4]}_{pdb[4]}" for pdb in pdbs]
    urls = [f"https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/{pdb}/{pdb}_analysis.zip" for pdb in pdbs]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(download_zip, urls)


def post_process(test_size):
    train_split, test_split = [os.path.join(args.save_dir, f'{i}.jsonl') for i in ['train', 'test']]
    items = []
    for file in os.listdir(raw_dir):
        if file.endswith('.zip'):
            print(f"[+] unzip {file}")
            pdb, chain = file[:4], file[5]
            unzip_dir = os.path.splitext(zip_path := os.path.join(raw_dir, file))[0]
            os.makedirs(unzip_dir, exist_ok=True)
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(unzip_dir)
            except:
                continue
            pdb_path = os.path.join(unzip_dir, f'{pdb}_{chain}.pdb')
            seq = get_seq(pdb_path)
            item = {
                "pdb": pdb,
                "chain": chain,
                "seq": seq,
                "state0_path": pdb_path,
                "traj_xtc_path": os.path.join(unzip_dir, f'{pdb}_{chain}_R1.xtc')
            }
            items.append(item)

    # cluster based on sequence
    clu_items = seq_cluster(items, min_seq_id=0.3)
    print(f'data length: {len(clu_items)}')
    # train split
    with open(train_split, 'w') as fout:
        for item in clu_items[:-test_size]:
            item.pop('seq', None)
            item_str = json.dumps(item)
            fout.write(f'{item_str}\n')
    # test split
    with open(test_split, 'w') as fout:
        for item in clu_items[-test_size:]:
            item.pop('seq', None)
            item_str = json.dumps(item)
            fout.write(f'{item_str}\n')


def main():
    download(max_workers=args.n_cpu)
    post_process(args.test_size)


if __name__ == "__main__":
    main()
