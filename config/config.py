import argparse


def train_config():
    parser = argparse.ArgumentParser()

    # config
    parser.add_argument('--config', type=str, required=True, default='./config/train.yaml')

    # device
    parser.add_argument('--gpus', type=int, nargs='+', required=True, help='gpu to use, -1 for cpu')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")

    return parser.parse_args()


def infer_config():
    parser = argparse.ArgumentParser()

    # config
    parser.add_argument('--config', type=str, required=True, default='./config/infer.yaml')
    parser.add_argument('--index', type=int, default=None)
    parser.add_argument('--max_iter_energy_minimization', type=int, default=10_000)
    parser.add_argument('--energy_eval_budget', type=int, default=10_000)

    return parser.parse_args()
