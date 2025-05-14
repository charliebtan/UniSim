#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from types import SimpleNamespace

from utils.logger import print_log
from utils.random_seed import setup_seed, SEED

### set backend == "pytorch"
os.environ["GEOMSTATS_BACKEND"] = "pytorch"

setup_seed(SEED)

########### Import your packages below ##########
from trainer import TrainConfig
from config import train_config


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    else:
        return d


def main(args):
    ########### load your train / valid set ###########
    if len(args.gpus) > 1:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', world_size=len(args.gpus))
        args.training.batch_size = int(args.training.batch_size / len(args.gpus))
        if args.local_rank == 0:
            print_log(f'Batch size on a single GPU: {args.training.batch_size}')
    else:
        args.local_rank = -1

    # prepare datasets
    if args.data.dataset.upper() == "UNI":
        from data import collate_fn, UniDataset, MixDatasetWrapper, DynamicBatchWrapper
        train_set_paths = [value for key, value in vars(args.data.path).items() if key.endswith('_train')]
        valid_set_paths = [value for key, value in vars(args.data.path).items() if key.endswith('_valid')]
        train_set_list = [DynamicBatchWrapper(UniDataset(path), complexity=args.data.complexity,
                                              ubound_per_batch=args.data.ubound_per_batch,
                                              same_origin=args.data.same_origin) for path in train_set_paths]
        valid_set_list = [DynamicBatchWrapper(UniDataset(path), complexity=args.data.complexity,
                                              ubound_per_batch=args.data.ubound_per_batch,
                                              same_origin=args.data.same_origin) for path in valid_set_paths]
        train_set = MixDatasetWrapper(*train_set_list, reweigh=True, max_batches=args.data.max_batches)
        valid_set = MixDatasetWrapper(*valid_set_list, reweigh=True, max_batches=args.data.max_batches)
    else:
        raise NotImplementedError(f'Dataset type {args.data.dataset} not implemented.')

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=args.training.shuffle) \
        if len(args.gpus) > 1 else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_set, shuffle=False) \
        if len(args.gpus) > 1 else None
    train_loader = DataLoader(train_set, batch_size=args.training.batch_size,
                              num_workers=args.training.num_workers,
                              shuffle=(args.training.shuffle and train_sampler is None),
                              sampler=train_sampler,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=args.training.batch_size,
                              num_workers=args.training.num_workers,
                              shuffle=False,
                              sampler=valid_sampler,
                              collate_fn=collate_fn)

    ########## define your model/trainer/trainconfig #########
    config = TrainConfig(args.training.save_dir, args.training.lr, args.training.max_epoch,
                         warmup=args.training.warmup,
                         patience=args.training.patience,
                         grad_clip=args.training.grad_clip,
                         save_topk=args.training.save_topk,
                         loss_type=args.training.loss_type,
                         temperature=args.training.temperature)

    if args.model.model_type.upper() == 'EKERNEL':
        from trainer import EkernelTrainer as Trainer
        from module import EnergyKernel
        if not args.model.ckpt:
            model = EnergyKernel(args.model.hidden_dim, args.model.ffn_dim, args.model.rbf_dim, args.model.expand_embed_dim,
                                 args.model.heads, args.model.layers,
                                 cutoff_lower=args.model.cutoff_lower,
                                 cutoff_upper=args.model.cutoff_upper,
                                 cutoff_H=args.model.cutoff_H,
                                 k_neighbors=args.model.k_neighbors,
                                 s_a=args.model.s_a)
        else:
            model = torch.load(args.model.ckpt, map_location='cpu')
    elif args.model.model_type.upper() == 'BM':
        from trainer import BMTrainer as Trainer
        from module import GeomBM, EnergyKernel
        if not args.model.ckpt:
            if not args.model.encoder:
                encoder = EnergyKernel(args.model.hidden_dim, args.model.ffn_dim, args.model.rbf_dim, args.model.expand_embed_dim,
                                 args.model.heads, args.model.layers,
                                 cutoff_lower=args.model.cutoff_lower,
                                 cutoff_upper=args.model.cutoff_upper,
                                 cutoff_H=args.model.cutoff_H,
                                 k_neighbors=args.model.k_neighbors,
                                 s_a=args.model.s_a)
            else:
                encoder = torch.load(args.model.encoder, map_location='cpu')
            model = GeomBM(encoder, args.model.ffn_dim, s_eu=args.model.s_eu)
        else:
            model = torch.load(args.model.ckpt, map_location='cpu')
    elif args.model.model_type.upper() == 'FBM':
        from trainer import FBMTrainer as Trainer
        from module import GeomFBM
        if not args.model.ckpt:
            baseline = torch.load(args.model.baseline, map_location='cpu')
            model = GeomFBM(baseline, args.model.hidden_dim, args.model.ffn_dim, args.model.rbf_dim, args.model.expand_embed_dim,
                            args.model.heads, args.model.layers)
        else:
            model = torch.load(args.model.ckpt, map_location='cpu')
    else:
        raise NotImplementedError(f'model type {args.model.model_type} not implemented.')

    torch.set_default_dtype(torch.float32)

    trainer = Trainer(model, train_loader, valid_loader, config)
    trainer.train(args.gpus, args.local_rank)


if __name__ == '__main__':
    args = train_config()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = dict_to_namespace(config)
    config.gpus = args.gpus
    main(config)
