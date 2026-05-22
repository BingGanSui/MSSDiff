import warnings

import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from .meteo import MeteoLoader
from .augmented import AugmentedDataset


def get_train_dataloader(options):
    if options.train.dataset in ['Asia']:
        dataset = MeteoLoader(options.train.dataset,mode='train')
    else:
        raise ValueError("unknown dataset")
    dataset = AugmentedDataset(dataset, aggressive=False)

    if options.train.distributed:
        world_size = dist.get_world_size()
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
        if options.train.batch_size % world_size != 0:
            warnings.warn(
                "batch size is not divisible by world size, batch size will be inaccurate"
            )
        return DataLoader(
            dataset,
            batch_size=options.train.batch_size // world_size,
            sampler=sampler,
            drop_last=True,
            num_workers=8,
            pin_memory=True,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=options.train.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )


def get_val_dataloader(options):
    if options.train.dataset in ['Asia']:
        dataset = MeteoLoader(options.train.dataset,mode='val')
    else:
        raise ValueError("unknown dataset")
    dataset = AugmentedDataset(dataset, random_flip=False, random_rotate=False)

    if options.train.distributed:
        world_size = dist.get_world_size()
        if len(dataset) % world_size != 0:
            warnings.warn(
                "validation dataset size is not divisible by world size, validation results will be inaccurate"
            )
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
        if options.train.batch_size % world_size != 0:
            warnings.warn(
                "batch size is not divisible by world size, batch size will be inaccurate"
            )
        return DataLoader(
            dataset,
            batch_size=options.train.batch_size // world_size,
            sampler=sampler,
            num_workers=8,
            pin_memory=True,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=options.train.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )


def get_test_dataset(options):
    if options.train.dataset in ['Asia']:
        dataset = MeteoLoader(options.train.dataset,mode='test')
    else:
        raise ValueError("unknown dataset")
    return dataset
