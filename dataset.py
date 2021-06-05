from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

import os.path
from pathlib import Path

import logging

logging.basicConfig(level=logging.INFO)


def build_augmentations():
    from torchvision import transforms as transforms

    return {
        "train": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "eval": transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }


def get_dataset(path, transform=None):
    if not isinstance(path, Path):
        path = Path(path)

    if not path.exists():
        raise RuntimeError(f"failed to read dataset from {path.as_posix()}")

    if not path.is_dir():
        raise RuntimeError(f"{path.as_posix()} is not a folder")

    skiplist = {
        "data256x256_shorthair/00071.jpg"
    }

    def skip_file(path):
        if not os.path.exists(path):
            return False
        if not os.path.isfile(path):
            return False

        return os.path.join(*list(path.split("/"))[-2:]) not in skiplist

    dataset = ImageFolder(
        root=path.as_posix(), transform=transform, is_valid_file=skip_file
    )

    samples_num = len(dataset)

    train_size = int(0.95 * samples_num)

    splits = random_split(dataset, [train_size, samples_num - train_size])

    train_dataset = DataLoader(splits[0], batch_size=128, shuffle=True, num_workers=4)
    val_dataset = DataLoader(splits[1], batch_size=128, shuffle=True, num_workers=4)

    logging.info(f"train split: {len(splits[0])}, eval split: {len(splits[1])}")

    # takes quite a lot of time to compute, but roughly balanced
    #train_zero_class_count = sum([b[1] == 0 for b in splits[0]])
    #val_zero_class_count = sum([b[1] == 0 for b in splits[1]])

    # logging.info(f"train zero class: {train_zero_class_count}, eval zero class: {val_zero_class_count}")

    return {
        "train": train_dataset,
        "eval": val_dataset
    }


