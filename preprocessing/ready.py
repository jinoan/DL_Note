import os
import random
import numpy as np
import pathlib
from sklearn.model_selection import StratifiedKFold
from ..utils.chart import stacked_bar

def set_random_seed(seed_num:int=None):
    if seed_num:
        os.environ["PYTHONHASHSEED"] = str(seed_num)
        random.seed(seed_num)
        np.random.seed(seed_num)

def get_dataset(ds_path:str, suffixes:str=None):
    ds_path = pathlib.Path(ds_path)
    assert ds_path.parent.parts[-1] == "datasets", "the path should be in 'datasets/'"

    def _extract_suffix(path):
        if suffixes is None or path.suffix in suffixes:
            return str(path)

    images = list(filter(None, map(_extract_suffix, ds_path.rglob("*/*"))))
    labels = list(map(lambda data: data.split("/")[-2], images))
    
    return images, labels

def fold_dataset(images, labels, folds, info=False):
    skf = StratifiedKFold(n_splits=folds, shuffle=True).split(images, labels)
    _get_images = lambda idx: images[idx]
    _get_labels = lambda idx: labels[idx]
    fold_ds = map(lambda x: (
        map(_get_images, x[0]),  # train_images
        map(_get_labels, x[0]),  # train_labels
        map(_get_images, x[1]),  # test_images
        map(_get_labels, x[1])   # test_labels
    ),skf)
    if info:
        fold_dataset_info(fold_ds)
    return fold_ds

def fold_dataset_info(fold_ds):
    kwargs = {"x": [], "y": [[], []]}
    for i, (train_images, _, test_images, _) in enumerate(fold_ds):
        kwargs["x"].append(i+1)
        train_data_size = 0
        test_data_size = 0
        for _ in train_images: train_data_size += 1
        for _ in test_images: test_data_size += 1
        kwargs["y"][0].append(train_data_size)
        kwargs["y"][1].append(test_data_size)
    for fold, train_data_size, test_data_size in zip(kwargs["x"], *kwargs["y"]):
        print(f"[fold {fold}] train_data_size: {train_data_size}, test_data_size: {test_data_size}")
    stacked_bar(**kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=str,
                        help="Dataset path. This path should be in 'datasets/'")
    parser.add_argument("--suffixes", "-s", type=str, nargs="+",
                        help="Extracting data containing specific suffixes. e.g. --suffixes .jpg .png")
    parser.add_argument("--folds", "-f", type=int)
    parser.add_argument("--random_seed", "-r", type=int, default=None)
    args = parser.parse_args()
    set_random_seed(args.random_seed)
    images, labels = get_dataset(args.path, args.suffixes)
    if args.folds:
        folds = fold_dataset(images, labels, 5)
        info_folded_dataset(folds)
        for train_images, train_labels, test_images, test_labels in folds:
            for train_image in train_images:
                print(train_image)