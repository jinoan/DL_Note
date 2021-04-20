import os
import random
import numpy as np
import pathlib
from sklearn.model_selection import StratifiedKFold

def set_random_seed(seed_num:int=None):
    if seed_num:
        os.environ["PYTHONHASHSEED"] = str(seed_num)
        random.seed(seed_num)
        np.random.seed(seed_num)

def get_dataset(ds_path:str, suffixes:str=None):
    ds_path = pathlib.Path(ds_path)
    assert ds_path.parent.parts[-1] == "datasets", "the path should be in 'datasets/'"

    def extract_suffix(path):
        if suffixes is None or path.suffix in suffixes:
            return str(path)

    images = list(filter(None, map(extract_suffix, ds_path.rglob("*/*"))))
    labels = list(map(lambda data: data.split("/")[-2], images))
    
    return images, labels

def fold_dataset(images, labels, folds):
    return StratifiedKFold(n_splits=folds, shuffle=True).split(images, labels)


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
        for train_idx, test_idx in folds:
            print(train_idx)
            print(test_idx)


