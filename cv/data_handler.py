import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold as KF, StratifiedKFold as SKF
import cv2
from torch.utils.data import Dataset
from .utils.chart import stacked_bar
from copy import copy

base = Dataset

class TorchDataset(base):
    def __init__(
        self,
        images=None,
        labels=None,
        **kwargs
    ):
        super(TorchDataset, self).__init__()
        self.__dict__ = kwargs
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return {"image": self.images[index], "label": self.labels[index]}


class LoadImages:
    def __init__(self, dataset, color_mode=1):
        # color_mode 1: "color", 0: "grey", -1: "unchanged"
        self.__dict__ = dataset.__dict__.copy()
        self.dataset = dataset
        self.color_mode = {1: "color", 0: "grey", -1: "unchanged"}.get(color_mode, color_mode)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample["image"] = cv2.imread(
            sample["image"],
            flags={
                "color": cv2.IMREAD_COLOR,
                "grey": cv2.IMREAD_GRAYSCALE,
                "unchanged": cv2.IMREAD_UNCHANGED
            }[self.color_mode]
        )
        if sample["image"].ndim == 2:
            sample["image"] = sample["image"][..., np.newaxis]
        return sample


class ResizeImages:
    def __init__(self, dataset, image_size, interpolation=cv2.INTER_LINEAR):
        self.__dict__ = dataset.__dict__.copy()
        self.dataset = dataset
        self.image_size = image_size
        self.interpolation = interpolation
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample["image"] = cv2.resize(sample["image"], dsize=self.image_size, interpolation=self.interpolation)
        return sample


class OneHotLabels:
    def __init__(self, dataset):
        self.__dict__ = dataset.__dict__.copy()
        self.dataset = dataset

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample["label"] = list(map(lambda x: int(x==sample["label"]), self.classes))
        return sample


class SparseLabels:
    def __init__(self, dataset):
        self.__dict__ = dataset.__dict__.copy()
        self.dataset = dataset

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample["label"] = self.classes.index(sample["label"])
        return sample


class Augmentations:
    def __init__(self, dataset, augmentations):
        self.__dict__ = dataset.__dict__.copy()
        self.dataset = dataset
        self.augmentations = augmentations

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample["image"] = self.augmentations(image=sample["image"])["image"]
        return sample


class Transforms:
    def __init__(self, dataset, transforms):
        self.__dict__ = dataset.__dict__.copy()
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample = self.transforms(sample)
        return sample


class KFold:
    def __init__(self, dataset):
        self.__dict__ = dataset.__dict__.copy()
        self.dataset = dataset

    def split(self, n_splits=5, info=False):
        kf = KF(n_splits=n_splits, shuffle=True).split(self.images, self.labels)
        folds = []
        for train_idx, test_idx in kf:
            train_dataset = copy(self.dataset)
            train_dataset.__dict__["images"] = list(map(lambda idx: self.images[idx], train_idx))
            train_dataset.__dict__["labels"] = list(map(lambda idx: self.labels[idx], train_idx))

            test_dataset = copy(self.dataset)
            test_dataset.__dict__["images"] = list(map(lambda idx: self.images[idx], test_idx))
            test_dataset.__dict__["labels"] = list(map(lambda idx: self.labels[idx], test_idx))

            folds.append((train_dataset, test_dataset))

        if info:
            folds_info(folds)
        return folds


class StratifiedKFold:
    def __init__(self, dataset):
        self.__dict__ = dataset.__dict__.copy()
        self.dataset = dataset

    def split(self, n_splits=5, info=False):
        skf = SKF(n_splits=n_splits, shuffle=True).split(self.images, self.labels)
        folds = []
        for train_idx, test_idx in skf:
            train_dataset = copy(self.dataset)
            train_dataset.__dict__["images"] = list(map(lambda idx: self.images[idx], train_idx))
            train_dataset.__dict__["labels"] = list(map(lambda idx: self.labels[idx], train_idx))

            test_dataset = copy(self.dataset)
            test_dataset.__dict__["images"] = list(map(lambda idx: self.images[idx], test_idx))
            test_dataset.__dict__["labels"] = list(map(lambda idx: self.labels[idx], test_idx))

            folds.append((train_dataset, test_dataset))

        if info:
            folds_info(folds)
        return folds

def load_dataset(data_path, suffixes=None):
    path = Path(data_path)
    assert path.parent.parts[-1] == "datasets", "The dataset dir path should be in 'datasets/'."

    def _extract_suffix(path):
        if suffixes is None or path.suffix in suffixes:
            return str(path)

    images = list(filter(None, map(_extract_suffix, path.rglob("*/*"))))
    classes = os.listdir(path)
    labels = list(map(lambda x: x.split("/")[-2], images))
    return images, labels, classes

def folds_info(folds, **kwargs):
    if folds:
        kwargs = {"x": [], "y": [[], []]}
        for i, (train_dataset, test_dataset) in enumerate(folds):
            kwargs["x"].append(i+1)
            train_data_size = 0
            test_data_size = 0
            for _ in train_dataset.images: train_data_size += 1
            for _ in test_dataset.images: test_data_size += 1
            kwargs["y"][0].append(train_data_size)
            kwargs["y"][1].append(test_data_size)
        for fold, train_data_size, test_data_size in zip(kwargs["x"], *kwargs["y"]):
            print(f"[fold {fold}] train_data_size: {train_data_size}, test_data_size: {test_data_size}")
    stacked_bar(**kwargs)