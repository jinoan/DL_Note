
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
import cv2
from typing import Union
from .utils.chart import stacked_bar

class TorchDataset:
    def __init__(self,
                 data_path:str=None,
                 suffixes:str=None,
                 image_size:tuple=(256, 256),
                 color_mode:Union[int, str]=1,  # 1: 'color', '0': 'grey', '-1': 'unchanged'
                 augmentations=None,
                 transforms=None,
                 **kwargs):
        super(TorchDataset, self).__init__()

        self.images = None
        self.labels = None
        self.classes = None

        if data_path is not None:
            path = Path(data_path)
            assert path.parent.parts[-1] == "datasets", "the dataset dir path should be in 'datasets/'"

            def _extract_suffix(path):
                if suffixes is None or path.suffix in suffixes:
                    return str(path)

            self.images = list(filter(None, map(_extract_suffix, path.rglob("*/*"))))
            self.classes = os.listdir(path)
            sparse_encode = {}
            for i, c in enumerate(self.classes):
                sparse_encode[c] = i
            self.labels = list(map(lambda data: sparse_encode[data.split("/")[-2]], self.images))

        if "images" in kwargs.keys(): self.images = kwargs["images"]
        if "labels" in kwargs.keys(): self.labels = kwargs["labels"]
        if "classes" in kwargs.keys(): self.classes = kwargs["classes"]

        self.image_size = image_size
        self.color_mode = {1: "color", 0: "grey", -1:"unchanged"}.get(color_mode, color_mode)
        self.augmentatinos = augmentations
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = cv2.imread(self.images[index], flags={"color": cv2.IMREAD_COLOR,
                                                      "grey": cv2.IMREAD_GRAYSCALE,
                                                      "unchanged": cv2.IMREAD_UNCHANGED}[self.color_mode])

        if self.augmentations is not None:
            image = self.augmentations(image=image)["image"]
        
        image = (image/255).astype("float")
        if image.ndim == 2:
            image = image[..., np.newaxis]
            
        if self.transforms is not None:
            label = self.transforms(self.labels[index])
        
        return 
        
        """
        필요한 작업:
        augmentation 적용
        transforms 적용
        augmentation 객체 받아오는 기능 추가
        """

    def config(self, **kwargs):
        self.image_size = kwargs.get("image_size", (256, 256))
        self.grey_scale = kwargs.get("grey_scale", False)

    def fold(self, num_folds, info=False):
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True).split(self.images, self.labels)
        _get_images = lambda idx: self.images[idx]
        _get_labels = lambda idx: self.labels[idx]
        fold_ds = list(map(lambda x: (
            TorchDataset(images=list(map(_get_images, x[0])),
                         labels=list(map(_get_labels, x[0])),
                         classes=self.classes), # train dataset
            TorchDataset(images=list(map(_get_images, x[1])),
                         labels=list(map(_get_labels, x[1])),
                         classes=self.classes) # test dataset
        ),skf))
        if info:
            self.__fold_dataset_info(fold_ds)
        return fold_ds

    def __fold_dataset_info(self, fold_ds):
        kwargs = {"x": [], "y": [[], []]}
        for i, (train_ds, test_ds) in enumerate(fold_ds):
            kwargs["x"].append(i+1)
            train_data_size = 0
            test_data_size = 0
            for _ in train_ds.images: train_data_size += 1
            for _ in test_ds.images: test_data_size += 1
            kwargs["y"][0].append(train_data_size)
            kwargs["y"][1].append(test_data_size)
        for fold, train_data_size, test_data_size in zip(kwargs["x"], *kwargs["y"]):
            print(f"[fold {fold}] train_data_size: {train_data_size}, test_data_size: {test_data_size}")
        stacked_bar(**kwargs)