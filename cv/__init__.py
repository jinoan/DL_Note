
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from .utils.chart import stacked_bar
import cv2

class TorchDataset:
    def __init__(self, data_path:str=None, suffixes:str=None, **kwargs):
        super(TorchDataset, self).__init__()
        if data_path is not None:
            path = Path(data_path)
            assert path.parent.parts[-1] == "datasets", "the dataset dir path should be in 'datasets/'"

            def _extract_suffix(path):
                if suffixes is None or path.suffix in suffixes:
                    return str(path)

            self.images = list(filter(None, map(_extract_suffix, path.rglob("*/*"))))
            self.labels = list(map(lambda data: data.split("/")[-2], self.images))

        if "images" in kwargs.keys(): self.images = kwargs["images"]
        if "labels" in kwargs.keys(): self.labels = kwargs["labels"]
        self.image_shape = kwargs["image_shape"] if "image_shape" in kwargs.keys() else (256, 256)
        self.greyscale = kwargs["greyscale"] if "greyscale" in kwargs.keys() else False
        self.augmentation = None
        self.transforms = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        images = cv2.imread(self.images[index])
        """
        필요한 작업:
        augmentation 적용
        transforms 적용
        augmentation 객체 받아오는 기능 추가
        """

    def config(self, **kwargs):
        if "image_shape" in kwargs.keys(): self.image_shape = kwargs["image_shape"]
        if "greyscale" in kwargs.keys(): self.greyscale = kwargs["greyscale"]

    def fold(self, num_folds, info=False):
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True).split(self.images, self.labels)
        _get_images = lambda idx: self.images[idx]
        _get_labels = lambda idx: self.labels[idx]
        fold_ds = list(map(lambda x: (
            TorchDataset(images=list(map(_get_images, x[0])),
                         labels=list(map(_get_labels, x[0]))), # train dataset
            TorchDataset(images=list(map(_get_images, x[1])),
                         labels=list(map(_get_labels, x[1]))) # test dataset
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