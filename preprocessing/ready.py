import pathlib
from sklearn.model_selection import StratifiedKFold
from ..utils.chart import stacked_bar

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