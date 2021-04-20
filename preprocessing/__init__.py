import pathlib
from sklearn.model_selection import train_test_split

def get_datasets(ds_path, suffixes=None):
    ds_path = pathlib.Path(ds_path)
    assert ds_path.parent.parts[-1] == "datasets", "the path should be in 'datasets/'"

    def extract_suffix(path):
        if suffixes is None or path.suffix in suffixes:
            return str(path)
    
    return list(filter(None, map(extract_suffix, ds_path.rglob("**/*"))))

def split_datasets(ds, test_size=0.2, valid_size=0.2):
    train_ds, test_ds = train_test_split(ds, test_size=test_size)
    train_ds, valid_ds = train_test_split(ds, test_size=valid_size)
    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=str,
                        help="Dataset path. This path should be in 'datasets/'")
    parser.add_argument("--suffixes", "-s", type=str, nargs="+",
                        help="Extracting data containing specific suffixes. e.g. --suffixes .jpg .png")
    args = parser.parse_args()
    print(get_datasets(args.path, args.suffixes))