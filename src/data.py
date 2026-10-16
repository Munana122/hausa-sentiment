from datasets import load_dataset, DatasetDict


def load_splits(lang="hau"):
    base = "hf://datasets/HausaNLP/NaijaSenti-Twitter@refs/convert/parquet"
    return DatasetDict({
        "train":      load_dataset("parquet", data_files=f"{base}/{lang}/train/0000.parquet", split="train"),
        "validation": load_dataset("parquet", data_files=f"{base}/{lang}/validation/0000.parquet", split="train"),
        "test":       load_dataset("parquet", data_files=f"{base}/{lang}/test/0000.parquet", split="train"),
    })


def label_names(ds):
    feat = ds["train"].features["label"]
    return feat.names if hasattr(feat, "names") else sorted(set(ds["train"]["label"]))


if __name__ == "__main__":
    ds = load_splits()
    print(ds)
    print(label_names(ds))
    print(ds["train"][0])
