from datasets import load_dataset


def load_splits(lang="hau"):
    return load_dataset("HausaNLP/NaijaSenti-Twitter", lang, trust_remote_code=True)


def label_names(ds):
    feat = ds["train"].features["label"]
    return feat.names if hasattr(feat, "names") else sorted(set(ds["train"]["label"]))


if __name__ == "__main__":
    ds = load_splits()
    print(ds)
    print(label_names(ds))
    print(ds["train"][0])
