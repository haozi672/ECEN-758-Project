import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

def read_meta(path="meta.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError("meta.csv not found")
    meta = pd.read_csv(path, sep="\t", quotechar='"')
    for c in ["filename","scene_label","identifier"]:
        if c not in meta:
            raise ValueError(f"Missing {c} in meta")
    meta["filepath"] = meta["filename"].apply(lambda f: os.path.join(".", f))
    return meta

def grouped_holdout(meta, test_size=0.2, random_state=42):
    x = np.arange(len(meta))
    y = meta["scene_label"].to_numpy()
    groups = meta["identifier"].to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tr, te = next(gss.split(x, y, groups))
    return tr, te
