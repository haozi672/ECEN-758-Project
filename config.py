import os
from itertools import product

SR = 22050
N_MFCC = 20
FEATURE_DEFAULTS = {
    "mfcc": {
        "n_mfcc": N_MFCC,
    },
    "logmel": {
        "n_mels": 128,
        "hop": 1024,
        "fmin": 50,
        "fmax": None,
        "use_zscore": True,
        "percentiles": (10, 90),
    },
}
FORCE_REBUILD = False
CACHE_DIR = "./.cache"
os.makedirs(CACHE_DIR, exist_ok=True)
FEATURES_CACHE = os.path.join(CACHE_DIR, f"features_mfcc{N_MFCC}_sr{SR}.joblib")
MODELS_DIR = "./models"
os.makedirs(MODELS_DIR, exist_ok=True)
TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_DEFAULTS = {"folds": 5,"grouped": True,}
AUG_DEFAULTS = {"n_aug": 1,}
SVM_GRID = {
    "svm__C":     [0.3, 1, 3, 10, 30, 100],
    "svm__gamma": ["scale", 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
}
RF_GRID = {
    "rf__n_estimators": [300, 600, 900, 1200],
    "rf__max_depth":    [None, 30, 60],
    "rf__max_features": ["sqrt", 0.3, 0.5],
}
def expand_grid(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    vals = [grid[k] if isinstance(grid[k], (list, tuple)) else [grid[k]] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*vals)]
SVM_PARAM_SPACE_DEFAULT = expand_grid(SVM_GRID)
RF_PARAM_SPACE_DEFAULT  = expand_grid(RF_GRID)
