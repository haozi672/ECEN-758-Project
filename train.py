import argparse, json, os
import numpy as np
import librosa
from joblib import dump
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from config import SR, N_MFCC, FEATURES_CACHE, TEST_SIZE, RANDOM_STATE, MODELS_DIR, FEATURE_DEFAULTS
from data import read_meta, grouped_holdout
from features import build_feature_matrix, extract_from_wave,build_logmel_stats_matrix, logmel_stats_from_wave
from models import get_svm, get_rf
from augment import make_aug_variants

def _maybe_load_json(path: str | None) -> dict:
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"params file not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            obj = json.load(f)
            return obj.get("params", obj)
        except Exception as e:
            print(f"failed parsing {path}: {e}")
            return {}

def _apply_manual_overrides(base: dict, cli: dict) -> dict:
    out = dict(base) if base else {}
    for k, v in cli.items():
        if v is not None:
            out[k] = v
    return out

def _parse_max_features(s: str | None):
    if s is None:
        return None
    s = s.strip()
    if s.lower() == "sqrt":
        return "sqrt"

def _auto_params_path(model: str, feature: str, n_mels: int, hop: int, fmin: int, fmax, aug: int) -> str:
    if feature == "mfcc":
        tag = "mfcc"
    else:
        tag = f"logmel_m{n_mels}_h{hop}_f{fmin}-{0 if fmax is None else fmax}"
    return f".cache/best_{model}_{tag}_aug{aug}.json"

# augmentation
def build_augmented_train(meta, tr_idx, base_X, feature_kind, n_mels, hop, fmin, fmax, n_aug):
    y_tr_labels = meta["scene_label"].to_numpy()[tr_idx]
    file_tr = meta["filepath"].to_numpy()[tr_idx]

    if n_aug <= 0:
        return base_X, y_tr_labels

    feats, labs = [], []
    for fp, lab in zip(file_tr, y_tr_labels):
        y, _ = librosa.load(fp, sr=SR, mono=True)
        variants = make_aug_variants(y, n_aug=n_aug, sr=SR)  #includes original
        for z in variants:
            if feature_kind == "logmel":
                f = logmel_stats_from_wave(z, sr=SR, n_mels=n_mels, hop=hop, fmin=fmin, fmax=fmax, use_zscore=True)
            else:
                f = extract_from_wave(z, sr=SR, n_mfcc=N_MFCC)
            feats.append(f[None, :]); labs.append(lab)

    X_train = np.vstack(feats).astype(np.float32)
    y_train = np.array(labs)
    return X_train, y_train

def report(name, clf, Xtr, ytr, Xte, yte, classes):
    clf.fit(Xtr, ytr)
    yhat_tr = clf.predict(Xtr); yhat_te = clf.predict(Xte)
    print(f"\n{name}")
    print(f"Train acc: {accuracy_score(ytr, yhat_tr):.4f} | Test acc: {accuracy_score(yte, yhat_te):.4f}")

def main():
    ap = argparse.ArgumentParser(description="Train SVM & RF (MFCC or Log-mel pooled stats) with optional aug + auto/bespoke params.")
    ap.add_argument("--meta", type=str, default="meta.csv")
    ap.add_argument("--feature", choices=["mfcc","logmel"], default="mfcc")
    ap.add_argument("--n_mels", type=int, default=FEATURE_DEFAULTS["logmel"]["n_mels"])
    ap.add_argument("--hop", type=int, default=FEATURE_DEFAULTS["logmel"]["hop"])
    ap.add_argument("--fmin", type=int, default=FEATURE_DEFAULTS["logmel"]["fmin"])
    ap.add_argument("--fmax", type=int, default=0, help="0 => sr/2")
    ap.add_argument("--aug", type=int, default=0, help="# augs per clip for TRAIN (0=off)")
    ap.add_argument("--models", nargs="+", default=["svm", "rf"], choices=["svm", "rf"])
    ap.add_argument("--svm_params_json", type=str, default=None)
    ap.add_argument("--rf_params_json",  type=str, default=None)
    ap.add_argument("--svm_C", type=float, default=None)
    ap.add_argument("--svm_gamma", type=str, default=None)
    ap.add_argument("--rf_n_estimators", type=int, default=None)
    ap.add_argument("--rf_max_depth", type=int, default=None)
    ap.add_argument("--rf_max_features", type=str, default=None)
    args = ap.parse_args()
    fmax = None if args.fmax == 0 else args.fmax
    meta = read_meta(args.meta)
    filepaths = meta["filepath"].to_list()

    if args.feature == "logmel":
        X_all = build_logmel_stats_matrix(filepaths, sr=SR, n_mels=args.n_mels, hop=args.hop,
                                          fmin=args.fmin, fmax=fmax, use_zscore=True)
    else:
        X_all = build_feature_matrix(filepaths, cache_path=FEATURES_CACHE, sr=SR, n_mfcc=N_MFCC)

    tr_idx, te_idx = grouped_holdout(meta, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_tr_base, X_te = X_all[tr_idx], X_all[te_idx]
    y_tr_labels = meta["scene_label"].to_numpy()[tr_idx]
    y_te_labels = meta["scene_label"].to_numpy()[te_idx]
    #aug
    X_tr, y_tr_labels = build_augmented_train(
        meta, tr_idx, X_tr_base, feature_kind=args.feature,
        n_mels=args.n_mels, hop=args.hop, fmin=args.fmin, fmax=fmax, n_aug=args.aug
    )
    le = LabelEncoder(); le.fit(meta["scene_label"])
    y_tr = le.transform(y_tr_labels); y_te = le.transform(y_te_labels)

    # model
    svm = get_svm()
    rf  = get_rf(RANDOM_STATE)

    #param
    manual_svm = {
        "svm__C": args.svm_C,
        "svm__gamma": (float(args.svm_gamma) if (args.svm_gamma is not None and args.svm_gamma not in {"scale","auto"})
                       else args.svm_gamma),
    }
    manual_rf = {
        "n_estimators": args.rf_n_estimators,
        "max_depth": args.rf_max_depth,
        "max_features": _parse_max_features(args.rf_max_features),
    }
    auto_svm_json = _auto_params_path("svm", args.feature, args.n_mels, args.hop, args.fmin, fmax, args.aug)
    auto_rf_json  = _auto_params_path("rf",  args.feature, args.n_mels, args.hop, args.fmin, fmax, args.aug)
    svm_json = _maybe_load_json(args.svm_params_json or auto_svm_json)
    rf_json  = _maybe_load_json(args.rf_params_json  or auto_rf_json)
    if "svm" in args.models:
        svm_params = _apply_manual_overrides(svm_json, manual_svm)
        if svm_params:
            svm.set_params(**svm_params)
    if "rf" in args.models:
        rf_json_clean = {k.replace("rf__", ""): v for k, v in rf_json.items()} if rf_json else {}
        rf_params = _apply_manual_overrides(rf_json_clean, manual_rf)
        if rf_params:
            rf.set_params(**rf_params)
    print("\nTraining & evaluation")
    if "svm" in args.models:
        report("SVM (RBF)", svm, X_tr, y_tr, X_te, y_te, le.classes_)
        dump({"label_encoder": le, "svm": svm}, f"{MODELS_DIR}/svm.joblib")
    if "rf" in args.models:
        report("Random Forest", rf, X_tr, y_tr, X_te, y_te, le.classes_)
        dump({"label_encoder": le, "rf": rf}, f"{MODELS_DIR}/rf.joblib")

    print(f"\nSaved models to: {MODELS_DIR}")

if __name__ == "__main__":
    main()
