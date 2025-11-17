import argparse, json, os, sys
import numpy as np
import librosa
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score
from config import SR, N_MFCC, FEATURES_CACHE, RANDOM_STATE,SVM_PARAM_SPACE_DEFAULT, RF_PARAM_SPACE_DEFAULT
from data import read_meta
from features import build_feature_matrix, extract_from_wave,build_logmel_stats_matrix, logmel_stats_from_wave
from models import get_svm, get_rf
from augment import make_aug_variants

def _feat_from_wave(y, feature_kind, n_mels, hop, fmin, fmax):
    if feature_kind == "logmel":
        return logmel_stats_from_wave(y, sr=SR, n_mels=n_mels, hop=hop, fmin=fmin, fmax=fmax, use_zscore=True)
    return extract_from_wave(y, sr=SR, n_mfcc=N_MFCC)

def _save_best_json(path, params, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"params": params, "meta": meta}, f, indent=2)
    print(f"[saved] {path}")

def _feature_tag(args, fmax):
    if args.feature == "mfcc":
        return "mfcc"
    else:
        return f"logmel_m{args.n_mels}_h{args.hop}_f{args.fmin}-{0 if fmax is None else fmax}"

def run_cv_for_estimator(estimator_name, param_space, X_all, y_enc, groups, filepaths,feature_kind, n_mels, hop, fmin, fmax, aug_per_clip, n_splits):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = defaultdict(list)
    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_all, y_enc, groups)):
        X_va, y_va = X_all[va_idx], y_enc[va_idx]
        X_tr_parts = [X_all[tr_idx]]
        y_tr_parts = [y_enc[tr_idx]]
        if aug_per_clip > 0:
            for i in tqdm(tr_idx, desc=f"[fold {fold_idx}] aug", leave=False):
                y_wav, _ = librosa.load(filepaths[i], sr=SR, mono=True)
                for z in make_aug_variants(y_wav, n_aug=aug_per_clip, sr=SR)[1:]:
                    X_tr_parts.append(_feat_from_wave(z, feature_kind, n_mels, hop, fmin, fmax)[None, :])
                    y_tr_parts.append(np.array([y_enc[i]], dtype=np.int64))
        X_tr = np.vstack(X_tr_parts).astype(np.float32)
        y_tr = np.concatenate(y_tr_parts)

        for params in param_space:
            if estimator_name == "svm":
                model = get_svm(); model.set_params(**params)
            else:#rf
                model = get_rf(RANDOM_STATE)
                model.set_params(**{k.replace("rf__", ""): v for k, v in params.items()})

            model.fit(X_tr, y_tr)
            acc = accuracy_score(y_va, model.predict(X_va))
            scores[tuple(sorted(params.items()))].append(acc)
            print(f"[{estimator_name.upper()} fold {fold_idx}] {params} -> {acc:.4f}")
    summarized = []
    for k, accs in scores.items():
        params = dict(k)
        summarized.append((float(np.mean(accs)), float(np.std(accs)), params))
    summarized.sort(key=lambda t: t[0], reverse=True)
    best_mean, best_std, best_params = summarized[0]
    print(f"\n>>> Best {estimator_name.upper()}: {best_params} | CV {best_mean:.4f} ± {best_std:.4f}")
    return best_params, best_mean, best_std


def main():
    ap = argparse.ArgumentParser(description="Grouped k-fold CV (simple) with optional train-fold augmentation.")
    ap.add_argument("--meta", type=str, default="meta.csv")
    ap.add_argument("--feature", choices=["mfcc","logmel"], default="logmel")
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--hop", type=int, default=1024)
    ap.add_argument("--fmin", type=int, default=50)
    ap.add_argument("--fmax", type=int, default=0, help="0 => sr/2")
    ap.add_argument("--aug", type=int, default=1, help="# augs per clip in TRAIN folds (0=off)")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    fmax = None if args.fmax == 0 else args.fmax

    #data
    meta = read_meta(args.meta)
    filepaths = meta["filepath"].to_list()
    groups = meta["identifier"].to_numpy()
    y_enc = LabelEncoder().fit_transform(meta["scene_label"])
    if args.feature == "logmel":
        X_all = build_logmel_stats_matrix(filepaths, sr=SR, n_mels=args.n_mels, hop=args.hop,
                                          fmin=args.fmin, fmax=fmax, use_zscore=True)
    else:
        X_all = build_feature_matrix(filepaths, cache_path=FEATURES_CACHE, sr=SR, n_mfcc=N_MFCC)
    svm_space = SVM_PARAM_SPACE_DEFAULT
    rf_space  = RF_PARAM_SPACE_DEFAULT
    print("\nSVM search")
    svm_best, svm_mean, svm_std = run_cv_for_estimator(
        "svm", svm_space, X_all, y_enc, groups, filepaths,
        args.feature, args.n_mels, args.hop, args.fmin, fmax, args.aug, args.folds
    )
    print("\nRF search")
    rf_best, rf_mean, rf_std = run_cv_for_estimator(
        "rf", rf_space, X_all, y_enc, groups, filepaths,
        args.feature, args.n_mels, args.hop, args.fmin, fmax, args.aug, args.folds
    )
    tag = _feature_tag(args, fmax)
    meta_common = {
        "feature": args.feature,
        "n_mels": (args.n_mels if args.feature == "logmel" else None),
        "hop": (args.hop if args.feature == "logmel" else None),
        "fmin": (args.fmin if args.feature == "logmel" else None),
        "fmax": (fmax if args.feature == "logmel" else None),
        "aug": args.aug, "folds": args.folds,
    }

    if svm_best is not None:
        _save_best_json(f".cache/best_svm_{tag}_aug{args.aug}.json", svm_best, meta_common)
    if rf_best is not None:
        rf_best_clean = {k.replace("rf__", ""): v for k, v in rf_best.items()}
        _save_best_json(f".cache/best_rf_{tag}_aug{args.aug}.json", rf_best_clean, meta_common)

    print("\nDone.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)
