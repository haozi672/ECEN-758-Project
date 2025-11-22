import argparse
import os
import random

import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tqdm.auto import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from config import (
    SR,
    FEATURE_DEFAULTS,
    TEST_SIZE,
    RANDOM_STATE,
    MODELS_DIR,
    CNN_DEFAULTS,
    CNN_MODEL_PATH,
)
from data import read_meta, grouped_holdout
from augment import make_aug_variants
from models import get_cnn

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class LogMelCNNDataset(Dataset):
    def __init__(
        self,
        meta,
        indices,
        label_encoder: LabelEncoder,
        n_mels: int,
        hop: int,
        fmin: int,
        fmax,
        max_frames: int,
        train: bool,
        n_aug: int,
    ):
        self.meta = meta.reset_index(drop=True)
        self.indices = np.array(indices, dtype=int)
        self.le = label_encoder

        self.n_mels = n_mels
        self.hop = hop
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else SR // 2
        self.max_frames = max_frames

        self.train = train
        self.n_aug = n_aug

    def __len__(self):
        return len(self.indices)

    def _wav_to_logmel(self, y: np.ndarray) -> np.ndarray:
        M = librosa.feature.melspectrogram(
            y=y,
            sr=SR,
            n_mels=self.n_mels,
            hop_length=self.hop,
            fmin=self.fmin,
            fmax=self.fmax,
            power=2.0,
        )
        L = librosa.power_to_db(M, ref=np.max).astype(np.float32)
        # per-clip z-score
        mu = float(L.mean())
        sd = float(L.std() + 1e-8)
        L = (L - mu) / sd
        return L

    def _pad_or_crop(self, S: np.ndarray) -> np.ndarray:
        T = S.shape[1]
        if T < self.max_frames:
            pad_width = self.max_frames - T
            S = np.pad(S, ((0, 0), (0, pad_width)), mode="constant")
        else:
            S = S[:, : self.max_frames]
        return S

    def __getitem__(self, idx):
        idx = self.indices[idx]
        row = self.meta.iloc[idx]
        path = row["filepath"]

        y, _ = librosa.load(path, sr=SR, mono=True)
        if self.train and self.n_aug > 0:
            variants = make_aug_variants(y, n_aug=self.n_aug, sr=SR)
            choice = random.randint(0, len(variants) - 1)
            y = variants[choice].astype(np.float32)

        S = self._wav_to_logmel(y)
        S = self._pad_or_crop(S)
        x = torch.from_numpy(S).float().unsqueeze(0)

        label_str = row["scene_label"]
        y_idx = self.le.transform([label_str])[0]
        y_t = torch.tensor(y_idx, dtype=torch.long)

        return x, y_t

def train_one_epoch(model, loader, optimizer, criterion, device, epoch=None, total_epochs=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    desc = "Train"
    if epoch is not None and total_epochs is not None:
        desc = f"Train [{epoch}/{total_epochs}]"

    progress_bar = tqdm(loader, desc=desc, leave=False)

    for xb, yb in progress_bar:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * yb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

        avg_loss = running_loss / max(total, 1)
        acc = correct / max(total, 1)
        progress_bar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.4f}")

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def evaluate(model, loader, criterion, device, split_name="Val"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []

    progress_bar = tqdm(loader, desc=f"{split_name}", leave=False)

    with torch.no_grad():
        for xb, yb in progress_bar:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            running_loss += loss.item() * yb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

            avg_loss = running_loss / max(total, 1)
            acc = correct / max(total, 1)
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.4f}")

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    if all_preds:
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
    else:
        all_preds = np.array([])
        all_targets = np.array([])

    return avg_loss, acc, all_preds, all_targets

def main():
    parser = argparse.ArgumentParser(
        description="Train a CNN on log-mel spectrograms for TAU 2019 ASC."
    )
    parser.add_argument("--meta", type=str, default="meta.csv")
    parser.add_argument("--batch_size", type=int, default=CNN_DEFAULTS["batch_size"])
    parser.add_argument("--epochs", type=int, default=CNN_DEFAULTS["epochs"])
    parser.add_argument("--lr", type=float, default=CNN_DEFAULTS["lr"])
    parser.add_argument("--weight_decay", type=float, default=CNN_DEFAULTS["weight_decay"])
    parser.add_argument("--n_mels", type=int, default=CNN_DEFAULTS["n_mels"])
    parser.add_argument("--hop", type=int, default=CNN_DEFAULTS["hop"])
    parser.add_argument("--fmin", type=int, default=CNN_DEFAULTS["fmin"])
    parser.add_argument(
        "--fmax",
        type=int,
        default=0,
        help="0 => use config/default (None -> sr/2)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=CNN_DEFAULTS["max_frames"],
        help="Number of time frames to keep after pad/crop.",
    )
    parser.add_argument(
        "--n_aug",
        type=int,
        default=CNN_DEFAULTS["n_aug"],
        help="# waveform-level aug variants per clip (0 = no aug).",
    )

    args = parser.parse_args()
    cfg_fmax = CNN_DEFAULTS["fmax"]
    if args.fmax == 0:
        fmax = cfg_fmax
    else:
        fmax = args.fmax

    set_global_seed(RANDOM_STATE)
    meta = read_meta(args.meta)
    tr_idx, te_idx = grouped_holdout(meta, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    le = LabelEncoder()
    le.fit(meta["scene_label"])

    #data
    train_ds = LogMelCNNDataset(
        meta=meta,
        indices=tr_idx,
        label_encoder=le,
        n_mels=args.n_mels,
        hop=args.hop,
        fmin=args.fmin,
        fmax=fmax,
        max_frames=args.max_frames,
        train=True,
        n_aug=args.n_aug,
    )
    test_ds = LogMelCNNDataset(
        meta=meta,
        indices=te_idx,
        label_encoder=le,
        n_mels=args.n_mels,
        hop=args.hop,
        fmin=args.fmin,
        fmax=fmax,
        max_frames=args.max_frames,
        train=False,
        n_aug=0,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    n_classes = len(le.classes_)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_cnn(n_classes=n_classes, n_mels=args.n_mels, n_frames=args.max_frames)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(f"Using device: {device}")
    print(f"Train size: {len(train_ds)} | Test size: {len(test_ds)}")
    print(f"Classes: {list(le.classes_)}")

    best_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch=epoch,
            total_epochs=args.epochs,
        )
        test_loss, test_acc, _, _ = evaluate(
            model,
            test_loader,
            criterion,
            device,
            split_name="Test",
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_loss:.4f}, acc={train_acc:.4f} | "
            f"test loss={test_loss:.4f}, acc={test_acc:.4f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {
                "model_state_dict": model.state_dict(),
                "label_encoder_classes_": le.classes_,
                "n_mels": args.n_mels,
                "hop": args.hop,
                "fmin": args.fmin,
                "fmax": fmax,
                "max_frames": args.max_frames,
                "cnn_hparams": {
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "n_aug": args.n_aug,
                },
            }

    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])
    test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, device)
    print("\nFinal evaluation on hold-out test split")
    print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")
    if y_true.size > 0:
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, target_names=le.classes_))
        print("\nConfusion matrix:")
        print(confusion_matrix(y_true, y_pred))

    os.makedirs(MODELS_DIR, exist_ok=True)
    if best_state is None:
        best_state = {
            "model_state_dict": model.state_dict(),
            "label_encoder_classes_": le.classes_,
            "n_mels": args.n_mels,
            "hop": args.hop,
            "fmin": args.fmin,
            "fmax": fmax,
            "max_frames": args.max_frames,
            "cnn_hparams": {
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "n_aug": args.n_aug,
            },
        }

    torch.save(best_state, CNN_MODEL_PATH)
    print(f"\nSaved best CNN model to: {CNN_MODEL_PATH}")


if __name__ == "__main__":
    main()
