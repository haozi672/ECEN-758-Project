from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_svm():
    return Pipeline([("scaler", StandardScaler()),("svm", SVC(kernel="rbf", class_weight="balanced"))])

def get_rf(random_state=42):
    return RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state
    )


class CNNAcousticScene(nn.Module):
    def __init__(self, n_classes: int, n_mels: int, n_frames: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_mels, n_frames)
            out = self.features(dummy)
            flat_dim = out.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        z = self.features(x)
        z = z.view(z.size(0), -1)
        logits = self.classifier(z)
        return logits

def get_cnn(n_classes: int, n_mels: int, n_frames: int) -> nn.Module:
    return CNNAcousticScene(n_classes=n_classes, n_mels=n_mels, n_frames=n_frames)