# TAU Urban Acoustic Scenes 2019 Project
Website: https://haozi672.github.io/ECEN-758-Project/

Description here WIP
---

## Quickstart
1.New VM recommanded as we will be adding more library later for CNN
```bash
python -m venv .venv && source .venv/bin/activate
```
```bash
# requirement
pip install -r Requirement.txt --quiet
```
2.Clone this repo into your desired file location. Due to the size of the dataset they are not included here which means you should download the dataset into the following directory:
```bash
./audio
./evaluation_setup
./meta.csv
```
3.If you only want to run the evaluation code, the trained models are attached all information will be obtailed by running the following file:
```bash
Model_Comparison.ipynb
```
4.If you want to run the training files, you will have to download the dataset(37gbs) and follow the file structure provided in step 1 and run any of the following files:
```bash
SVM.ipynb
RF.ipynb
CNN_with_MetricLearning.ipynb
```
5. Additionally, the EDA file also require to download the dataset, and you can run it with
```bash
EDA_and_DataPrep.ipynb
...
```
---

## Project Features

WIP

---

## Requirements

- Python 3.9+ (3.10/3.11 also fine)
- `numpy`, `scipy`, `pandas`, `scikit-learn`, `librosa`, `tqdm`, `joblib`

Install:
```bash
pip install -U numpy scipy pandas scikit-learn librosa tqdm joblib
```

---

## Dataset

TAU Urban Acoustic Scenes 2019: https://zenodo.org/records/2589280 

Example rows:
```
audio/airport-lisbon-1000-40000-a.wav    airport            lisbon-1000   a
audio/bus-lyon-1001-40001-a.wav          bus                lyon-1001     a
audio/street_pedestrian-milan-1005-...   street_pedestrian  milan-1005    a
```

---

## Cross‑Validation (CV)

### Simple CV (recommended starting point)

**MFCC, no augmentation**
```bash
python cv_search.py --feature mfcc --aug 0 --folds 5
```

**MFCC, with augmentation (1 per clip)**
```bash
python cv_search.py --feature mfcc --aug 1 --folds 5
```

**Log‑Mel (128 mels), no augmentation**
```bash
python cv_search.py --feature logmel --n_mels 128 --hop 1024 --fmin 50 --fmax 0 \
  --aug 0 --folds 5
```

**Log‑Mel (128 mels), with augmentation**
```bash
python cv_search.py --feature logmel --n_mels 128 --hop 1024 --fmin 50 --fmax 0 \
  --aug 1 --folds 5
```
---

## Training

`train.py` uses a **grouped hold‑out** (by `identifier`) for a quick final evaluation and will **auto‑load** the JSON produced by CV **matching your feature + aug flags**.

**Train SVM & RF — MFCC, no augmentation**
```bash
python train.py --feature mfcc --aug 0 --models svm rf
```

**Train SVM only — Log‑Mel 128, light augmentation**
```bash
python train.py --feature logmel --n_mels 128 --hop 1024 --fmin 50 --fmax 0 \
  --aug 1 --models svm
```

### Manual overrides (if you want)

- Use best‑params **JSON** explicitly:
  ```bash
  python train.py --feature mfcc --aug 0 --models svm \
    --svm_params_json .cache/best_svm_mfcc_aug0.json
  ```

- Or pass **flags** (highest priority):
  ```bash
  python train.py --feature mfcc --aug 0 --models svm \
    --svm_C 10 --svm_gamma 0.001

  python train.py --feature mfcc --aug 0 --models rf \
    --rf_n_estimators 900 --rf_max_depth 60 --rf_max_features sqrt
  ```

`train.py` precedence: **manual flags > params JSON > built‑in defaults**.

---

## Configuration

All defaults and grids live in **`config.py`**:
- Audio: `SR`, `N_MFCC`
- Feature defaults (MFCC / Log‑Mel)
- Cache directories
- Split/random seeds
- **SVM/RF grids** (`SVM_GRID`, `RF_GRID`) + `expand_grid()`
- Pre‑expanded: `SVM_PARAM_SPACE_DEFAULT`, `RF_PARAM_SPACE_DEFAULT`

Change grids in **one place** and both CV and training will pick them up.

---

## License
WIP

---

## Citation
WIP
