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
```
## Dataset

TAU Urban Acoustic Scenes 2019: https://zenodo.org/records/2589280 

Example rows:
```
audio/airport-lisbon-1000-40000-a.wav    airport            lisbon-1000   a
audio/bus-lyon-1001-40001-a.wav          bus                lyon-1001     a
audio/street_pedestrian-milan-1005-...   street_pedestrian  milan-1005    a
```
You can also acquire the dataset with google drive link:
https://drive.google.com/drive/folders/1POLnlVu4lNKPy9V2-_Gh_vZY70iRoBZP?usp=sharing

https://drive.google.com/drive/folders/1FTAqSvYaIZ-bTZG0sidt9bvX-U3JO_d_?usp=sharing
---
