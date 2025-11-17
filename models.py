from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def get_svm():
    return Pipeline([("scaler", StandardScaler()),("svm", SVC(kernel="rbf", class_weight="balanced"))])

def get_rf(random_state=42):
    return RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state
    )
