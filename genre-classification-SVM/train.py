from sklearn.model_selection import train_test_split
from features import load_dataset
from model import train_svm
from evaluate import evaluate_model

import os
import pandas as pd
import joblib


# -----------------------
# config
# -----------------------
DATA_DIR = "data/genres_original"
OUTPUT_DIR = "outputs"
CACHE_PATH = os.path.join(OUTPUT_DIR, "features_cached.csv")
MODEL_PATH = os.path.join(OUTPUT_DIR, "svm_model.pkl")


# make sure outputs folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------
# 1) load or extract features
# -----------------------
if os.path.exists(CACHE_PATH):
    print("loading cached features...")
    df = pd.read_csv(CACHE_PATH)
    X = df.drop("label", axis=1).values
    y = df["label"].values
    print(f"loaded {len(df)} samples from cache")
else:
    print("extracting features from audio files...")
    X, y = load_dataset(DATA_DIR)

    df = pd.DataFrame(X)
    df["label"] = y
    df.to_csv(CACHE_PATH, index=False)

    print(f"saved features to {CACHE_PATH}")


# -----------------------
# 2) train/test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------
# 3) train model
# -----------------------
print("training svm model...")
svm_model = train_svm(X_train, y_train)


# -----------------------
# 4) save trained model
# -----------------------
joblib.dump(svm_model, MODEL_PATH)
print(f"saved model to {MODEL_PATH}")


# -----------------------
# 5) evaluate
# -----------------------
print("evaluating model...")
genres = sorted(list(set(y)))
evaluate_model(svm_model, X_test, y_test, genre_labels=genres)