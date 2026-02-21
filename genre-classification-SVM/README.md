# 🎵 Music Genre Classification — SVM Baseline
**GTZAN Dataset | MFCC Features | Scikit-learn**

A reproducible classical ML pipeline for automatic music genre classification.
Built as a clean baseline before extending toward temporal and deep learning approaches (LSTM, CNN).

---

## Overview

Classifies 30-second audio clips into 10 genres using handcrafted MFCC features and an RBF-kernel SVM.

**Stack:** `librosa` · `scikit-learn` · `numpy` · `matplotlib` · `seaborn`

---

## Setup
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Dataset:** Download [GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) and place under:
```
data/genres_original/<genre>/*.wav
```

---

## Run
```bash
python train.py
```

Outputs saved to `outputs/`: cached features, trained model, confusion matrix.

---

## Methodology

**Feature extraction:** 13 MFCC coefficients per file → mean across time → 26-dim vector (mean + std)

**Pipeline:** `StandardScaler` → `SVC(kernel="rbf")`

**Split:** 80/20 train/test, `random_state=42`

999/1000 files loaded (1 corrupted file skipped automatically).

---

## Results

**Test Accuracy: 63%**

| Genre | F1 |
|---|---|
| Metal | 0.87 |
| Pop | 0.76 |
| Classical | 0.74 |
| Jazz | 0.67 |
| Blues | 0.65 |
| Country | 0.57 |
| Hip-Hop | 0.52 |
| Disco | 0.52 |
| Reggae | 0.51 |
| Rock | 0.51 |

![Confusion Matrix](outputs/ConfusionMatrix.png)

---

## Next Steps

- MFCC variance features + segment-level classification
- Hyperparameter tuning (GridSearchCV) + k-fold cross-validation  
- LSTM for temporal modelling
- CNN on mel-spectrograms

---

## Author

Krishna Vinod · MEng Computer Science, University of Birmingham