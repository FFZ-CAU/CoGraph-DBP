from __future__ import annotations
import numpy as np
from scipy.stats import rankdata
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
PATHS = ('reference', 'structure', 'topology', 'residue')

def validation_quantiles(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (rankdata(values, method='average') - 1.0) / (len(values) - 1.0)

def apply_reference_quantiles(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=np.float64))
    values = np.asarray(values, dtype=np.float64)
    left = np.searchsorted(reference, values, side='left')
    right = np.searchsorted(reference, values, side='right')
    mapped = (left + right - 1.0) / (2.0 * (len(reference) - 1.0))
    return np.clip(mapped, 0.0, 1.0)

def select_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    best_key = None
    best_threshold = float(np.median(scores))
    for threshold in np.unique(scores):
        predictions = (scores >= threshold).astype(np.int64)
        key = (float(metrics.matthews_corrcoef(labels, predictions)), float(metrics.f1_score(labels, predictions, zero_division=0)), float(metrics.accuracy_score(labels, predictions)))
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold

def evaluate(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = metrics.confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision_curve, recall_curve, _ = metrics.precision_recall_curve(labels, scores)
    return {'n': int(len(labels)), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp), 'threshold': float(threshold), 'specificity': float(tn / (tn + fp)), 'accuracy': float(metrics.accuracy_score(labels, predictions)), 'precision': float(metrics.precision_score(labels, predictions, zero_division=0)), 'recall': float(metrics.recall_score(labels, predictions, zero_division=0)), 'f1': float(metrics.f1_score(labels, predictions, zero_division=0)), 'auc': float(metrics.roc_auc_score(labels, scores)), 'auprc': float(metrics.auc(recall_curve, precision_curve)), 'mcc': float(metrics.matthews_corrcoef(labels, predictions))}

def beta_features(scores: np.ndarray) -> np.ndarray:
    scores = np.clip(np.asarray(scores, dtype=np.float64), 1e-06, 1.0 - 1e-06)
    return np.column_stack((np.log(scores), -np.log1p(-scores)))

def calibrate(validation_scores: np.ndarray, validation_labels: np.ndarray, external_scores: np.ndarray) -> np.ndarray:
    model = LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs', max_iter=1000)
    model.fit(beta_features(validation_scores), validation_labels)
    return model.predict_proba(beta_features(external_scores))[:, 1]

def expected_f1_threshold(probabilities: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    ranked = np.sort(probabilities)[::-1]
    cumulative_tp = np.cumsum(ranked)
    predicted_positive = np.arange(1, len(ranked) + 1, dtype=np.float64)
    expected_f1 = 2.0 * cumulative_tp / (predicted_positive + probabilities.sum())
    best_k = int(np.argmax(expected_f1)) + 1
    return float(ranked[best_k - 1])
