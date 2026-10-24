import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from core import (PATHS, apply_reference_quantiles, beta_features, evaluate,
                  expected_f1_threshold, select_threshold, validation_quantiles)


def load_scores(path, validation=False):
    with np.load(path, allow_pickle=False) as data:
        names = data['names'].astype(str)
        matrix = np.column_stack([data[key] for key in PATHS]).astype(np.float64)
        labels = data['labels'].astype(np.int64) if validation else None
        if validation and 'valid_index' in data.files:
            index = data['valid_index'].astype(np.int64)
            names, matrix, labels = names[index], matrix[index], labels[index]
    if len(names) != len(set(names)) or matrix.shape != (len(names), 4) or not np.isfinite(matrix).all():
        raise ValueError('Invalid names or path-score dimensions')
    return names, matrix, labels


def fit(matrix, labels, candidates):
    ranks = np.column_stack([validation_quantiles(matrix[:, i]) for i in range(4)])
    screened = []
    for vector in candidates:
        weights = np.asarray(vector, dtype=np.float64)
        score = ranks @ weights
        result = evaluate(labels, score, select_threshold(labels, score))
        screened.append(((result['mcc'], result['f1'], result['auc']), weights))
    weights = max(screened, key=lambda item: item[0])[1]
    score = ranks @ weights
    model = LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs', max_iter=1000)
    model.fit(beta_features(score), labels)
    return {'paths': list(PATHS), 'reference_scores': matrix.tolist(),
            'weights': weights.tolist(), 'beta_coef': model.coef_[0].tolist(),
            'beta_intercept': float(model.intercept_[0]),
            'validation_auc': float(roc_auc_score(labels, score)),
            'auc_boundary': 0.9}


def predict(matrix, model):
    if model['paths'] != list(PATHS):
        raise ValueError('Path order differs from the fitted model')
    reference = np.asarray(model['reference_scores'], dtype=np.float64)
    ranks = np.column_stack([apply_reference_quantiles(reference[:, i], matrix[:, i]) for i in range(4)])
    fused = ranks @ np.asarray(model['weights'])
    probabilities = expit(beta_features(fused) @ np.asarray(model['beta_coef']) + model['beta_intercept'])
    threshold = 0.5 if model['validation_auc'] >= model['auc_boundary'] else expected_f1_threshold(probabilities)
    return probabilities, float(threshold)


def main():
    parser = argparse.ArgumentParser(description='Fit rank fusion, predict, or evaluate probabilities.')
    sub = parser.add_subparsers(dest='action', required=True)
    fitting = sub.add_parser('fit')
    fitting.add_argument('--source-set', choices=['Train-1698', 'Train-2104'], required=True)
    fitting.add_argument('--scores', type=Path, required=True)
    fitting.add_argument('--output', type=Path, required=True)
    prediction = sub.add_parser('predict')
    prediction.add_argument('--scores', type=Path, required=True)
    prediction.add_argument('--model', type=Path, required=True)
    prediction.add_argument('--output', type=Path, required=True)
    evaluation = sub.add_parser('evaluate')
    evaluation.add_argument('--predictions', type=Path, required=True)
    evaluation.add_argument('--labels', type=Path, required=True)
    evaluation.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.action == 'fit':
        names, matrix, labels = load_scores(args.scores, validation=True)
        config = json.loads((ROOT/'configs/fusion.json').read_text(encoding='utf-8'))
        model = fit(matrix, labels, config['candidate_weights'][args.source_set])
        model['source_set'] = args.source_set
        args.output.write_text(json.dumps(model, indent=2)+'\n', encoding='utf-8')
    elif args.action == 'predict':
        names, matrix, _ = load_scores(args.scores)
        model = json.loads(args.model.read_text(encoding='utf-8'))
        probability, threshold = predict(matrix, model)
        np.savez_compressed(args.output, names=names, probabilities=probability, threshold=np.asarray([threshold]))
    else:
        with np.load(args.predictions, allow_pickle=False) as prediction, np.load(args.labels, allow_pickle=False) as labels:
            np.testing.assert_array_equal(prediction['names'].astype(str), labels['names'].astype(str))
            result = evaluate(labels['labels'], prediction['probabilities'], float(prediction['threshold'][0]))
        args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
