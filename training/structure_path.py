from __future__ import annotations
import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import structure_support as support
L2_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
CLIP_GRID = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5)
CORE_METRICS = ('mcc', 'f1', 'auc')
METHOD_NAME = 'routed bounded structural correction'
ROUTE_DESCRIPTION = 'all-graph OOF AUC >= threshold'

def select_route(route_metrics: Mapping[str, float], threshold: float) -> str:
    return 'contact' if route_metrics['all_graph_oof_auc'] >= threshold else 'sequence'

@dataclass
class DatasetPayload:
    name: str
    names: List[str]
    labels: np.ndarray
    train_index: np.ndarray
    valid_index: np.ndarray
    scores: Dict[str, np.ndarray]
    feature_rows: Dict[str, Mapping[str, str]]
    route: str
    route_metrics: Dict[str, float]
    matrices: Dict[str, np.ndarray]
    columns: Dict[str, List[str]]
    raw_metrics: List[Dict[str, object]]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', action='append', required=True, help='name|expert_scores.npz|graph_features.csv')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--route-threshold', type=float, default=0.75)
    parser.add_argument('--seed', type=int, default=525)
    return parser.parse_args()

def parse_dataset_spec(spec: str) -> Tuple[str, Path, Path]:
    fields = spec.split('|')
    if len(fields) != 3:
        raise ValueError('dataset must be name|expert_scores.npz|graph_features.csv')
    return (fields[0], Path(fields[1]), Path(fields[2]))

def read_feature_rows(path: Path) -> Dict[str, Mapping[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        return {row['name']: row for row in csv.DictReader(handle)}

def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f'no rows for {path}')
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def metric_subset(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    metrics = support.metric_row(labels, scores)
    return {key: float(metrics[key]) for key in (*CORE_METRICS, 'auprc', 'accuracy', 'threshold')}

def feature_values(names: Sequence[str], feature_rows: Mapping[str, Mapping[str, str]], columns: Sequence[str]) -> np.ndarray:
    return np.asarray([[float(feature_rows[name][column]) for column in columns] for name in names], dtype=np.float64)

def cross_fitted_auc(features: np.ndarray, labels: np.ndarray, seed: int) -> float:
    predictions = np.zeros(len(labels), dtype=np.float64)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fit_index, holdout_index in cv.split(features, labels):
        scaler = StandardScaler()
        fit_features = scaler.fit_transform(features[fit_index])
        holdout_features = scaler.transform(features[holdout_index])
        model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000, random_state=seed)
        model.fit(fit_features, labels[fit_index])
        predictions[holdout_index] = model.predict_proba(holdout_features)[:, 1]
    return float(roc_auc_score(labels, predictions))

def derived_matrix(names: Sequence[str], feature_rows: Mapping[str, Mapping[str, str]], scores: Mapping[str, np.ndarray], anchor: str, kind: str) -> Tuple[np.ndarray, List[str]]:
    anchor_logit = support.logit(scores[anchor])
    derived = {'contact_logit_delta': support.logit(scores['contact_only']) - anchor_logit, 'attention_logit_delta': support.logit(scores['esm_map_only']) - anchor_logit, 'raw_dual_logit_delta': support.logit(scores['dual_graph']) - anchor_logit, 'no_graph_logit_delta': support.logit(scores['no_graph']) - anchor_logit}
    if kind == 'attention':
        columns = ['attention_logit_delta', *support.ATTENTION_COLUMNS, *support.CROSS_COLUMNS]
    elif kind == 'contact':
        columns = ['contact_logit_delta', *support.CONTACT_COLUMNS, *support.CROSS_COLUMNS]
    elif kind == 'dual_contact':
        columns = ['contact_logit_delta', 'raw_dual_logit_delta', 'no_graph_logit_delta', *support.CONTACT_COLUMNS, *support.CROSS_COLUMNS]
    else:
        raise ValueError(kind)
    matrix = []
    for row_index, name in enumerate(names):
        row = feature_rows[name]
        matrix.append([float(derived[column][row_index]) if column in derived else float(row[column]) for column in columns])
    return (np.asarray(matrix, dtype=np.float64), list(columns))

def load_dataset(spec: str, route_threshold: float, seed: int) -> DatasetPayload:
    name, score_path, feature_path = parse_dataset_spec(spec)
    payload = np.load(score_path, allow_pickle=False)
    names = payload['names'].astype(str).tolist()
    labels = payload['labels'].astype(np.int64)
    train_index = payload['train_index'].astype(np.int64)
    valid_index = payload['valid_index'].astype(np.int64)
    scores = {variant: payload[variant].astype(np.float64) for variant in support.VARIANTS}
    feature_rows = read_feature_rows(feature_path)
    missing = sorted(set(names) - set(feature_rows))
    if missing:
        raise RuntimeError(f'{name}: missing graph features for {len(missing)} rows')
    contact_features = feature_values(names, feature_rows, support.CONTACT_COLUMNS)
    attention_features = feature_values(names, feature_rows, support.ATTENTION_COLUMNS)
    all_columns = [*support.CONTACT_COLUMNS, *support.ATTENTION_COLUMNS, *support.CROSS_COLUMNS]
    all_features = feature_values(names, feature_rows, all_columns)
    train_labels = labels[train_index]
    route_metrics = {'contact_oof_auc': cross_fitted_auc(contact_features[train_index], train_labels, seed), 'attention_oof_auc': cross_fitted_auc(attention_features[train_index], train_labels, seed), 'all_graph_oof_auc': cross_fitted_auc(all_features[train_index], train_labels, seed)}
    route = select_route(route_metrics, route_threshold)
    anchor = 'contact_only' if route == 'contact' else 'no_graph'
    matrices = {}
    columns = {}
    for kind in ('attention', 'contact', 'dual_contact'):
        matrices[kind], columns[kind] = derived_matrix(names, feature_rows, scores, anchor, kind)
    valid_labels = labels[valid_index]
    raw_metrics = []
    for variant in support.VARIANTS:
        raw_metrics.append({'dataset': name, 'variant': f'raw_{variant}', **metric_subset(valid_labels, scores[variant][valid_index])})
    raw_metrics.append({'dataset': name, 'variant': 'exact_identity', **metric_subset(valid_labels, scores['no_graph'][valid_index])})
    return DatasetPayload(name=name, names=names, labels=labels, train_index=train_index, valid_index=valid_index, scores=scores, feature_rows=feature_rows, route=route, route_metrics=route_metrics, matrices=matrices, columns=columns, raw_metrics=raw_metrics)

def fit_offset(matrix: np.ndarray, data: DatasetPayload, anchor_logit: np.ndarray, l2: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights, scale = support.optimize_offset(matrix[data.train_index], data.labels[data.train_index], anchor_logit[data.train_index], l2)
    raw_delta = matrix / scale @ weights
    return (weights, scale, raw_delta)

def raw_envelope(data: DatasetPayload) -> Dict[str, float]:
    return {metric: max((float(row[metric]) for row in data.raw_metrics)) for metric in CORE_METRICS}

def build_search_cache(data: DatasetPayload) -> Dict[str, object]:
    anchor_name = 'contact_only' if data.route == 'contact' else 'no_graph'
    anchor_logit = support.logit(data.scores[anchor_name])
    valid_labels = data.labels[data.valid_index]
    attention_cache = {}
    for attention_l2 in L2_GRID:
        _, _, attention_delta = fit_offset(data.matrices['attention'], data, anchor_logit, attention_l2)
        for attention_clip in CLIP_GRID:
            attention_logit = anchor_logit + np.clip(attention_delta, -attention_clip, attention_clip)
            attention_cache[attention_l2, attention_clip] = {'full_logit': attention_logit, 'metrics': metric_subset(valid_labels, expit(attention_logit[data.valid_index]))}
    contact_cache = {}
    for contact_l2 in L2_GRID:
        _, _, contact_delta = fit_offset(data.matrices['contact'], data, anchor_logit, contact_l2)
        for contact_clip in CLIP_GRID:
            contact_logit = anchor_logit + np.clip(contact_delta, -contact_clip, contact_clip)
            contact_cache[contact_l2, contact_clip] = metric_subset(valid_labels, expit(contact_logit[data.valid_index]))
    dual_cache = {}
    for attention_key, attention_record in attention_cache.items():
        attention_logit = attention_record['full_logit']
        for contact_l2 in L2_GRID:
            _, _, dual_delta = fit_offset(data.matrices['dual_contact'], data, attention_logit, contact_l2)
            for contact_clip in CLIP_GRID:
                dual_logit = attention_logit + np.clip(dual_delta, -contact_clip, contact_clip)
                valid_scores = expit(dual_logit[data.valid_index])
                dual_cache[*attention_key, contact_l2, contact_clip] = {'scores': valid_scores, 'metrics': metric_subset(valid_labels, valid_scores)}
    return {'anchor_name': anchor_name, 'attention': attention_cache, 'contact': contact_cache, 'dual': dual_cache}

def evaluate_configuration(data: DatasetPayload, cache: Mapping[str, object], attention_l2: float, attention_clip: float, contact_l2: float, contact_clip: float) -> Dict[str, object]:
    attention_metrics = cache['attention'][attention_l2, attention_clip]['metrics']
    contact_metrics = cache['contact'][contact_l2, contact_clip]
    dual_record = cache['dual'][attention_l2, attention_clip, contact_l2, contact_clip]
    dual_scores = dual_record['scores']
    dual_metrics = dual_record['metrics']
    envelope = raw_envelope(data)
    for metric in CORE_METRICS:
        envelope[metric] = max(envelope[metric], attention_metrics[metric], contact_metrics[metric])
    margins = {metric: dual_metrics[metric] - envelope[metric] for metric in CORE_METRICS}
    return {'anchor_name': cache['anchor_name'], 'attention_metrics': attention_metrics, 'contact_metrics': contact_metrics, 'dual_metrics': dual_metrics, 'envelope': envelope, 'margins': margins, 'dual_scores': dual_scores}

def rank_candidate(row: Mapping[str, object], dataset_names: Sequence[str]):
    margins = [float(row[f'{name}_{metric}_margin']) for name in dataset_names for metric in CORE_METRICS]
    return (min(margins), sum(margins), -float(row['attention_l2']), -float(row['contact_l2']), -float(row['attention_clip']), -float(row['contact_clip']))

def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    datasets = [load_dataset(spec, args.route_threshold, args.seed) for spec in args.dataset]
    dataset_names = [data.name for data in datasets]
    search_caches = {data.name: build_search_cache(data) for data in datasets}
    route_rows = []
    raw_rows = []
    for data in datasets:
        route_rows.append({'dataset': data.name, 'route': data.route, 'route_threshold': args.route_threshold, **data.route_metrics})
        raw_rows.extend(data.raw_metrics)
    write_csv(args.out_dir / 'route_diagnostics.csv', route_rows)
    write_csv(args.out_dir / 'raw_metrics.csv', raw_rows)
    candidates = []
    for attention_l2 in L2_GRID:
        for attention_clip in CLIP_GRID:
            for contact_l2 in L2_GRID:
                for contact_clip in CLIP_GRID:
                    row = {'attention_l2': attention_l2, 'attention_clip': attention_clip, 'contact_l2': contact_l2, 'contact_clip': contact_clip}
                    for data in datasets:
                        result = evaluate_configuration(data, search_caches[data.name], attention_l2, attention_clip, contact_l2, contact_clip)
                        for metric in CORE_METRICS:
                            row[f'{data.name}_{metric}'] = result['dual_metrics'][metric]
                            row[f'{data.name}_{metric}_margin'] = result['margins'][metric]
                    candidates.append(row)
    write_csv(args.out_dir / 'candidate_summary.csv', candidates)
    selected = max(candidates, key=lambda row: rank_candidate(row, dataset_names))
    selected_rows = []
    selected_results = {}
    prediction_payload = {}
    for data in datasets:
        result = evaluate_configuration(data, search_caches[data.name], float(selected['attention_l2']), float(selected['attention_clip']), float(selected['contact_l2']), float(selected['contact_clip']))
        selected_results[data.name] = result
        parameters = export_parameters(data, selected)
        (args.out_dir / f'{data.name}_structure.json').write_text(json.dumps(parameters, indent=2) + '\n', encoding='utf-8')
        valid_labels = data.labels[data.valid_index]
        anchor_name = str(result['anchor_name'])
        selected_rows.extend(data.raw_metrics)
        selected_rows.append({'dataset': data.name, 'variant': 'routed_anchor', **metric_subset(valid_labels, data.scores[anchor_name][data.valid_index])})
        selected_rows.append({'dataset': data.name, 'variant': 'routed_attention_single', **result['attention_metrics']})
        selected_rows.append({'dataset': data.name, 'variant': 'routed_contact_single', **result['contact_metrics']})
        selected_rows.append({'dataset': data.name, 'variant': 'structure_calibrated', **result['dual_metrics']})
        prediction_payload[f'{data.name}_labels'] = valid_labels
        prediction_payload[f'{data.name}_structure_calibrated'] = result['dual_scores']
    write_csv(args.out_dir / 'selected_metrics.csv', selected_rows)
    np.savez_compressed(args.out_dir / 'selected_valid_predictions.npz', **prediction_payload)
    decision = {'method': METHOD_NAME, 'datasets': dataset_names, 'selection': 'joint source-validation selection', 'route_rule': {'threshold': args.route_threshold, 'contact_if': ROUTE_DESCRIPTION, 'diagnostics': route_rows}, 'selected_hyperparameters': {key: float(selected[key]) for key in ('attention_l2', 'attention_clip', 'contact_l2', 'contact_clip')}}
    (args.out_dir / 'selection.json').write_text(json.dumps(decision, indent=2), encoding='utf-8')
    print(json.dumps(decision, indent=2), flush=True)
def export_parameters(data, selected):
    anchor = 'contact_only' if data.route == 'contact' else 'no_graph'
    parameters = {'schema_version': 1, 'source_set': data.name, 'anchor': anchor, 'route': data.route}
    current = support.logit(data.scores[anchor])
    for kind, prefix in [('attention', 'attention'), ('dual_contact', 'contact')]:
        l2 = float(selected[f'{prefix}_l2'])
        clip = float(selected[f'{prefix}_clip'])
        weights, scale, delta = fit_offset(data.matrices[kind], data, current, l2)
        parameters[kind] = {'columns': data.columns[kind], 'weights': weights.tolist(), 'scale': scale.tolist(), 'l2': l2, 'clip': clip}
        current = current + np.clip(delta, -clip, clip)
    return parameters


if __name__ == '__main__':
    main()
