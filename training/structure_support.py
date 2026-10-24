from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import base_path as base
CONTACT_COLUMNS = ('contact_mean', 'contact_std', 'contact_q95', 'contact_q99', 'contact_topk_mean', 'contact_topk_std', 'contact_row_entropy')
ATTENTION_COLUMNS = ('attention_mean', 'attention_std', 'attention_q95', 'attention_q99', 'attention_topk_mean', 'attention_topk_std', 'attention_row_entropy', 'attention_asymmetry')
CROSS_COLUMNS = ('topk_overlap', 'topk_jaccard', 'topk_score_product', 'offdiag_pearson')
VARIANTS = ('no_graph', 'contact_only', 'esm_map_only', 'identity_graph_control', 'dual_graph')

def logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability.astype(np.float64), 1e-06, 1.0 - 1e-06)
    return np.log(probability) - np.log1p(-probability)

def optimize_offset(features: np.ndarray, labels: np.ndarray, anchor_logit: np.ndarray, l2: float) -> Tuple[np.ndarray, np.ndarray]:
    scale = np.std(features, axis=0)
    scale = np.where(scale > 1e-08, scale, 1.0)
    standardized = features / scale

    def objective(weights: np.ndarray) -> Tuple[float, np.ndarray]:
        prediction_logit = anchor_logit + standardized @ weights
        probability = expit(prediction_logit)
        loss = np.mean(np.logaddexp(0.0, prediction_logit) - labels * prediction_logit)
        loss += 0.5 * l2 * float(np.dot(weights, weights))
        gradient = standardized.T @ (probability - labels) / len(labels)
        gradient += l2 * weights
        return (float(loss), gradient)
    initial = np.zeros(features.shape[1], dtype=np.float64)
    result = minimize(objective, initial, method='L-BFGS-B', jac=True, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-08})
    if not result.success:
        candidate_weights = np.asarray(result.x, dtype=np.float64)
        weights = candidate_weights if np.all(np.isfinite(candidate_weights)) else initial.copy()
        converged = False
        for _ in range(1, 201):
            loss, gradient = objective(weights)
            gradient_norm = float(np.max(np.abs(gradient)))
            if gradient_norm <= 1e-08:
                converged = True
                break
            prediction_logit = anchor_logit + standardized @ weights
            probability = expit(prediction_logit)
            curvature = probability * (1.0 - probability)
            hessian = standardized.T @ (curvature[:, None] * standardized) / len(labels) + l2 * np.eye(standardized.shape[1], dtype=np.float64)
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            directional = float(np.dot(gradient, step))
            step_size = 1.0
            accepted = False
            for _ in range(60):
                candidate = weights - step_size * step
                candidate_loss, _ = objective(candidate)
                if candidate_loss <= loss - 0.0001 * step_size * directional:
                    weights = candidate
                    accepted = True
                    break
                step_size *= 0.5
            if not accepted:
                break
        _, final_gradient = objective(weights)
        final_gradient_norm = float(np.max(np.abs(final_gradient)))
        if not converged and final_gradient_norm > 1e-07:
            raise RuntimeError(f'offset optimization did not reach the required tolerance: gradient_inf={final_gradient_norm:.6g}')
        return (weights, scale)
    if not np.all(np.isfinite(result.x)):
        raise RuntimeError('offset optimization returned non-finite coefficients')
    return (result.x, scale)

def metric_row(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    return base.analyze(labels.tolist(), scores.tolist(), loss=0.0, threshold=None)
