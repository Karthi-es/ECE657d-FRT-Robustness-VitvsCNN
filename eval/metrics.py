"""
Face verification metrics and evaluation utilities.

Computes Equal Error Rate (EER), ROC curves, and verification accuracy.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from typing import Tuple
import matplotlib.pyplot as plt


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) and threshold at EER.
    
    Args:
        y_true: Binary labels (1 = same person, 0 = different)
        y_scores: Similarity scores or distances (higher = more similar)
    
    Returns:
        (eer, eer_threshold)
    
    EER is the point where False Accept Rate (FAR) = False Reject Rate (FRR)
    """
    fpr, fnr, thresholds = compute_fpr_fnr(y_true, y_scores)
    
    # Find threshold where FPR == FNR (EER point)
    eer_threshold = thresholds[np.argmin(np.abs(fpr - fnr))]
    eer = fpr[np.argmin(np.abs(fpr - fnr))]
    
    return eer, eer_threshold


def compute_fpr_fnr(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    num_thresholds: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute False Positive Rate (FPR) and False Negative Rate (FNR) at various thresholds.
    
    Args:
        y_true: Binary labels (1 = same, 0 = different)
        y_scores: Similarity scores (higher = more similar)
        num_thresholds: Number of threshold points to evaluate
    
    Returns:
        (fpr, fnr, thresholds)
    """
    # Create threshold range
    thresholds = np.linspace(y_scores.min(), y_scores.max(), num_thresholds)
    
    fpr = np.zeros(len(thresholds))
    fnr = np.zeros(len(thresholds))
    
    negatives = y_true == 0
    positives = y_true == 1
    
    n_negatives = np.sum(negatives)
    n_positives = np.sum(positives)
    
    for i, threshold in enumerate(thresholds):
        # False positives: different pairs predicted as same
        fp = np.sum((y_scores[negatives] >= threshold))
        fpr[i] = fp / n_negatives if n_negatives > 0 else 0
        
        # False negatives: same pairs predicted as different
        fn = np.sum((y_scores[positives] < threshold))
        fnr[i] = fn / n_positives if n_positives > 0 else 0
    
    return fpr, fnr, thresholds


def compute_accuracy_at_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float
) -> float:
    """
    Compute verification accuracy at a specific threshold.
    
    Args:
        y_true: Binary labels
        y_scores: Similarity scores
        threshold: Decision threshold
    
    Returns:
        Accuracy (0-1)
    """
    predictions = (y_scores >= threshold).astype(int)
    accuracy = np.mean(predictions == y_true)
    return accuracy


def compute_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    num_points: int = 1000
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute ROC curve and AUC.
    
    Args:
        y_true: Binary labels
        y_scores: Similarity scores
        num_points: Number of points on ROC curve
    
    Returns:
        (fpr, tpr, auc, eer)
    """
    fpr, fnr, thresholds = compute_fpr_fnr(y_true, y_scores, num_points)
    tpr = 1 - fnr  # True Positive Rate = 1 - FNR
    
    # Compute AUC using trapezoidal rule
    auc = np.trapz(tpr, fpr)
    
    # Compute EER
    eer = fpr[np.argmin(np.abs(fpr - fnr))]
    
    return fpr, tpr, auc, eer


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    output_path: str = None,
    title: str = "ROC Curve"
):
    """
    Plot ROC curve.
    
    Args:
        y_true: Binary labels
        y_scores: Similarity scores
        output_path: Path to save figure (optional)
        title: Plot title
    """
    fpr, tpr, auc, eer = compute_roc_curve(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.axvline(x=eer, color='r', linestyle='--', linewidth=1, label=f'EER = {eer:.3f}')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved ROC curve to {output_path}")
    
    plt.close()


def print_verification_results(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    name: str = "Verification Results"
):
    """
    Print comprehensive verification results.
    
    Args:
        y_true: Binary labels
        y_scores: Similarity scores
        name: Result name/identifier
    """
    eer, eer_threshold = compute_eer(y_true, y_scores)
    fpr, tpr, auc, _ = compute_roc_curve(y_true, y_scores)
    
    # Accuracy at EER
    acc_at_eer = compute_accuracy_at_threshold(y_true, y_scores, eer_threshold)
    
    # Accuracy at common working points
    acc_far_001 = compute_accuracy_at_threshold(
        y_true, y_scores,
        np.percentile(y_scores[y_true == 0], 0.1)  # FAR~0.1%
    )
    
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Equal Error Rate (EER): {eer:.4f} ({eer*100:.2f}%)")
    print(f"  Threshold at EER: {eer_threshold:.4f}")
    print(f"  Accuracy at EER: {acc_at_eer:.4f} ({acc_at_eer*100:.2f}%)")
    print(f"\nROC AUC: {auc:.4f}")
    print(f"Accuracy @ FAR~0.1%: {acc_far_001:.4f} ({acc_far_001*100:.2f}%)")
    print(f"\nMin similarity (different): {y_scores[y_true == 0].min():.4f}")
    print(f"Max similarity (different): {y_scores[y_true == 0].max():.4f}")
    print(f"Min similarity (same): {y_scores[y_true == 1].min():.4f}")
    print(f"Max similarity (same): {y_scores[y_true == 1].max():.4f}")
    print(f"{'='*60}\n")
