"""
Evaluation pipeline for face verification and robustness analysis.

Week 3: Evaluate model robustness against makeup/tattoo perturbations.
"""

from eval.metrics import compute_eer, compute_roc_curve, print_verification_results
from eval.evaluator import FaceEmbedder, VerificationEvaluator

__all__ = [
    "compute_eer",
    "compute_roc_curve",
    "print_verification_results",
    "FaceEmbedder",
    "VerificationEvaluator",
]
