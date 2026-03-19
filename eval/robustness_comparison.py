"""
Robustness comparison tool for ViT vs CNN across perturbations.

Compares model robustness by analyzing EER degradation.

Usage:
    python eval/robustness_comparison.py \
        --results-dir eval/results \
        --output-dir eval/analysis
"""

import json
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class RobustnessAnalyzer:
    """
    Analyze and compare robustness of different models across perturbations.
    """
    
    def __init__(self, results_dir: str | Path):
        """
        Load evaluation results.
        
        Args:
            results_dir: Directory containing results JSON files
        """
        self.results_dir = Path(results_dir)
        self.results = {}
        self._load_results()
    
    def _load_results(self):
        """Load all results JSON files."""
        for result_file in sorted(self.results_dir.glob("results_*.json")):
            # Parse filename: results_{perturbation}_{model}.json
            parts = result_file.stem.replace("results_", "").rsplit("_", 1)
            if len(parts) == 2:
                perturbation, model = parts
                key = (perturbation, model)
                with open(result_file) as f:
                    self.results[key] = json.load(f)
    
    def compute_robustness_metrics(self) -> Dict:
        """
        Compute robustness metrics for each model.
        
        Returns:
            Dict with robustness analysis
        """
        models = set()
        perturbations = set()
        
        for perturbation, model in self.results.keys():
            models.add(model)
            perturbations.add(perturbation)
        
        models = sorted(models)
        perturbations = sorted(perturbations)
        
        print(f"\nFound {len(models)} models: {models}")
        print(f"Found {len(perturbations)} perturbations: {perturbations}")
        
        # Build EER table
        eer_table = {}
        for model in models:
            eer_table[model] = {}
            for perturbation in perturbations:
                key = (perturbation, model)
                if key in self.results:
                    eer = self.results[key]['eer']
                    eer_table[model][perturbation] = eer
        
        # Compute EER degradation
        baseline_key = ('clean', models[0])
        if baseline_key not in self.results:
            print("❌ No baseline (clean) results found!")
            return None
        
        baseline_eer = self.results[baseline_key]['eer']
        
        analysis = {
            'baseline_eer': baseline_eer,
            'models': {},
        }
        
        for model in models:
            model_analysis = {
                'clean_eer': eer_table[model].get('clean', None),
                'perturbations': {},
            }
            
            for perturbation in perturbations:
                if perturbation == 'clean':
                    continue
                
                perturbed_eer = eer_table[model].get(perturbation, None)
                if perturbed_eer is not None:
                    # EER degradation (higher = worse robustness)
                    degradation = perturbed_eer - baseline_eer
                    degradation_pct = (degradation / baseline_eer * 100) if baseline_eer > 0 else 0
                    
                    model_analysis['perturbations'][perturbation] = {
                        'eer': perturbed_eer,
                        'degradation': degradation,
                        'degradation_pct': degradation_pct,
                    }
            
            analysis['models'][model] = model_analysis
        
        return analysis, eer_table, models, perturbations
    
    def plot_robustness(self, analysis: Dict, eer_table: Dict, models: list, perturbations: list):
        """Plot robustness comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: EER across perturbations
        ax = axes[0]
        for model in models:
            eers = [eer_table[model].get(p, None) for p in perturbations]
            ax.plot(perturbations, eers, marker='o', label=model, linewidth=2)
        
        ax.set_xlabel('Perturbation Type')
        ax.set_ylabel('Equal Error Rate (EER)')
        ax.set_title('Model Robustness to Perturbations')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 2: EER degradation from baseline
        ax = axes[1]
        perturbations_no_clean = [p for p in perturbations if p != 'clean']
        
        x = np.arange(len(perturbations_no_clean))
        width = 0.35
        
        for i, model in enumerate(models):
            degradations = []
            for perturbation in perturbations_no_clean:
                pert_eer = eer_table[model].get(perturbation, None)
                baseline = eer_table[model].get('clean', None)
                if pert_eer is not None and baseline is not None:
                    deg = (pert_eer - baseline) / baseline * 100
                    degradations.append(deg)
                else:
                    degradations.append(0)
            
            ax.bar(x + i*width, degradations, width, label=model)
        
        ax.set_xlabel('Perturbation Type')
        ax.set_ylabel('EER Degradation (%)')
        ax.set_title('Robustness Comparison: EER Degradation from Baseline')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(perturbations_no_clean, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def print_summary(self, analysis: Dict):
        """Print robustness summary."""
        print(f"\n{'='*80}")
        print(f"ROBUSTNESS ANALYSIS SUMMARY")
        print(f"{'='*80}")
        print(f"\nBaseline EER (clean): {analysis['baseline_eer']:.4f}")
        
        for model, model_data in analysis['models'].items():
            print(f"\n{model.upper()}")
            print(f"  Clean EER: {model_data['clean_eer']:.4f}")
            print(f"  Robustness (max degradation):")
            
            if model_data['perturbations']:
                max_deg = max(d['degradation'] for d in model_data['perturbations'].values())
                worst_pert = max(
                    model_data['perturbations'].items(),
                    key=lambda x: x[1]['degradation']
                )
                print(f"    {worst_pert[0]}: +{worst_pert[1]['degradation']:.4f} ({worst_pert[1]['degradation_pct']:.2f}%)")
            
            print(f"  All perturbations:")
            for pert, pert_data in sorted(model_data['perturbations'].items()):
                print(f"    {pert}: EER={pert_data['eer']:.4f}, degradation={pert_data['degradation_pct']:+.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Analyze robustness across models and perturbations")
    parser.add_argument("--results-dir", type=str, default="eval/results",
                        help="Directory with evaluation results")
    parser.add_argument("--output-dir", type=str, default="eval/analysis",
                        help="Output directory for analysis")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = RobustnessAnalyzer(args.results_dir)
    
    # Compute metrics
    result = analyzer.compute_robustness_metrics()
    if result is None:
        return
    
    analysis, eer_table, models, perturbations = result
    
    # Print summary
    analyzer.print_summary(analysis)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot and save
    fig = analyzer.plot_robustness(analysis, eer_table, models, perturbations)
    fig.savefig(output_dir / "robustness_comparison.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved robustness plot to {output_dir}/robustness_comparison.png")
    
    # Save analysis to JSON
    analysis_file = output_dir / "robustness_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"✓ Saved analysis to {analysis_file}")


if __name__ == "__main__":
    main()
