from __future__ import annotations
import argparse, json
from pathlib import Path
import hashlib

import matplotlib.pyplot as plt
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True)
    parser.add_argument('--figures', required=True)
    parser.add_argument('--artifacts', required=True)
    args = parser.parse_args()
    root = Path(args.results)
    figures = Path(args.figures)
    artifacts = Path(args.artifacts)
    figures.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(root / 'energy_candidates_all.csv')
    families = pd.read_csv(root / 'energy_family_results.csv')
    decisions = pd.read_csv(root / 'energy_deployment_decisions.csv')

    summary = (
        families.groupby('family', as_index=False)
        .agg(
            zones=('zone', 'count'),
            mean_certificate=('certificate', 'mean'),
            mean_validation_rmse=('validation_rmse', 'mean'),
            mean_test_rmse=('test_rmse_clipped', 'mean'),
            mean_test_weighted_cost=('test_weighted_cost', 'mean'),
            mean_total_kl=('total_kl', 'mean'),
            mean_rule_count=('rule_count', 'mean'),
            mean_dimension=('consequent_dimension', 'mean'),
            mean_certification_clipping=('certification_target_clipping_rate', 'mean'),
        )
    )
    summary.to_csv(root / 'energy_summary_by_family.csv', index=False)

    fixed = families[families.family == 'fixed_k_dense_tsk'].set_index('zone')
    radius = families[families.family == 'dense_tsk'].set_index('zone')
    structural = pd.DataFrame({
        'zone': fixed.index,
        'fixed_k_rule_count': fixed.rule_count,
        'radius_rule_count': radius.rule_count,
        'rule_reduction': fixed.rule_count - radius.rule_count,
        'fixed_k_certificate': fixed.certificate,
        'radius_certificate': radius.certificate,
        'certificate_reduction': fixed.certificate - radius.certificate,
        'fixed_k_test_rmse': fixed.test_rmse_clipped,
        'radius_test_rmse': radius.test_rmse_clipped,
        'test_rmse_change_radius_minus_fixed': radius.test_rmse_clipped - fixed.test_rmse_clipped,
    }).reset_index(drop=True)
    structural.to_csv(root / 'energy_structural_comparison.csv', index=False)

    for zone in sorted(candidates.zone.unique()):
        subset = candidates[candidates.zone == zone]
        fig, ax = plt.subplots(figsize=(7, 5))
        for family, group in subset.groupby('family'):
            ax.scatter(group.validation_rmse, group.certificate, label=family, alpha=0.75)
        ax.axhline(0.20, linestyle='--', linewidth=1, label='certificate threshold')
        ax.set_xlabel('Validation RMSE (source units)')
        ax.set_ylabel('PAC-Bayes certificate')
        ax.set_title(f'Tetouan {zone}: validation fit versus certificate')
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / f'{zone}_validation_certificate.png', dpi=180)
        plt.close(fig)

        pred = pd.read_csv(root / 'predictions' / f'{zone}_test_predictions.csv', parse_dates=['timestamp'])
        tail = pred.tail(84)
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.plot(tail.timestamp, tail.actual, label='actual')
        ax.plot(tail.timestamp, tail.seasonal_naive_24h, label='seasonal naive (24h)')
        ax.plot(tail.timestamp, tail.deployed_forecast, label='predeclared deployed forecast')
        ax.set_xlabel('Time')
        ax.set_ylabel('Power-consumption units')
        ax.set_title(f'Tetouan {zone}: final seven days of the test segment')
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(figures / f'{zone}_test_last_seven_days.png', dpi=180)
        plt.close(fig)

    outputs = sorted(
        [p for p in root.rglob('*') if p.is_file()]
        + [p for p in figures.rglob('*') if p.is_file()]
    )
    manifest = {
        'stage': 'energy_case_study_v3',
        'postprocessing_only': True,
        'outcome_files_unchanged': True,
        'files': {str(path): sha256(path) for path in outputs},
        'deployed_zones': int((decisions.deployment != 'seasonal_naive_24h').sum()),
        'mean_test_rmse_improvement_vs_fallback': float(decisions.test_rmse_improvement_vs_fallback.mean()),
        'mean_test_cost_improvement_vs_fallback': float(decisions.test_cost_improvement_vs_fallback.mean()),
    }
    (artifacts / 'energy_case_v3_manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8'
    )


if __name__ == '__main__':
    main()
