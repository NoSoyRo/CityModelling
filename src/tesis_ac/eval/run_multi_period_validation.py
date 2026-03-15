"""
Runner para validación multi-período sobre todo el histórico disponible.

Uso:
  python -m tesis_ac.eval.run_multi_period_validation \
    --data-dir data/processed/standardized_maps \
    --n-steps 10 --replicas 10 --seed 123 \
    --output-json reports/validation_all_periods.json \
    --output-csv reports/validation_all_periods.csv

Recorre los pares consecutivos de años disponibles (year_YYYY con svm_2_classes.npy)
y ejecuta `run_validation_period` por cada intervalo t0→t1, guardando agregados
y métricas por réplica en CSV.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np

from .run_period_validation import run_validation_period
from tesis_ac.historical.extract_transitions import load_historical_maps


def get_available_year_pairs(data_dir: Path) -> List[Tuple[int, int]]:
    """Obtiene pares consecutivos de años disponibles (con misma shape)."""
    years_data = load_historical_maps(data_dir, file_pattern="svm_2_classes.npy")
    sorted_years = sorted(years_data.keys())
    pairs = []
    for i in range(len(sorted_years) - 1):
        y0, y1 = sorted_years[i], sorted_years[i + 1]
        # Solo incluir si shapes coinciden
        if years_data[y0].shape == years_data[y1].shape:
            pairs.append((y0, y1))
    return pairs


def run_all_periods(
    data_dir: Path,
    n_steps: int,
    replicas: int,
    seed: int,
    output_json: Path | None,
    output_csv: Path | None,
) -> Dict[str, Any]:
    """Ejecuta validación para todos los pares consecutivos t0→t1 disponibles.

    Argumentos:
        data_dir: Directorio con mapas históricos.
        n_steps: Pasos de simulación por período.
        replicas: Réplicas por período.
        seed: Semilla base.
        output_json: Ruta opcional para JSON consolidado.
        output_csv: Ruta opcional para CSV agregada por período.

    Retorna:
        Diccionario con resumen (lista de períodos y resultados completos).
    """
    pairs = get_available_year_pairs(data_dir)
    results = []

    # CSV agregado de todos los períodos (promedios y std por período)
    csv_rows: List[Dict[str, Any]] = []

    for (y0, y1) in pairs:
        res = run_validation_period(
            data_dir=data_dir,
            initial_year=y0,
            target_year=y1,
            n_steps=n_steps,
            replicas=replicas,
            seed=seed,
        )
        results.append(res)

        # Fila con agregados por período
        agg = res['aggregates']
        csv_rows.append({
            'period': f"{y0}->{y1}",
            'accuracy_mean': agg['accuracy']['mean'],
            'accuracy_std': agg['accuracy']['std'],
            'kappa_mean': agg['kappa']['mean'],
            'kappa_std': agg['kappa']['std'],
            'iou_mean': agg['iou']['mean'],
            'iou_std': agg['iou']['std'],
            'fom_mean': agg['fom']['mean'],
            'fom_std': agg['fom']['std'],
            'fuzzy_mean': agg['fuzzy_overall']['mean'],
            'fuzzy_std': agg['fuzzy_overall']['std'],
            'quantity_mean': agg['quantity_disagreement']['mean'],
            'quantity_std': agg['quantity_disagreement']['std'],
            'allocation_mean': agg['allocation_disagreement']['mean'],
            'allocation_std': agg['allocation_disagreement']['std'],
            'overall_disagreement_mean': agg['overall_disagreement']['mean'],
            'overall_disagreement_std': agg['overall_disagreement']['std'],
            'agreement_mean': agg['agreement']['mean'],
            'agreement_std': agg['agreement']['std'],
        })

    summary = {
        'data_dir': str(data_dir),
        'n_steps': n_steps,
        'replicas': replicas,
        'seed': seed,
        'periods': [f"{y0}->{y1}" for (y0, y1) in pairs],
        'results': results,
    }

    if output_json:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(summary, f, indent=2)

    if output_csv:
        import csv
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(csv_rows[0].keys()) if csv_rows else ['period']
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_rows:
                writer.writerow(row)

    return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Validación multi-período t0→t1 sobre histórico disponible"
    )
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='Directorio con subcarpetas year_YYYY y archivos svm_2_classes.npy')
    parser.add_argument('--n-steps', type=int, default=10,
                        help='Pasos de simulación del AC por período')
    parser.add_argument('--replicas', type=int, default=5,
                        help='Réplicas por período (ensamble)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Semilla base')
    parser.add_argument('--output-json', type=Path, default=Path('reports/validation_all_periods.json'),
                        help='Salida JSON con resultados completos')
    parser.add_argument('--output-csv', type=Path, default=Path('reports/validation_all_periods.csv'),
                        help='Salida CSV agregada por período')

    args = parser.parse_args()

    summary = run_all_periods(
        data_dir=args.data_dir,
        n_steps=args.n_steps,
        replicas=args.replicas,
        seed=args.seed,
        output_json=args.output_json,
        output_csv=args.output_csv,
    )

    print(json.dumps({
        'periods': summary['periods'],
        'data_dir': summary['data_dir'],
        'n_steps': summary['n_steps'],
        'replicas': summary['replicas'],
    }, indent=2))


if __name__ == '__main__':
    main()
