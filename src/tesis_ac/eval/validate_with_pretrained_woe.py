"""
Validar períodos usando WoE pre-entrenado (más rápido).

Uso:
  python -m tesis_ac.eval.validate_with_pretrained_woe \
    --woe-file reports/woe_trained_1984_2014.pkl \
    --data-dir data/processed/standardized_maps \
    --years 2016 2017 2018 2019 2020 \
    --n-steps 3 --replicas 3
"""

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from tesis_ac.historical.extract_transitions import load_historical_maps
from tesis_ac.ca.rules import WoECellularAutomaton, ACParameters, create_default_spatial_variables
from tesis_ac.eval.metrics import calculate_spatial_metrics, calculate_figure_of_merit
from tesis_ac.eval.advanced_metrics import (
    calculate_multiple_resolution_validation,
    calculate_quantity_vs_allocation_disagreement,
)


def validate_period_with_woe(
    woe_calc,
    woe_results: Dict,
    grid_t0: np.ndarray,
    grid_t1: np.ndarray,
    n_steps: int = 3,
    replicas: int = 3,
    seed: int = 123,
    window_sizes: List[int] = None,
) -> Dict[str, Any]:
    """Valida un período usando WoE pre-entrenado (múltiples réplicas).

    Argumentos:
        woe_calc: Instancia de WoECalculator ya entrenada.
        woe_results: Dict con resultados WoE por variable.
        grid_t0: Grid inicial.
        grid_t1: Grid objetivo (ground truth).
        n_steps: Pasos de simulación del AC.
        replicas: Número de réplicas (semillas diferentes).
        seed: Semilla base.
        window_sizes: Ventanas para validación fuzzy (multi-resolución).

    Retorna:
        Diccionario con métricas por réplica y agregadas (media/std).
    """
    window_sizes = window_sizes or [3, 5, 9, 15, 25]
    grid_shape = grid_t0.shape
    
    # Variables espaciales del período
    spatial_vars = create_default_spatial_variables(grid_shape, existing_urban=grid_t0)
    
    # Resultados por réplica
    replicas_results = []
    csv_rows = []
    
    for r in range(replicas):
        np.random.seed(seed + r)
        
        # Crear AC con WoE pre-entrenado
        ac = WoECellularAutomaton(grid_shape, woe_calc, parameters=ACParameters())
        ac.set_initial_state(grid_t0)
        
        # Agregar variables con WoE pre-calculado
        for name, var in spatial_vars.items():
            if name in woe_results:
                ac.add_spatial_variable(name, var, compute_woe=True)
        
        # Simular
        grids = [grid_t0]
        stats_list = []
        for _ in range(n_steps):
            g, stats = ac.step()
            grids.append(g)
            stats_list.append(stats)
            if stats.get('new_urban_cells', 1) == 0:
                break
        
        grid_pred = grids[-1]
        
        # Métricas
        spatial = calculate_spatial_metrics(grid_pred, grid_t1)
        spatial['fom'] = float(calculate_figure_of_merit(grid_pred, grid_t1, initial_grid=grid_t0))
        fuzzy = calculate_multiple_resolution_validation(grid_pred, grid_t1, window_sizes=window_sizes)
        pontius = calculate_quantity_vs_allocation_disagreement(grid_pred, grid_t1)
        
        replica_result = {
            'replica': r,
            'simulation': {
                'steps_executed': len(stats_list),
                'final_stats': stats_list[-1] if stats_list else {},
            },
            'metrics': {
                'spatial': spatial,
                'fuzzy_multi_resolution': fuzzy,
                'quantity_allocation': pontius,
            }
        }
        replicas_results.append(replica_result)
        
        # CSV row
        csv_rows.append({
            'replica': r,
            'accuracy': spatial.get('accuracy'),
            'kappa': spatial.get('kappa'),
            'iou': spatial.get('iou'),
            'fom': spatial.get('fom'),
            'fuzzy_overall': fuzzy.get('overall_fuzzy_kappa'),
            'quantity_disagreement': pontius.get('quantity_disagreement'),
            'allocation_disagreement': pontius.get('allocation_disagreement'),
            'overall_disagreement': pontius.get('overall_disagreement'),
            'agreement': pontius.get('agreement'),
        })
    
    # Agregar stats
    def agg(key: str) -> Dict[str, float]:
        vals = [row[key] for row in csv_rows if row[key] is not None]
        if not vals:
            return {'mean': None, 'std': None}
        arr = np.array(vals, dtype=float)
        return {'mean': float(np.mean(arr)), 'std': float(np.std(arr))}
    
    aggregates = {
        'accuracy': agg('accuracy'),
        'kappa': agg('kappa'),
        'iou': agg('iou'),
        'fom': agg('fom'),
        'fuzzy_overall': agg('fuzzy_overall'),
        'quantity_disagreement': agg('quantity_disagreement'),
        'allocation_disagreement': agg('allocation_disagreement'),
        'overall_disagreement': agg('overall_disagreement'),
        'agreement': agg('agreement'),
    }
    
    return {
        'replicas': replicas_results,
        'aggregates': aggregates,
        'csv_rows': csv_rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Validar con WoE pre-entrenado")
    parser.add_argument('--woe-file', type=Path, required=True,
                        help='Archivo .pkl con WoE entrenado')
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='Directorio con grids por año')
    parser.add_argument('--years', type=int, nargs='+', required=True,
                        help='Años a validar (ej. 2016 2017 2018 2019 2020)')
    parser.add_argument('--n-steps', type=int, default=3,
                        help='Pasos de simulación AC')
    parser.add_argument('--replicas', type=int, default=3,
                        help='Número de réplicas por período')
    parser.add_argument('--seed', type=int, default=123,
                        help='Semilla base')
    parser.add_argument('--invert-classes', action='store_true',
                        help='Invertir 0/1')
    parser.add_argument('--output-dir', type=Path, default=Path('reports'),
                        help='Directorio para JSONs y CSVs')
    
    args = parser.parse_args()
    
    # Cargar WoE entrenado
    print(f"Cargando WoE desde {args.woe_file}...")
    with open(args.woe_file, 'rb') as f:
        woe_data = pickle.load(f)
    
    woe_calc = woe_data['woe_calculator']
    woe_results = woe_data['woe_results']
    
    print(f"  Entrenado con {woe_data['train_pairs']} pares ({woe_data['train_years'][0]}-{woe_data['train_years'][1]})")
    print(f"  Variables: {woe_data['variables']}")
    print(f"  Tasa de cambio histórica: {100*woe_data['change_rate']:.2f}%")
    
    # Cargar datos
    print(f"\nCargando grids desde {args.data_dir}...")
    years_data = load_historical_maps(args.data_dir, file_pattern="svm_2_classes.npy")
    
    # Validar cada par consecutivo
    years_sorted = sorted(args.years)
    all_results = []
    all_csv_rows = []
    
    for y0, y1 in zip(years_sorted[:-1], years_sorted[1:]):
        print(f"\n{'='*60}")
        print(f"Validando {y0}→{y1}")
        print('='*60)
        
        grid_t0 = years_data[y0]
        grid_t1 = years_data[y1]
        
        if args.invert_classes:
            grid_t0 = 1 - grid_t0
            grid_t1 = 1 - grid_t1
        
        result = validate_period_with_woe(
            woe_calc=woe_calc,
            woe_results=woe_results,
            grid_t0=grid_t0,
            grid_t1=grid_t1,
            n_steps=args.n_steps,
            replicas=args.replicas,
            seed=args.seed,
        )
        
        # Agregar período info
        result['period'] = {
            'initial_year': y0,
            'target_year': y1,
            'time_span_years': y1 - y0,
        }
        result['grid_shape'] = grid_t0.shape
        result['woe_training'] = woe_data['train_years']
        
        all_results.append(result)
        
        # Añadir años a CSV rows
        for row in result['csv_rows']:
            row.update({'initial_year': y0, 'target_year': y1})
            all_csv_rows.append(row)
        
        # Mostrar resumen
        agg = result['aggregates']
        print(f"  Accuracy: {agg['accuracy']['mean']:.4f} ± {agg['accuracy']['std']:.4f}")
        print(f"  Kappa:    {agg['kappa']['mean']:.4f} ± {agg['kappa']['std']:.4f}")
        print(f"  IoU:      {agg['iou']['mean']:.4f} ± {agg['iou']['std']:.4f}")
        print(f"  FoM:      {agg['fom']['mean']:.4f} ± {agg['fom']['std']:.4f}")
        
        # Guardar JSON individual
        output_json = args.output_dir / f"validation_{y0}_{y1}_hist.json"
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  ✓ Guardado: {output_json}")
    
    # Guardar CSV consolidado
    output_csv = args.output_dir / "validation_hist_all_years.csv"
    if all_csv_rows:
        fieldnames = list(all_csv_rows[0].keys())
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_csv_rows)
        print(f"\n✓ CSV consolidado: {output_csv}")
    
    # Resumen final
    print(f"\n{'='*60}")
    print("RESUMEN FINAL")
    print('='*60)
    
    df = pd.DataFrame(all_csv_rows)
    for metric in ['accuracy', 'kappa', 'iou', 'fom']:
        grouped = df.groupby(['initial_year', 'target_year'])[metric].agg(['mean', 'std'])
        print(f"\n{metric.upper()}:")
        for (y0, y1), row in grouped.iterrows():
            print(f"  {y0}→{y1}: {row['mean']:.4f} ± {row['std']:.4f}")


if __name__ == '__main__':
    main()
