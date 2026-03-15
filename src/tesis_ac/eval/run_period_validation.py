"""
Runner de validación por período histórico para AC-WoE.

Uso:
  python -m tesis_ac.eval.run_period_validation \
    --data-dir data/processed/batch_processing_YYYYMMDD_HHMMSS \
    --initial-year 1990 --target-year 1995 \
    --n-steps 10 --output reports/validation_1990_1995.json

Este script:
  1) Carga grids históricos binarios (0=vegetación, 1=urbano)
  2) Construye variables espaciales por defecto
  3) Calcula WoE y simula el AC hasta n-steps
  4) Evalúa métricas: FoM, fuzzy multi-escala, quantity/allocation, FRAGSTATS, patrones de crecimiento
  5) Guarda resultados en JSON
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

from tesis_ac.historical.extract_transitions import load_historical_maps
from tesis_ac.ca.rules import WoECellularAutomaton, ACParameters, create_default_spatial_variables
from tesis_ac.woe.woe import WoECalculator, calculate_spatial_woe_variables
from tesis_ac.eval.metrics import (
    calculate_spatial_metrics,
    calculate_figure_of_merit,
)
from tesis_ac.eval.advanced_metrics import (
    calculate_multiple_resolution_validation,
    calculate_quantity_vs_allocation_disagreement,
    calculate_landscape_metrics,
    calculate_growth_pattern_similarity,
)


def run_validation_period(
    data_dir: Path,
    initial_year: int,
    target_year: int,
    n_steps: int = 10,
    window_sizes: List[int] | None = None,
    output: Path | None = None,
    replicas: int = 1,
    seed: int = 42,
    csv_out: Path | None = None,
    invert_classes: bool = False,
    woe_train_pooled: bool = False,
    woe_train_years: tuple[int, int] | None = None,
) -> Dict[str, Any]:
    """Ejecuta validación para un período histórico t0→t1.

    Argumentos:
        data_dir: Directorio con subcarpetas por año y archivos de grid
            (por ejemplo ``svm_2_classes.npy``).
        initial_year: Año inicial (t0).
        target_year: Año objetivo (t1).
        n_steps: Número de pasos de simulación del AC.
        window_sizes: Ventanas para validación multi-resolución (fuzzy).
        output: Ruta opcional de salida JSON.
        replicas: Número de réplicas (ensamble).
        seed: Semilla base.
        csv_out: Ruta opcional para CSV (por réplica).
        invert_classes: Si invierte 0/1 de entrada para trabajar con 1=urbano.
        woe_train_pooled: Si entrena WoE con pooling de múltiples períodos.
        woe_train_years: Rango opcional de años para entrenar WoE pooled.

    Retorna:
        Diccionario con métricas (por réplica y agregadas) y metadatos del
        período.
    """
    window_sizes = window_sizes or [3, 5, 9, 15, 25]

    # 1) Cargar grids históricos
    print(f'\n  📂 Cargando mapas históricos desde: {data_dir}')
    years_data = load_historical_maps(data_dir, file_pattern="svm_2_classes.npy")
    print(f'  ✓ Mapas cargados: {len(years_data)} años disponibles')
    
    if initial_year not in years_data or target_year not in years_data:
        raise ValueError(f"Años no disponibles en {data_dir}: {initial_year}, {target_year}")

    grid_t0 = years_data[initial_year]
    grid_t1 = years_data[target_year]
    print(f'  ✓ Grid t0 ({initial_year}): {grid_t0.shape}, {100*(grid_t0==1).sum()/grid_t0.size:.1f}% urbano')
    print(f'  ✓ Grid t1 ({target_year}): {grid_t1.shape}, {100*(grid_t1==1).sum()/grid_t1.size:.1f}% urbano')
    # Si la data viene con 0=urbano y 1=no urbano, invierte para usar 1=urbano
    if invert_classes:
        grid_t0 = 1 - grid_t0
        grid_t1 = 1 - grid_t1
    grid_shape = grid_t0.shape
    if grid_t1.shape != grid_shape:
        raise ValueError(f"Shapes diferentes entre t0 y t1: {grid_shape} vs {grid_t1.shape}")

    # 2) Variables espaciales por defecto (solo objetivas)
    print(f'\n  🗺️  Creando variables espaciales...')
    spatial_vars = create_default_spatial_variables(
        grid_shape,
        existing_urban=grid_t0,
    )
    print(f'  ✓ Variables creadas: {list(spatial_vars.keys())}')

    # 3) Calcular WoE
    print(f'\n  🧮 Calculando WoE...')
    woe_calc = WoECalculator(min_bin_size=100, max_bins=8)
    if woe_train_pooled or woe_train_years:
        # Entrenamiento WoE con pooling de pares de años especificados o todos disponibles
        print(f'  → Modo: WoE pooled (entrenamiento con múltiples períodos)')
        all_years = sorted(years_data.keys())
        
        # Determinar rango de años para entrenamiento
        if woe_train_years:
            train_start, train_end = woe_train_years
            train_years = [y for y in all_years if train_start <= y <= train_end]
            print(f'  → Rango de entrenamiento: {train_start}-{train_end} ({len(train_years)-1} períodos)')
        else:
            train_years = all_years
            print(f'  → Usando todos los años disponibles ({len(train_years)-1} períodos)')
        
        pooled_vars: Dict[str, List[np.ndarray]] = {}
        pooled_targets: List[np.ndarray] = []

        for idx, (y0, y1) in enumerate(zip(train_years[:-1], train_years[1:]), 1):
            if y1 not in years_data:
                continue
            print(f'    • Período {idx}/{len(train_years)-1}: {y0}→{y1}', end=' ')
            g0 = years_data[y0]
            g1 = years_data[y1]
            if invert_classes:
                g0 = 1 - g0
                g1 = 1 - g1

            # features objetivas por periodo (derivadas de t0)
            fvars = create_default_spatial_variables(g0.shape, existing_urban=g0)
            # máscara de cambio 0→1
            uh = ((g0 == 0) & (g1 == 1)).astype(int).flatten()
            pooled_targets.append(uh)

            # acumular variables
            for name, var in fvars.items():
                pooled_vars.setdefault(name, []).append(var.flatten())
            
            print(f'✓')

        print(f'  → Calculando WoE en datos agregados...')
        target_concat = np.concatenate(pooled_targets) if pooled_targets else None
        woe_results = {}
        for name, parts in pooled_vars.items():
            var_concat = np.concatenate(parts)
            # calcular WOE global por variable usando datos concatenados
            res = woe_calc.calculate_woe(var_concat, target_concat, variable_name=name, binning_method='quantile')
            woe_results[name] = res
        print(f'  ✓ WoE calculado para {len(woe_results)} variables')
    else:
        # Entrenamiento WoE por periodo (t0→t1)
        print(f'  → Modo: WoE por período (solo {initial_year}→{target_year})')
        urban_history = (grid_t0 == 0) & (grid_t1 == 1)
        changes = urban_history.sum()
        print(f'  → Cambios detectados (0→1): {changes} píxeles ({100*changes/grid_t0.size:.2f}%)')
        woe_results = calculate_spatial_woe_variables(
            grid=grid_t0,
            urban_history=urban_history.astype(int),
            features=spatial_vars,
            calculator=woe_calc,
        )
        print(f'  ✓ WoE calculado para {len(woe_results)} variables')

    # 4) Réplicas de simulación AC con WoE
    print(f'\n  🔄 Ejecutando {replicas} réplicas de simulación AC...')
    replicas_results: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []

    for r in range(replicas):
        print(f'    Réplica {r+1}/{replicas}:', end=' ')
        np.random.seed(seed + r)

        ac = WoECellularAutomaton(grid_shape, woe_calc, parameters=ACParameters())
        ac.set_initial_state(grid_t0)
        for name, var in spatial_vars.items():
            ac.add_spatial_variable(name, var, compute_woe=True)

        grids = [grid_t0]
        stats_list = []
        print(f'steps [', end='')
        for step in range(n_steps):
            g, stats = ac.step()
            grids.append(g)
            stats_list.append(stats)
            print(f'{step+1}', end='')
            if stats.get('new_urban_cells', 1) == 0:
                print(f'→stopped', end='')
                break
            if (step + 1) < n_steps:
                print(f',', end='')
        print(f']', end=' ')

        grid_pred = grids[-1]

        # Métricas por réplica
        print(f'→ calculando métricas...', end=' ')
        spatial = calculate_spatial_metrics(grid_pred, grid_t1)
        spatial['fom'] = float(
            calculate_figure_of_merit(grid_pred, grid_t1, initial_grid=grid_t0)
        )
        fuzzy = calculate_multiple_resolution_validation(grid_pred, grid_t1, window_sizes=window_sizes)
        pontius = calculate_quantity_vs_allocation_disagreement(grid_pred, grid_t1)
        landscape_pred = calculate_landscape_metrics(grid_pred)
        landscape_obs = calculate_landscape_metrics(grid_t1)
        patterns = calculate_growth_pattern_similarity(grid_pred, grid_t1, initial_grid=grid_t0)
        print(f'✓ (FoM={spatial["fom"]:.3f}, Acc={spatial["accuracy"]:.3f})')

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
                'landscape_pred': landscape_pred,
                'landscape_obs': landscape_obs,
                'growth_patterns': patterns,
            }
        }
        replicas_results.append(replica_result)

        # Fila CSV por réplica (resumen clave)
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

    # Agregados (media ± std) sobre réplicas
    print(f'\n  📊 Agregando resultados de {replicas} réplicas...')
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
    
    print(f'  ✓ Agregados calculados')
    print(f'    • FoM:      {aggregates["fom"]["mean"]:.4f} ± {aggregates["fom"]["std"]:.4f}')
    print(f'    • Accuracy: {aggregates["accuracy"]["mean"]:.4f} ± {aggregates["accuracy"]["std"]:.4f}')
    print(f'    • Kappa:    {aggregates["kappa"]["mean"]:.4f} ± {aggregates["kappa"]["std"]:.4f}')

    result = {
        'period': {
            'initial_year': initial_year,
            'target_year': target_year,
            'time_span_years': int(target_year - initial_year),
        },
        'grid_shape': grid_shape,
        'replicas': replicas_results,
        'aggregates': aggregates,
        'woe': {
            'variables': list(woe_results.keys()),
            'summary': woe_calc.summary_report(),
        }
    }

    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'\n  💾 Resultados guardados en: {output}')

    # CSV opcional
    if csv_out:
        import csv
        csv_out = Path(csv_out)
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(csv_rows[0].keys()) if csv_rows else ['replica']
        with open(csv_out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_rows:
                writer.writerow(row)
        print(f'  💾 CSV guardado en: {csv_out}')

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Validación por período histórico t0→t1 para AC-WoE"
    )
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='Directorio con subcarpetas por año y archivos svm_2_classes.npy')
    parser.add_argument('--initial-year', type=int, required=True,
                        help='Año inicial t0')
    parser.add_argument('--target-year', type=int, required=True,
                        help='Año objetivo t1')
    parser.add_argument('--n-steps', type=int, default=10,
                        help='Número de pasos de simulación del AC')
    parser.add_argument('--output', type=Path, default=Path('reports/validation_period.json'),
                        help='Archivo JSON de salida')
    parser.add_argument('--window-sizes', type=int, nargs='*', default=[3, 5, 9, 15, 25],
                        help='Ventanas para validación multi-resolución')
    parser.add_argument('--replicas', type=int, default=1,
                        help='Número de réplicas (ensamble)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Semilla base para réplicas')
    parser.add_argument('--csv-out', type=Path, default=None,
                        help='Ruta CSV para resumen por réplica')
    parser.add_argument('--invert-classes', action='store_true',
                        help='Invierte 0/1 en t0 y t1 (si la data tiene 0=urbano)')
    parser.add_argument('--woe-train-pooled', action='store_true',
                        help='Entrena WOE agrupando (pooling) todos los pares consecutivos de años disponibles en data_dir')
    parser.add_argument('--woe-train-years', type=int, nargs=2, metavar=('START', 'END'), default=None,
                        help='Rango de años (START END) para entrenar WOE (ej. 1984 2015). Si se omite, usa solo initial_year→target_year')

    args = parser.parse_args()

    result = run_validation_period(
        data_dir=args.data_dir,
        initial_year=args.initial_year,
        target_year=args.target_year,
        n_steps=args.n_steps,
        window_sizes=args.window_sizes,
        output=args.output,
        replicas=args.replicas,
        seed=args.seed,
        csv_out=args.csv_out,
        invert_classes=args.invert_classes,
        woe_train_pooled=args.woe_train_pooled,
        woe_train_years=tuple(args.woe_train_years) if args.woe_train_years else None,
    )

    # Mostrar resumen compacto en consola
    print(json.dumps({
        'period': result['period'],
        'aggregates': result['aggregates'],
    }, indent=2))


if __name__ == '__main__':
    main()
