"""
Optimización automática de parámetros AC para minimizar FP y FN.

Usa grid search para encontrar la mejor combinación de:
- growth_rate: tasa de crecimiento por paso
- w_neighbors: peso de influencia del vecindario
- w_lbp: peso de patrones locales
- w_distance_urban: penalización por distancia

Objetivo: Minimizar FP (sobre-predicción) y FN (sub-predicción)
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any
import itertools
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from tesis_ac.historical.extract_transitions import load_historical_maps
from tesis_ac.ca.rules import WoECellularAutomaton, ACParameters, create_default_spatial_variables
from tesis_ac.eval.metrics import calculate_spatial_metrics


def evaluate_parameters(
    grid_t0: np.ndarray,
    grid_t1: np.ndarray,
    woe_calc,
    woe_results: Dict,
    params: ACParameters,
    growth_rate: float,
    n_steps: int = 3,
    seed: int = 123,
    verbose: bool = True
) -> Dict[str, float]:
    """Evalúa un conjunto de parámetros del AC y retorna métricas de ajuste.

    Argumentos:
        grid_t0: Grid inicial.
        grid_t1: Grid objetivo (observado).
        woe_calc: WoECalculator pre-entrenado.
        woe_results: Resultados WoE por variable.
        params: Parámetros del AC.
        growth_rate: Tasa de crecimiento por paso.
        n_steps: Número de pasos a simular.
        seed: Semilla aleatoria.
        verbose: Si imprime progreso.

    Retorna:
        Diccionario con métricas (accuracy, kappa, iou, precision, recall, F1)
        y tasas FP/FN/TP/TN.
    """
    np.random.seed(seed)
    
    grid_shape = grid_t0.shape
    spatial_vars = create_default_spatial_variables(grid_shape, existing_urban=grid_t0)
    
    # Crear AC con parámetros específicos
    ac = WoECellularAutomaton(grid_shape, woe_calc, parameters=params)
    ac.set_initial_state(grid_t0)
    ac.growth_rate = growth_rate
    
    # Agregar variables
    for name, var in spatial_vars.items():
        if name in woe_results:
            ac.add_spatial_variable(name, var, compute_woe=True)
    
    # Simular
    if verbose:
        print(f"    Simulando {n_steps} pasos...", end=" ", flush=True)
    
    for step in range(n_steps):
        grid_pred, stats = ac.step()
        if verbose:
            print(f"paso {step+1}({stats.get('new_urban_cells', 0)} nuevas)", end=" ", flush=True)
        if stats.get('new_urban_cells', 1) == 0:
            break
    
    if verbose:
        print("✓")
    
    # Invertir clases para comparación
    grid_pred = 1 - grid_pred
    
    # Calcular métricas detalladas
    if verbose:
        print("    Calculando métricas...", end=" ", flush=True)
    
    metrics = calculate_spatial_metrics(grid_pred, grid_t1)
    
    # Calcular FP, FN, TP, TN
    tn = np.sum((grid_t1 == 0) & (grid_pred == 0))
    fp = np.sum((grid_t1 == 0) & (grid_pred == 1))
    fn = np.sum((grid_t1 == 1) & (grid_pred == 0))
    tp = np.sum((grid_t1 == 1) & (grid_pred == 1))
    total = tn + fp + fn + tp
    
    if verbose:
        print("✓")
    
    return {
        'accuracy': metrics['accuracy'],
        'kappa': metrics['kappa'],
        'iou': metrics['iou'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1_score'],
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'fp_rate': fp / total,
        'fn_rate': fn / total,
        'tp_rate': tp / total,
        'tn_rate': tn / total
    }


def optimize_parameters(
    data_dir: Path,
    woe_file: Path,
    year_start: int,
    year_end: int,
    growth_rates: List[float] = None,
    w_neighbors_vals: List[float] = None,
    w_lbp_vals: List[float] = None,
    w_distance_urban_vals: List[float] = None,
    decay_rate_vals: List[float] = None,
    n_steps: int = 3,
    seed: int = 123,
    output_csv: Path = None
) -> Tuple[Dict, pd.DataFrame]:
    """Optimización por grid search de parámetros del Autómata Celular.

    Argumentos:
        data_dir: Directorio con grids históricos.
        woe_file: Archivo PKL con WoE entrenado.
        year_start: Año inicial.
        year_end: Año final.
        growth_rates: Valores candidatos para ``growth_rate``.
        w_neighbors_vals: Valores candidatos para ``w_neighbors``.
        w_lbp_vals: Valores candidatos para ``w_lbp``.
        w_distance_urban_vals: Valores candidatos para ``w_distance_urban``.
        decay_rate_vals: Valores candidatos para ``decay_rate``.
        n_steps: Pasos de simulación.
        seed: Semilla.
        output_csv: Ruta opcional para guardar la tabla completa.

    Retorna:
        best_config: Configuración ganadora (por IoU).
        results_df: DataFrame con resultados por combinación.
    """
    # Valores por defecto para grid search
    growth_rates = growth_rates or [0.02, 0.03, 0.04, 0.05, 0.06]
    w_neighbors_vals = w_neighbors_vals or [0.6, 0.8, 1.0, 1.2]
    w_lbp_vals = w_lbp_vals or [0.4, 0.6, 0.8]
    w_distance_urban_vals = w_distance_urban_vals or [-0.2, -0.4, -0.6, -0.8, -1.0]
    decay_rate_vals = decay_rate_vals or [0.05, 0.1, 0.15, 0.2]
    
    # Cargar datos
    print(f"\nCargando grids {year_start}→{year_end}...")
    years_data = load_historical_maps(data_dir)
    grid_t0 = years_data[year_start]
    grid_t1 = years_data[year_end]
    
    print(f"Cargando WoE desde {woe_file}...")
    with open(woe_file, 'rb') as f:
        woe_data = pickle.load(f)
    
    woe_calc = woe_data['woe_calculator']
    woe_results = woe_data['woe_results']
    
    # Generar todas las combinaciones
    param_grid = list(itertools.product(
        growth_rates,
        w_neighbors_vals,
        w_lbp_vals,
        w_distance_urban_vals,
        decay_rate_vals
    ))
    
    total_combinations = len(param_grid)
    print(f"\n{'='*60}")
    print(f"Grid Search: {total_combinations} combinaciones")
    print(f"{'='*60}\n")
    
    results = []
    best_iou_so_far = 0
    
    for i, (growth_rate, w_neighbors, w_lbp, w_distance_urban, decay_rate) in enumerate(param_grid, 1):
        print(f"\n[{i}/{total_combinations}] Probando configuración:")
        print(f"  growth_rate={growth_rate:.3f}, w_neighbors={w_neighbors:.2f}, "
              f"w_lbp={w_lbp:.2f}, w_distance_urban={w_distance_urban:.2f}, "
              f"decay_rate={decay_rate:.3f}")
        
        # Crear parámetros
        params = ACParameters(
            w_neighbors=w_neighbors,
            w_lbp=w_lbp,
            w_distance_urban=w_distance_urban,
            decay_rate=decay_rate
        )
        
        # Evaluar
        try:
            metrics = evaluate_parameters(
                grid_t0, grid_t1, woe_calc, woe_results,
                params, growth_rate, n_steps, seed
            )
            
            result = {
                'growth_rate': growth_rate,
                'w_neighbors': w_neighbors,
                'w_lbp': w_lbp,
                'w_distance_urban': w_distance_urban,
                'decay_rate': decay_rate,
                **metrics
            }
            results.append(result)
            
            # Log de resultados inmediatos
            print(f"  ✓ IoU={metrics['iou']:.4f}, Kappa={metrics['kappa']:.4f}, "
                  f"FP={metrics['fp_rate']:.4f}, FN={metrics['fn_rate']:.4f}")
            
            # Trackear mejor hasta ahora
            if metrics['iou'] > best_iou_so_far:
                best_iou_so_far = metrics['iou']
                print(f"  🏆 Nueva mejor IoU: {best_iou_so_far:.4f}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Convertir a DataFrame
    df = pd.DataFrame(results)
    
    # Ordenar por diferentes criterios
    print(f"\n{'='*60}")
    print("TOP 5 por diferentes criterios:")
    print(f"{'='*60}\n")
    
    # 1. Mejor IoU (métrica principal)
    print("1️⃣  Mejor IoU:")
    top_iou = df.nlargest(5, 'iou')[['growth_rate', 'w_neighbors', 'w_lbp', 'w_distance_urban', 'decay_rate', 'iou', 'kappa', 'fp_rate', 'fn_rate']]
    print(top_iou.to_string(index=False))
    print()
    
    # 2. Mejor balance FP/FN (minimizar suma)
    df['fp_fn_sum'] = df['fp_rate'] + df['fn_rate']
    print("2️⃣  Mejor balance FP+FN (menor suma):")
    top_balance = df.nsmallest(5, 'fp_fn_sum')[['growth_rate', 'w_neighbors', 'w_lbp', 'w_distance_urban', 'decay_rate', 'iou', 'kappa', 'fp_rate', 'fn_rate', 'fp_fn_sum']]
    print(top_balance.to_string(index=False))
    print()
    
    # 3. Mejor Kappa
    print("3️⃣  Mejor Kappa:")
    top_kappa = df.nlargest(5, 'kappa')[['growth_rate', 'w_neighbors', 'w_lbp', 'w_distance_urban', 'decay_rate', 'iou', 'kappa', 'fp_rate', 'fn_rate']]
    print(top_kappa.to_string(index=False))
    print()
    
    # 4. Menor FP (menos sobre-predicción)
    print("4️⃣  Menor FP (menos sobre-predicción):")
    top_fp = df.nsmallest(5, 'fp_rate')[['growth_rate', 'w_neighbors', 'w_lbp', 'w_distance_urban', 'decay_rate', 'iou', 'kappa', 'fp_rate', 'fn_rate']]
    print(top_fp.to_string(index=False))
    print()
    
    # Mejor configuración global (por IoU)
    best_idx = df['iou'].idxmax()
    best_config = df.loc[best_idx].to_dict()
    
    print(f"{'='*60}")
    print("✅ MEJOR CONFIGURACIÓN (por IoU):")
    print(f"{'='*60}")
    print(f"  growth_rate:       {best_config['growth_rate']:.3f}")
    print(f"  w_neighbors:       {best_config['w_neighbors']:.3f}")
    print(f"  w_lbp:             {best_config['w_lbp']:.3f}")
    print(f"  w_distance_urban:  {best_config['w_distance_urban']:.3f}")
    print(f"  decay_rate:        {best_config['decay_rate']:.3f}")
    print(f"\nMÉTRICAS:")
    print(f"  IoU:        {best_config['iou']:.4f}")
    print(f"  Kappa:      {best_config['kappa']:.4f}")
    print(f"  Accuracy:   {best_config['accuracy']:.4f}")
    print(f"  Precision:  {best_config['precision']:.4f}")
    print(f"  Recall:     {best_config['recall']:.4f}")
    print(f"\nERRORES:")
    print(f"  FP rate:    {best_config['fp_rate']:.4f} ({best_config['fp']:,} celdas)")
    print(f"  FN rate:    {best_config['fn_rate']:.4f} ({best_config['fn']:,} celdas)")
    print(f"  TP rate:    {best_config['tp_rate']:.4f} ({best_config['tp']:,} celdas)")
    print(f"  TN rate:    {best_config['tn_rate']:.4f} ({best_config['tn']:,} celdas)")
    print(f"{'='*60}\n")
    
    # Guardar resultados
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"✓ Resultados guardados en: {output_csv}")
        
        # Guardar mejor configuración en JSON
        json_path = output_csv.parent / output_csv.name.replace('.csv', '_best.json')
        with open(json_path, 'w') as f:
            json.dump(best_config, f, indent=2)
        print(f"✓ Mejor configuración guardada en: {json_path}")
    
    return best_config, df


def main():
    parser = argparse.ArgumentParser(
        description="Optimiza parámetros AC usando grid search"
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data/processed/standardized_maps'),
        help='Directorio con grids procesados'
    )
    parser.add_argument(
        '--woe-file',
        type=Path,
        default=Path('reports/woe_trained_1984_2014.pkl'),
        help='Archivo pickle con WoE pre-entrenado'
    )
    parser.add_argument(
        '--year-start',
        type=int,
        default=2016,
        help='Año inicial para optimización'
    )
    parser.add_argument(
        '--year-end',
        type=int,
        default=2017,
        help='Año final para optimización'
    )
    parser.add_argument(
        '--n-steps',
        type=int,
        default=3,
        help='Número de pasos AC'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=123,
        help='Semilla para reproducibilidad'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('reports/optimization_results.csv'),
        help='Archivo CSV de salida con todos los resultados'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Grid search reducido (más rápido)'
    )
    
    args = parser.parse_args()
    
    # Definir grid según modo
    if args.quick:
        print("\n🚀 Modo rápido: grid reducido")
        growth_rates = [0.03, 0.05]
        w_neighbors_vals = [0.8, 1.0]
        w_lbp_vals = [0.6]
        w_distance_urban_vals = [-0.4, -0.6, -0.8]
        decay_rate_vals = [0.1, 0.15]
    else:
        print("\n🔍 Modo completo: grid exhaustivo")
        growth_rates = [0.02, 0.03, 0.04, 0.05, 0.06]
        w_neighbors_vals = [0.6, 0.8, 1.0, 1.2]
        w_lbp_vals = [0.4, 0.6, 0.8]
        w_distance_urban_vals = [-0.2, -0.4, -0.6, -0.8, -1.0]
        decay_rate_vals = [0.05, 0.1, 0.15, 0.2]
    
    best_config, df = optimize_parameters(
        data_dir=args.data_dir,
        woe_file=args.woe_file,
        year_start=args.year_start,
        year_end=args.year_end,
        growth_rates=growth_rates,
        w_neighbors_vals=w_neighbors_vals,
        w_lbp_vals=w_lbp_vals,
        w_distance_urban_vals=w_distance_urban_vals,
        decay_rate_vals=decay_rate_vals,
        n_steps=args.n_steps,
        seed=args.seed,
        output_csv=args.output
    )


if __name__ == '__main__':
    main()
