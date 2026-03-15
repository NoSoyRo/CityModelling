"""
Visualización de comparación entre predicciones y realidad.

Genera mapas lado a lado mostrando:
- Grid real (t1)
- Grid predicho (mejor réplica)
- Mapa de diferencias (TP, FP, FN, TN)
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional

from tesis_ac.historical.extract_transitions import load_historical_maps


def load_grids_for_period(
    data_dir: Path,
    year_start: int,
    year_end: int,
    woe_file: Path,
    n_steps: int = 3,
    seed: int = 123
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Carga y simula los grids para un período específico.
    
    Args:
        data_dir: Directorio con los grids procesados
        year_start: Año inicial
        year_end: Año final
        woe_file: Path al archivo pickle con WoE entrenado
        n_steps: Número de pasos de simulación
        seed: Semilla para reproducibilidad
        
    Returns:
        (grid_t0, grid_t1_real, grid_t1_pred)
    """
    import pickle
    from tesis_ac.ca.rules import WoECellularAutomaton, ACParameters, create_default_spatial_variables
    
    # Cargar grids reales desde archivos
    years_data = load_historical_maps(data_dir)
    
    if year_start not in years_data:
        raise ValueError(f"Año {year_start} no encontrado en {data_dir}")
    if year_end not in years_data:
        raise ValueError(f"Año {year_end} no encontrado en {data_dir}")
    
    grid_t0 = years_data[year_start]
    grid_t1_real = years_data[year_end]
    
    # Cargar WoE pre-entrenado
    with open(woe_file, 'rb') as f:
        woe_data = pickle.load(f)
    
    woe_calc = woe_data['woe_calculator']
    woe_results = woe_data['woe_results']
    
    # Re-simular para obtener predicción
    np.random.seed(seed)
    
    grid_shape = grid_t0.shape
    spatial_vars = create_default_spatial_variables(grid_shape, existing_urban=grid_t0)
    
    ac = WoECellularAutomaton(grid_shape, woe_calc, parameters=ACParameters())
    ac.set_initial_state(grid_t0)
    
    # Agregar variables con WoE pre-calculado
    for name, var in spatial_vars.items():
        if name in woe_results:
            ac.add_spatial_variable(name, var, compute_woe=True)
    
    # Simular n_steps
    for _ in range(n_steps):
        grid_pred, stats = ac.step()
        if stats.get('new_urban_cells', 1) == 0:
            break
    
    # IMPORTANTE: Invertir clases para comparación consistente
    # El modelo predice con clases invertidas (0=urbano, 1=no-urbano)
    # pero los grids reales usan (1=urbano, 0=no-urbano)
    grid_t1_pred = 1 - grid_pred
    
    return grid_t0, grid_t1_real, grid_t1_pred


def create_difference_map(real: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Crea un mapa categórico de diferencias:
    0 = True Negative (correcto: no urbanizado)
    1 = False Positive (error: predijo urbanización que no ocurrió)
    2 = False Negative (error: no predijo urbanización que sí ocurrió)
    3 = True Positive (correcto: predijo urbanización correctamente)
    """
    diff_map = np.zeros_like(real, dtype=int)
    
    # True Negatives: ambos 0
    diff_map[(real == 0) & (pred == 0)] = 0
    
    # False Positives: real=0, pred=1
    diff_map[(real == 0) & (pred == 1)] = 1
    
    # False Negatives: real=1, pred=0
    diff_map[(real == 1) & (pred == 0)] = 2
    
    # True Positives: ambos 1
    diff_map[(real == 1) & (pred == 1)] = 3
    
    return diff_map


def plot_comparison(
    grid_t0: np.ndarray,
    grid_t1_real: np.ndarray,
    grid_t1_pred: np.ndarray,
    period: str,
    output_path: Path,
    figsize: tuple[int, int] = (18, 5)
):
    """
    Genera la visualización comparativa.
    """
    diff_map = create_difference_map(grid_t1_real, grid_t1_pred)
    
    # Contar estadísticas
    tn = np.sum(diff_map == 0)
    fp = np.sum(diff_map == 1)
    fn = np.sum(diff_map == 2)
    tp = np.sum(diff_map == 3)
    total = tn + fp + fn + tp
    
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Crear figura
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # 1. Grid t0 (contexto)
    axes[0].imshow(grid_t0, cmap='Greys', interpolation='nearest')
    axes[0].set_title(f'Estado Inicial (t0)\n{period.split("→")[0]}', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Grid real t1
    axes[1].imshow(grid_t1_real, cmap='Reds', interpolation='nearest')
    axes[1].set_title(f'Real (t1)\n{period.split("→")[1]}', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # 3. Grid predicho
    axes[2].imshow(grid_t1_pred, cmap='Blues', interpolation='nearest')
    axes[2].set_title('Predicción AC-WoE', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # 4. Mapa de diferencias
    # Colores: TN=gris claro, FP=naranja, FN=morado, TP=verde
    colors = ['#e0e0e0', '#ff8c00', '#9400d3', '#00cc66']
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    
    im = axes[3].imshow(diff_map, cmap=cmap, interpolation='nearest', vmin=0, vmax=3)
    axes[3].set_title('Análisis de Diferencias', fontsize=12, fontweight='bold')
    axes[3].axis('off')
    
    # Leyenda para mapa de diferencias
    labels = [
        f'TN: {tn:,} ({tn/total*100:.1f}%)',
        f'FP: {fp:,} ({fp/total*100:.1f}%)',
        f'FN: {fn:,} ({fn/total*100:.1f}%)',
        f'TP: {tp:,} ({tp/total*100:.1f}%)'
    ]
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(4)]
    axes[3].legend(handles=patches, loc='upper right', bbox_to_anchor=(1.3, 1.0), 
                   fontsize=9, framealpha=0.9)
    
    # Título general con métricas
    fig.suptitle(
        f'Comparación Predicción vs Realidad — {period}\n'
        f'Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}',
        fontsize=14, fontweight='bold', y=0.98
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: {output_path}")
    plt.close()


def plot_all_periods(
    data_dir: Path,
    woe_file: Path,
    results_dir: Path,
    output_dir: Path,
    n_steps: int = 3,
    seed: int = 123,
    pattern: str = "validation_*_*_hist.json"
):
    """
    Genera comparaciones para todos los períodos encontrados.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_files = sorted(results_dir.glob(pattern))
    
    if not json_files:
        print(f"❌ No se encontraron archivos con patrón {pattern} en {results_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"Generando comparaciones para {len(json_files)} períodos")
    print(f"{'='*60}\n")
    
    for json_file in json_files:
        # Extraer período del nombre
        # validation_2016_2017_hist.json -> 2016→2017
        parts = json_file.stem.split('_')
        if len(parts) >= 3:
            year_start = int(parts[1])
            year_end = int(parts[2])
            period = f"{year_start}→{year_end}"
        else:
            period = json_file.stem
            continue
        
        print(f"Procesando {period}...")
        
        # Cargar grids
        try:
            grid_t0, grid_t1_real, grid_t1_pred = load_grids_for_period(
                data_dir, year_start, year_end, woe_file, n_steps, seed
            )
        except Exception as e:
            print(f"  ⚠️  Error cargando grids: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Generar plot
        output_path = output_dir / f"comparison_{year_start}_{year_end}.png"
        plot_comparison(grid_t0, grid_t1_real, grid_t1_pred, period, output_path)
    
    print(f"\n{'='*60}")
    print(f"✓ Comparaciones completadas en {output_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Genera visualizaciones comparando predicciones vs realidad"
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data/processed/standardized_maps'),
        help='Directorio con los grids procesados'
    )
    parser.add_argument(
        '--woe-file',
        type=Path,
        default=Path('reports/woe_trained_1984_2014.pkl'),
        help='Archivo pickle con WoE pre-entrenado'
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=Path('reports'),
        help='Directorio con los JSONs de validación'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('figures/comparisons'),
        help='Directorio de salida para las imágenes'
    )
    parser.add_argument(
        '--n-steps',
        type=int,
        default=3,
        help='Número de pasos de simulación AC'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=123,
        help='Semilla para reproducibilidad'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='validation_*_*_hist.json',
        help='Patrón para buscar archivos JSON'
    )
    parser.add_argument(
        '--year-start',
        type=int,
        help='Año inicial (para procesamiento único con --year-end)'
    )
    parser.add_argument(
        '--year-end',
        type=int,
        help='Año final (para procesamiento único con --year-start)'
    )
    
    args = parser.parse_args()
    
    if args.year_start and args.year_end:
        # Procesar un solo período específico
        period = f"{args.year_start}→{args.year_end}"
        print(f"Procesando {period}...")
        
        try:
            grid_t0, grid_t1_real, grid_t1_pred = load_grids_for_period(
                args.data_dir, args.year_start, args.year_end, 
                args.woe_file, args.n_steps, args.seed
            )
        except Exception as e:
            print(f"❌ Error cargando grids: {e}")
            import traceback
            traceback.print_exc()
            return
        
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / f"comparison_{args.year_start}_{args.year_end}.png"
        
        plot_comparison(grid_t0, grid_t1_real, grid_t1_pred, period, output_path)
    else:
        # Procesar todos los períodos
        plot_all_periods(
            args.data_dir, args.woe_file, args.results_dir, 
            args.output_dir, args.n_steps, args.seed, args.pattern
        )


if __name__ == '__main__':
    main()
