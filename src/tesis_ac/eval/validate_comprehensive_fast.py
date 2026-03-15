"""
Script de validación comprehensiva RÁPIDA (sin nearest_cluster_size).

Esta versión omite la variable más lenta para acelerar el proceso.
"""
import argparse
import pickle
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict

from tesis_ac.historical.extract_transitions import load_historical_maps
from tesis_ac.ca.rules import WoECellularAutomaton, ACParameters
from tesis_ac.eval.comprehensive_metrics import (
    calculate_all_metrics,
    print_metrics_report
)
from tesis_ac.eval.morphological_metrics import (
    calculate_morphological_metrics,
    compare_morphology
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_fast_spatial_variables(shape, existing_urban):
    """Crea variables espaciales rápidas (omite `nearest_cluster_size`).

    Argumentos:
        shape: Forma del grid (alto, ancho).
        existing_urban: Grid base para derivar máscaras/distancias (urbano>0).

    Retorna:
        Diccionario ``{nombre_variable: array}``.
    """
    from scipy.ndimage import distance_transform_edt, uniform_filter, generic_filter, sobel
    import time
    
    logger.info("  Generando variables espaciales (6/7, omitiendo nearest_cluster_size)...")
    
    urban_mask = (existing_urban > 0).astype(int)
    variables = {}
    
    # 1. Distance to urban
    start = time.time()
    logger.info("  [1/6] Calculando distance_urban...")
    dist = distance_transform_edt(~urban_mask.astype(bool))
    variables['distance_urban'] = dist
    logger.info(f"        ✓ Completado en {time.time()-start:.2f}s")
    
    # 2-4. Neighbor densities
    for i, size in enumerate([3, 5, 7], start=2):
        start = time.time()
        logger.info(f"  [{i}/6] Calculando neighbor_density_{size}x{size}...")
        density = uniform_filter(urban_mask.astype(float), size=size, mode='constant')
        variables[f'neighbor_density_{size}x{size}'] = density
        logger.info(f"        ✓ Completado en {time.time()-start:.2f}s")
    
    # 5. Local fragmentation
    start = time.time()
    logger.info("  [5/6] Calculando local_fragmentation...")
    def local_variance(values):
        return np.var(values)
    frag = generic_filter(urban_mask.astype(float), local_variance, size=3)
    variables['local_fragmentation'] = frag
    logger.info(f"        ✓ Completado en {time.time()-start:.2f}s")
    
    # 6. Urban gradient (SKIP nearest_cluster_size)
    start = time.time()
    logger.info("  [6/6] Calculando urban_gradient...")
    grad_x = sobel(urban_mask.astype(float), axis=0)
    grad_y = sobel(urban_mask.astype(float), axis=1)
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    variables['urban_gradient'] = gradient
    logger.info(f"        ✓ Completado en {time.time()-start:.2f}s")
    
    return variables


def validate_period(data_dir: Path,
                   woe_data: dict,
                   year_start: int,
                   year_end: int) -> Dict[str, any]:
    """Valida un periodo con métricas comprehensivas (versión rápida)."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Validando periodo {year_start}→{year_end}")
    logger.info(f"{'='*70}")
    
    # Cargar grids reales
    logger.info(f"Cargando grids {year_start} y {year_end}...")
    years_data = load_historical_maps(data_dir)
    
    if year_start not in years_data:
        raise ValueError(f"Año {year_start} no encontrado en {data_dir}")
    if year_end not in years_data:
        raise ValueError(f"Año {year_end} no encontrado en {data_dir}")
    
    t0_real = years_data[year_start]
    t1_real = years_data[year_end]
    
    # Extraer componentes WoE
    woe_calc = woe_data['woe_calculator']
    woe_results = woe_data['woe_results']
    
    # Simular predicción
    logger.info(f"Simulando {year_start}→{year_end}...")
    np.random.seed(42)
    
    grid_shape = t0_real.shape
    spatial_vars = create_fast_spatial_variables(grid_shape, existing_urban=t0_real)
    
    ac = WoECellularAutomaton(grid_shape, woe_calc, parameters=ACParameters())
    ac.set_initial_state(t0_real)
    
    # Agregar variables con WoE pre-calculado (omitir nearest_cluster_size)
    for name, var in spatial_vars.items():
        if name in woe_results:
            ac.add_spatial_variable(name, var, compute_woe=True)
    
    # Simular n_steps
    n_steps = year_end - year_start
    for step in range(n_steps):
        grid_pred, stats = ac.step()
        if stats.get('new_urban_cells', 1) == 0:
            logger.info(f"  Simulación detenida en paso {step+1}/{n_steps} (sin más cambios)")
            break
    
    # IMPORTANTE: Invertir clases para comparación consistente
    t1_predicted = 1 - grid_pred
    
    # Calcular métricas comprehensivas
    logger.info("Calculando métricas comprehensivas...")
    metrics = calculate_all_metrics(t0_real, t1_real, t1_predicted)
    
    # Calcular métricas morfológicas
    logger.info("Calculando métricas morfológicas...")
    morphology = compare_morphology(t1_real, t1_predicted)
    metrics['morphology'] = morphology
    
    # Agregar metadatos
    metrics['period'] = f"{year_start}→{year_end}"
    metrics['year_start'] = year_start
    metrics['year_end'] = year_end
    metrics['years_simulated'] = year_end - year_start
    
    # Mostrar reporte
    print_metrics_report(metrics, period=f"{year_start}→{year_end}")
    
    return metrics


def metrics_to_dataframe(all_metrics: List[Dict[str, any]]) -> pd.DataFrame:
    """Convierte lista de métricas a DataFrame consolidado."""
    rows = []
    
    for m in all_metrics:
        row = {
            'period': m['period'],
            'year_start': m['year_start'],
            'year_end': m['year_end'],
            'years': m['years_simulated'],
            
            # Matriz de confusión
            'TP': m['confusion_matrix']['TP'],
            'TN': m['confusion_matrix']['TN'],
            'FP': m['confusion_matrix']['FP'],
            'FN': m['confusion_matrix']['FN'],
            
            # Métricas espaciales
            'accuracy': m['spatial_metrics']['accuracy'],
            'precision': m['spatial_metrics']['precision'],
            'recall': m['spatial_metrics']['recall'],
            'f1_score': m['spatial_metrics']['f1_score'],
            'kappa': m['spatial_metrics']['kappa'],
            
            # Figure of Merit
            'fom': m['figure_of_merit']['fom'],
            'A_correct_change': m['figure_of_merit']['A_correct_change'],
            'B_false_change': m['figure_of_merit']['B_false_change'],
            'C_missed_change': m['figure_of_merit']['C_missed_change'],
            'change_rate_real': m['figure_of_merit']['change_rate_real'],
            'change_rate_predicted': m['figure_of_merit']['change_rate_predicted'],
            
            # Detección de cambio
            'change_precision': m['change_detection']['change_precision'],
            'change_recall': m['change_detection']['change_recall'],
            'change_f1': m['change_detection']['change_f1'],
            
            # Métricas morfológicas (real)
            'num_patches_real': m['morphology']['real']['num_patches'],
            'mean_patch_size_real': m['morphology']['real']['mean_patch_size'],
            'aggregation_index_real': m['morphology']['real']['aggregation_index'],
            'fragmentation_index_real': m['morphology']['real']['fragmentation_index'],
            
            # Métricas morfológicas (predicho)
            'num_patches_pred': m['morphology']['predicted']['num_patches'],
            'mean_patch_size_pred': m['morphology']['predicted']['mean_patch_size'],
            'aggregation_index_pred': m['morphology']['predicted']['aggregation_index'],
            'fragmentation_index_pred': m['morphology']['predicted']['fragmentation_index'],
            
            # Diferencias morfológicas (%)
            'num_patches_diff_pct': m['morphology']['differences']['num_patches_diff_pct'],
            'aggregation_diff_pct': m['morphology']['differences']['aggregation_index_diff_pct']
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def print_summary_table(df: pd.DataFrame):
    """Imprime tabla resumen de métricas."""
    print("\n" + "="*90)
    print("RESUMEN DE VALIDACIÓN - TODAS LAS MÉTRICAS")
    print("="*90)
    
    print("\n📊 MÉTRICAS ESPACIALES GLOBALES")
    print(df[['period', 'accuracy', 'precision', 'recall', 'f1_score', 'kappa']].to_string(index=False))
    
    print("\n🎯 FIGURE OF MERIT (FoM) - Métrica Crítica")
    print(df[['period', 'fom', 'change_rate_real', 'change_rate_predicted']].to_string(index=False))
    
    print("\n🔄 DETECCIÓN DE CAMBIO")
    print(df[['period', 'change_precision', 'change_recall', 'change_f1']].to_string(index=False))
    
    print("\n🏙️ MÉTRICAS MORFOLÓGICAS")
    print(df[['period', 'num_patches_real', 'num_patches_pred', 'num_patches_diff_pct']].to_string(index=False))
    print(df[['period', 'aggregation_index_real', 'aggregation_index_pred', 'aggregation_diff_pct']].to_string(index=False))
    
    # Estadísticas promedio
    print("\n📈 ESTADÍSTICAS PROMEDIO")
    print(f"  Accuracy promedio:       {df['accuracy'].mean():>8.2%}")
    print(f"  Precision promedio:      {df['precision'].mean():>8.2%}")
    print(f"  Recall promedio:         {df['recall'].mean():>8.2%}")
    print(f"  F1-Score promedio:       {df['f1_score'].mean():>8.4f}")
    print(f"  Kappa promedio:          {df['kappa'].mean():>8.4f}")
    print(f"  FoM promedio:            {df['fom'].mean():>8.4f}")
    print(f"  Agregación (real):       {df['aggregation_index_real'].mean():>8.4f}")
    print(f"  Agregación (pred):       {df['aggregation_index_pred'].mean():>8.4f}")
    
    # Interpretación FoM promedio
    fom_avg = df['fom'].mean()
    if fom_avg > 0.50:
        status = "🟢 EXCELENTE"
    elif fom_avg > 0.40:
        status = "🟡 MUY BUENO"
    elif fom_avg > 0.25:
        status = "🟠 ACEPTABLE"
    else:
        status = "🔴 MEJORABLE"
    print(f"  FoM Interpretación:      {status}")
    
    print("\n" + "="*90 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Validación comprehensiva RÁPIDA (sin nearest_cluster_size)"
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        required=True,
        help='Directorio con grids estandarizados'
    )
    parser.add_argument(
        '--woe-file',
        type=Path,
        required=True,
        help='Archivo .pkl con modelo WoE entrenado'
    )
    parser.add_argument(
        '--validation-years',
        type=int,
        nargs='+',
        required=True,
        help='Años para validación (ej: 2016 2017 2018 2019 2020)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('reports/validation_comprehensive_6vars.csv'),
        help='Archivo CSV de salida con métricas'
    )
    
    args = parser.parse_args()
    
    # Validar inputs
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Directorio no existe: {args.data_dir}")
    
    if not args.woe_file.exists():
        raise FileNotFoundError(f"Archivo WoE no existe: {args.woe_file}")
    
    if len(args.validation_years) < 2:
        raise ValueError("Se requieren al menos 2 años para validación")
    
    # Cargar modelo WoE
    logger.info(f"Cargando modelo WoE desde {args.woe_file}")
    with open(args.woe_file, 'rb') as f:
        woe_data = pickle.load(f)
    
    # Verificar estructura
    if 'woe_calculator' not in woe_data or 'woe_results' not in woe_data:
        raise ValueError("Archivo WoE no tiene estructura esperada (woe_calculator, woe_results)")
    
    logger.info(f"Variables en modelo: {list(woe_data['woe_results'].keys())}")
    logger.info("⚠️  NOTA: Omitiendo 'nearest_cluster_size' para acelerar validación")
    
    # Validar todos los periodos consecutivos
    years = sorted(args.validation_years)
    all_metrics = []
    
    for i in range(len(years) - 1):
        year_start = years[i]
        year_end = years[i + 1]
        
        try:
            metrics = validate_period(
                data_dir=args.data_dir,
                woe_data=woe_data,
                year_start=year_start,
                year_end=year_end
            )
            all_metrics.append(metrics)
            
        except Exception as e:
            logger.error(f"Error validando {year_start}→{year_end}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_metrics:
        logger.error("No se pudo validar ningún periodo")
        return
    
    # Convertir a DataFrame
    df = metrics_to_dataframe(all_metrics)
    
    # Guardar CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, float_format='%.6f')
    logger.info(f"✓ Métricas guardadas en {args.output}")
    
    # Mostrar resumen
    print_summary_table(df)
    
    logger.info("✓ Validación comprehensiva completada")


if __name__ == '__main__':
    main()
