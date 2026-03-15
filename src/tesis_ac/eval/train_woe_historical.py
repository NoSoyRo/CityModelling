"""
Entrenar WoE usando datos históricos (1984-2014) y guardar para reutilizar.

Uso:
  python -m tesis_ac.eval.train_woe_historical \
    --data-dir data/processed/standardized_maps \
    --train-years 1984 2014 \
    --output reports/woe_trained_1984_2014.pkl
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict

import numpy as np

from tesis_ac.historical.extract_transitions import load_historical_maps
from tesis_ac.ca.rules import create_default_spatial_variables
from tesis_ac.woe.woe import WoECalculator

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_woe_historical(
    data_dir: Path,
    train_start: int,
    train_end: int,
    output: Path,
    invert_classes: bool = False,
    min_bin_size: int = 100,
    max_bins: int = 8,
) -> Dict:
    """Entrena WoE usando pooling de pares de años consecutivos.

    Argumentos:
        data_dir: Directorio con grids por año.
        train_start: Año inicial del rango de entrenamiento.
        train_end: Año final del rango de entrenamiento.
        output: Ruta para guardar el WoE entrenado.
        invert_classes: Si invierte 0/1 (si la data trae 0=urbano).
        min_bin_size: Tamaño mínimo de bin para WoE.
        max_bins: Número máximo de bins.

    Retorna:
        Diccionario con el WoECalculator, resultados por variable y metadatos
        del entrenamiento.
    """
    print(f"Cargando datos históricos desde {data_dir}...")
    years_data = load_historical_maps(data_dir, file_pattern="svm_2_classes.npy")
    
    all_years = sorted(years_data.keys())
    train_years = [y for y in all_years if train_start <= y <= train_end]
    
    print(f"Años de entrenamiento: {train_years[0]} - {train_years[-1]} ({len(train_years)} años, {len(train_years)-1} pares)")
    
    # Pooling de variables y targets
    pooled_vars: Dict[str, list] = {}
    pooled_targets: list = []
    
    print("Procesando pares de años...")
    for i, (y0, y1) in enumerate(zip(train_years[:-1], train_years[1:])):
        if i % 5 == 0:
            print(f"  Par {i+1}/{len(train_years)-1}: {y0}→{y1}")
            
        g0 = years_data[y0]
        g1 = years_data[y1]
        
        if invert_classes:
            g0 = 1 - g0
            g1 = 1 - g1
        
        # Crear variables espaciales objetivas
        fvars = create_default_spatial_variables(g0.shape, existing_urban=g0)
        
        # Máscara de cambio 0→1
        urban_history = ((g0 == 0) & (g1 == 1)).astype(int).flatten()
        pooled_targets.append(urban_history)
        
        # Acumular variables
        for name, var in fvars.items():
            pooled_vars.setdefault(name, []).append(var.flatten())
    
    # Concatenar todos los datos
    print("Concatenando datos...")
    target_concat = np.concatenate(pooled_targets)
    
    print(f"Total de celdas: {len(target_concat):,}")
    print(f"Celdas que cambiaron (0→1): {target_concat.sum():,} ({100*target_concat.mean():.2f}%)")
    
    # Entrenar WoE por variable
    print(f"\nEntrenando WoE (min_bin_size={min_bin_size}, max_bins={max_bins})...")
    woe_calc = WoECalculator(min_bin_size=min_bin_size, max_bins=max_bins)
    woe_results = {}
    
    for name, parts in pooled_vars.items():
        print(f"  Variable: {name}")
        var_concat = np.concatenate(parts)
        res = woe_calc.calculate_woe(
            var_concat, 
            target_concat, 
            variable_name=name, 
            binning_method='quantile'
        )
        woe_results[name] = res
        print(f"    IV = {res.iv_total:.4f}, bins = {len(res.bins)}")
    
    # Guardar
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    result = {
        'woe_calculator': woe_calc,
        'woe_results': woe_results,
        'train_years': (train_start, train_end),
        'train_pairs': len(train_years) - 1,
        'total_cells': len(target_concat),
        'changed_cells': int(target_concat.sum()),
        'change_rate': float(target_concat.mean()),
        'variables': list(woe_results.keys()),
    }
    
    with open(output, 'wb') as f:
        pickle.dump(result, f)
    
    print(f"\n✓ WoE entrenado guardado en: {output}")
    print(f"  Variables: {list(woe_results.keys())}")
    print(f"  Pares usados: {len(train_years)-1}")
    print(f"  Tasa de cambio global: {100*result['change_rate']:.3f}%")
    
    # Resumen WoE
    print("\nResumen por variable:")
    print(woe_calc.summary_report())
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Entrenar WoE con datos históricos y guardar para reutilizar"
    )
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='Directorio con subcarpetas year_YYYY/')
    parser.add_argument('--train-years', type=int, nargs=2, metavar=('START', 'END'),
                        required=True,
                        help='Rango de años para entrenamiento (ej. 1984 2014)')
    parser.add_argument('--output', type=Path, 
                        default=Path('reports/woe_trained.pkl'),
                        help='Ruta de salida para WoE entrenado (.pkl)')
    parser.add_argument('--invert-classes', action='store_true',
                        help='Invertir 0/1 si la data tiene 0=urbano')
    parser.add_argument('--min-bin-size', type=int, default=100,
                        help='Tamaño mínimo de bin para WoE')
    parser.add_argument('--max-bins', type=int, default=8,
                        help='Número máximo de bins')
    
    args = parser.parse_args()
    
    train_woe_historical(
        data_dir=args.data_dir,
        train_start=args.train_years[0],
        train_end=args.train_years[1],
        output=args.output,
        invert_classes=args.invert_classes,
        min_bin_size=args.min_bin_size,
        max_bins=args.max_bins,
    )


if __name__ == '__main__':
    main()
