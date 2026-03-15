#!/usr/bin/env python3
"""
Entrena WoE con datos históricos 1984-2014 y lo guarda
Usar este WoE guardado hace predicciones MÁS RÁPIDAS
"""

import pickle
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from tesis_ac.historical.extract_transitions import load_historical_maps
from tesis_ac.ca.rules import create_default_spatial_variables
from tesis_ac.woe.woe import calculate_spatial_woe_variables

def train_pooled_woe():
    print('='*80)
    print('🧮 ENTRENAMIENTO DE WoE POOLED (1984-2010)')
    print('='*80)
    print()
    
    start = datetime.now()
    
    # Cargar mapas de entrenamiento
    print('📂 Cargando mapas históricos 1984-2010...')
    data_dir = Path('data/processed/standardized_maps')
    years_data = load_historical_maps(data_dir, file_pattern="svm_2_classes.npy")
    
    train_years = list(range(1984, 2011))  # Hasta 2010 (dejar 2011-2016 para validación)
    available_years = [y for y in train_years if y in years_data]
    
    print(f'✓ {len(available_years)} años disponibles')
    
    # Calcular transiciones acumuladas
    print(f'\n🔄 Calculando transiciones de {len(available_years)-1} períodos...')
    
    all_transitions = []
    all_spatial_vars = {var: [] for var in [
        'distance_urban', 'neighbor_density_3x3', 'neighbor_density_5x5',
        'neighbor_density_7x7', 'local_fragmentation', 'nearest_cluster_size',
        'urban_gradient'
    ]}
    
    for i in range(len(available_years) - 1):
        t0_year = available_years[i]
        t1_year = available_years[i + 1]
        
        grid_t0 = years_data[t0_year]
        grid_t1 = years_data[t1_year]
        
        # Detectar transiciones
        transitions = (grid_t0 == 0) & (grid_t1 == 1)
        all_transitions.append(transitions)
        
        # Calcular variables espaciales
        spatial_vars = create_default_spatial_variables(
            grid_shape=grid_t0.shape,
            existing_urban=grid_t0
        )
        
        # Acumular variables
        for var_name, var_data in spatial_vars.items():
            all_spatial_vars[var_name].append(var_data)
        
        n_trans = transitions.sum()
        print(f'  {t0_year}→{t1_year}: {n_trans:,} transiciones')
    
    print(f'\n✓ Total transiciones acumuladas: {sum(t.sum() for t in all_transitions):,}')
    
    # Promediar variables espaciales
    print(f'\n🧮 Promediando variables espaciales...')
    avg_spatial_vars = {
        var_name: np.mean(var_list, axis=0)
        for var_name, var_list in all_spatial_vars.items()
    }
    
    # Preparar datos para WoE
    print(f'\n🎯 Preparando datos para WoE...')
    
    # Stack de todos los grids t0 y t1
    grids_t0 = [years_data[available_years[i]] for i in range(len(available_years)-1)]
    grids_t1 = [years_data[available_years[i+1]] for i in range(len(available_years)-1)]
    
    # Aplanar todo para calcular WoE global
    all_transitions_flat = np.concatenate([
        ((grids_t0[i] == 0) & (grids_t1[i] == 1)).flatten() 
        for i in range(len(grids_t0))
    ])
    
    # Usar promedio de variables
    grid_shape = grids_t0[0].shape
    
    # Calcular WoE manualmente
    from tesis_ac.woe.woe import WoECalculator
    
    woe_calc = WoECalculator(min_bin_size=50, max_bins=10)
    
    # Para cada variable, calcular WoE
    woe_results = {}
    for var_name, var_data in avg_spatial_vars.items():
        print(f'  Calculando WoE para {var_name}...')
        
        # Aplanar variable
        var_flat = var_data.flatten()
        
        # Calcular WoE
        woe_result = woe_calc.calculate_woe(
            variable=np.tile(var_flat, len(grids_t0)),  # Repetir para cada período
            target=all_transitions_flat,
            variable_name=var_name
        )
        
        woe_results[var_name] = woe_result
    
    # Guardar resultados en el calculador
    woe_calc.woe_results = woe_results
    
    print(f'✓ WoE calculado para {len(woe_results)} variables')
    
    # Guardar WoE
    output_file = Path('data/processed/woe_pooled_1984_2010.pkl')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f'\n💾 Guardando WoE en {output_file}...')
    
    woe_data = {
        'woe_calculator': woe_calc,
        'metadata': {
            'trained_years': f'{available_years[0]}-{available_years[-1]}',
            'n_periods': len(available_years) - 1,
            'total_transitions': sum(t.sum() for t in all_transitions),
            'variables': list(woe_calc.woe_results.keys()),
            'n_bins': 10,
            'trained_date': datetime.now().isoformat()
        }
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(woe_data, f)
    
    elapsed = (datetime.now() - start).total_seconds()
    
    print('✓ WoE guardado exitosamente')
    print(f'\n{"="*80}')
    print('✅ ENTRENAMIENTO COMPLETADO')
    print(f'{"="*80}')
    print(f'\n📊 Resumen:')
    print(f'  • Períodos entrenados: {len(available_years)-1}')
    print(f'  • Variables: {len(woe_calc.woe_results)}')
    print(f'  • Archivo: {output_file}')
    print(f'  • Tamaño: {output_file.stat().st_size / 1024:.1f} KB')
    print(f'  • Tiempo: {elapsed:.1f} segundos')
    print(f'\n💡 Ahora puedes usar este WoE en predicciones rápidas')
    print(f'   sin necesidad de recalcularlo cada vez.')

if __name__ == '__main__':
    train_pooled_woe()
