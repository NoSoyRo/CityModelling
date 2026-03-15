#!/usr/bin/env python3
"""
Re-entrenar modelo WoE usando SOLO 1984-2010 (26 períodos)
Esto deja 2011-2016 como validación completamente independiente.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from pathlib import Path
import pickle
from datetime import datetime
from tesis_ac.woe.woe import WoECalculator
from tesis_ac.ca.rules import create_default_spatial_variables
import time

print("="*80)
print("🎯 ENTRENAMIENTO WoE 1984-2010 (26 PERÍODOS)")
print("="*80)
print()

# ==============================================================================
# 1. CONFIGURACIÓN
# ==============================================================================
START_YEAR = 1984
END_YEAR = 2010  # Hasta 2010 (no incluir 2011-2016)
MAPS_DIR = Path('data/processed/standardized_maps')
OUTPUT_FILE = 'data/processed/woe_pooled_1984_2010.pkl'

# Años disponibles (excluir 2015 que no existe)
available_years = [y for y in range(START_YEAR, END_YEAR + 1) if y != 2015]

print(f"📅 Período de entrenamiento: {START_YEAR}-{END_YEAR}")
print(f"   • Total de períodos: {len(available_years) - 1} transiciones")
print(f"   • Validación separada: 2011-2016")
print()

# ==============================================================================
# 2. CARGAR Y PROCESAR MAPAS
# ==============================================================================
print("📂 Cargando mapas...")
maps = {}
for year in available_years:
    map_path = MAPS_DIR / f'{year}.npy'
    if map_path.exists():
        maps[year] = np.load(map_path)
        print(f"   ✓ {year}: {maps[year].shape}")
    else:
        print(f"   ✗ {year}: No encontrado")

print(f"\n✓ Cargados {len(maps)} mapas")
print()

# ==============================================================================
# 3. CALCULAR TRANSICIONES Y FEATURES
# ==============================================================================
print("🔄 Calculando transiciones y features espaciales...")
print()

all_features = []
all_targets = []
transition_count = 0

start_time = time.time()

for i, year in enumerate(sorted(maps.keys())[:-1]):
    next_year = sorted(maps.keys())[i + 1]
    
    print(f"  Período {year}→{next_year}:")
    
    # Mapas
    map_t0 = maps[year]
    map_t1 = maps[next_year]
    
    # Solo analizar celdas no-urbanas en t0 (candidatas a urbanizar)
    mask_candidates = (map_t0 == 0)
    n_candidates = np.sum(mask_candidates)
    
    # Target: transiciones a urbano
    transitions = (map_t0 == 0) & (map_t1 == 1)
    n_transitions = np.sum(transitions)
    
    print(f"    • Candidatas: {n_candidates:,}")
    print(f"    • Transiciones: {n_transitions:,} ({n_transitions/n_candidates*100:.2f}%)")
    
    # Calcular features espaciales
    period_start = time.time()
    features_dict = create_default_spatial_variables(map_t0.shape, map_t0)
    period_time = time.time() - period_start
    
    print(f"    • Features calculadas: {list(features_dict.keys())}")
    print(f"    • Tiempo: {period_time:.1f}s")
    
    # Extraer features solo de celdas candidatas
    features_flat = []
    for feature_name in sorted(features_dict.keys()):
        feature_values = features_dict[feature_name][mask_candidates]
        features_flat.append(feature_values)
    
    features_array = np.column_stack(features_flat)
    target_array = map_t1[mask_candidates]
    
    all_features.append(features_array)
    all_targets.append(target_array)
    
    transition_count += 1
    print()

# Consolidar datos
print("📦 Consolidando datos...")
X = np.vstack(all_features)
y = np.hstack(all_targets)

feature_names = sorted(features_dict.keys())

total_time = time.time() - start_time
print(f"✓ Datos preparados en {total_time:.1f}s ({total_time/60:.1f} min)")
print(f"  • Samples totales: {len(X):,}")
print(f"  • Features: {X.shape[1]} ({feature_names})")
print(f"  • Transiciones: {np.sum(y==1):,} ({np.sum(y==1)/len(y)*100:.3f}%)")
print(f"  • No-transiciones: {np.sum(y==0):,}")
print()

# ==============================================================================
# 4. ENTRENAR WoE
# ==============================================================================
print("🧮 Entrenando modelo WoE...")
print()

woe_calculator = WoECalculator(min_bin_size=100, max_bins=10)

training_start = time.time()

for i, feature_name in enumerate(feature_names):
    print(f"  [{i+1}/{len(feature_names)}] {feature_name}...", end=" ", flush=True)
    
    feature_start = time.time()
    result = woe_calculator.calculate_woe(
        variable=X[:, i],
        target=y,
        variable_name=feature_name,
        binning_method='quantile'
    )
    feature_time = time.time() - feature_start
    
    print(f"✓ (IV={result.iv_total:.4f}, {len(result.bins)-1} bins, {feature_time:.1f}s)")

training_time = time.time() - training_start

print()
print(f"✓ Entrenamiento completado en {training_time:.1f}s ({training_time/60:.1f} min)")
print()

# ==============================================================================
# 5. REPORTE WoE
# ==============================================================================
print("="*80)
print("📊 REPORTE WoE")
print("="*80)
print()

for feature_name in feature_names:
    result = woe_calculator.woe_results[feature_name]
    
    iv_strength = "Muy fuerte" if result.iv_total >= 0.5 else \
                  "Fuerte" if result.iv_total >= 0.3 else \
                  "Medio" if result.iv_total >= 0.1 else \
                  "Débil" if result.iv_total >= 0.02 else "Muy débil"
    
    print(f"📈 {feature_name}")
    print(f"   • IV: {result.iv_total:.4f} ({iv_strength})")
    print(f"   • Bins: {len(result.bins)-1}")
    print(f"   • WoE range: [{result.woe_values.min():.3f}, {result.woe_values.max():.3f}]")
    print()

# ==============================================================================
# 6. GUARDAR MODELO
# ==============================================================================
print("="*80)
print("💾 GUARDANDO MODELO")
print("="*80)
print()

# Metadata
metadata = {
    'trained_years': f'{START_YEAR}-{END_YEAR}',
    'n_periods': transition_count,
    'total_transitions': int(np.sum(y == 1)),
    'total_samples': len(y),
    'variables': feature_names,
    'n_bins': 10,
    'min_bin_size': 100,
    'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'training_time_seconds': int(training_time),
    'validation_reserved': '2011-2016'
}

# Guardar
data_to_save = {
    'woe_calculator': woe_calculator,
    'metadata': metadata
}

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"✓ Modelo guardado: {OUTPUT_FILE}")
print(f"  • Tamaño: {Path(OUTPUT_FILE).stat().st_size / 1024:.1f} KB")
print()

# Guardar metadata legible
metadata_file = OUTPUT_FILE.replace('.pkl', '_metadata.txt')
with open(metadata_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("WoE MODEL METADATA (1984-2010)\n")
    f.write("="*80 + "\n\n")
    for key, value in metadata.items():
        f.write(f"{key}: {value}\n")
    f.write("\n")
    f.write("FEATURE INFORMATION VALUES:\n")
    f.write("-"*80 + "\n")
    for feature_name in feature_names:
        result = woe_calculator.woe_results[feature_name]
        f.write(f"{feature_name:25s} IV={result.iv_total:.4f}\n")

print(f"✓ Metadata guardada: {metadata_file}")
print()

# Crear backup
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
backup_dir = Path('data/backups')
backup_dir.mkdir(exist_ok=True)

backup_file = backup_dir / f'woe_pooled_1984_2010_{timestamp}.pkl'
with open(backup_file, 'wb') as f:
    pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"✓ Backup creado: {backup_file}")
print()

print("="*80)
print("✅ ENTRENAMIENTO COMPLETADO")
print("="*80)
print()
print(f"🎯 Modelo listo para validación 2011→2016")
print(f"   Usar con threshold calibrado (recomendado: 0.75)")
print()
