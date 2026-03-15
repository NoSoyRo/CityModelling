#!/usr/bin/env python3
"""
Validación Quinquenal 2011 → 2016
==================================
Predicción a 5 años usando WoE pooled (1984-2010).
Evalúa si el modelo captura tendencias estructurales a mediano plazo.

VERSIÓN MEJORADA:
- WoE entrenado SOLO con 1984-2010 (validación pura)
- Threshold ajustado a 0.75 (más restrictivo, evita sobre-predicción)
"""

import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
import time
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Importar módulos del proyecto
import sys
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from tesis_ac.ca.rules import create_default_spatial_variables

# ============================================================================
# FUNCIÓN AUXILIAR: VISUALIZACIÓN
# ============================================================================

def visualize_prediction(year_prev, year_current, map_prev, map_current, output_path):
    """
    Crea visualización de 3 columnas: anterior, actual, diferencia.
    Similar a la imagen de validación de consistencia.
    """
    # Calcular cambios
    change_map = np.zeros_like(map_current, dtype=int)
    change_map[(map_prev == 0) & (map_current == 1)] = 1  # Nuevas urbanas (rojo)
    change_map[(map_prev == 1) & (map_current == 0)] = -1  # Pérdidas urbanas (azul)
    
    # Estadísticas
    urban_prev = np.sum(map_prev == 1)
    urban_curr = np.sum(map_current == 1)
    new_urban = np.sum(change_map == 1)
    lost_urban = np.sum(change_map == -1)
    net_change = new_urban - lost_urban
    pct_change = 100 * net_change / urban_prev if urban_prev > 0 else 0
    pct_urban_curr = 100 * urban_curr / map_current.size
    
    # Crear figura
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Mapa anterior
    ax1 = axes[0]
    # Verde=no urbano, Rojo=urbano
    img_prev = np.zeros((*map_prev.shape, 3))
    img_prev[map_prev == 0] = [0, 0.5, 0]  # Verde oscuro
    img_prev[map_prev == 1] = [0.8, 0, 0]  # Rojo
    ax1.imshow(img_prev, aspect='auto')
    ax1.set_title(f'{year_prev}\n{urban_prev:,} urbano', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Mapa actual
    ax2 = axes[1]
    img_curr = np.zeros((*map_current.shape, 3))
    img_curr[map_current == 0] = [0, 0.5, 0]  # Verde oscuro
    img_curr[map_current == 1] = [0.8, 0, 0]  # Rojo
    ax2.imshow(img_curr, aspect='auto')
    ax2.set_title(f'{year_current}\n{urban_curr:,} urbano', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Mapa de diferencias
    ax3 = axes[2]
    img_diff = np.ones((*change_map.shape, 3))  # Blanco por defecto
    img_diff[change_map == 1] = [0.6, 0, 0]    # Rojo oscuro = nuevas urbanas
    img_diff[change_map == -1] = [0, 0, 0.6]   # Azul oscuro = pérdidas
    ax3.imshow(img_diff, aspect='auto')
    
    # Título con estadísticas
    sign = '+' if net_change >= 0 else ''
    title = f'Cambio: {pct_change:.1f}%\n'
    title += f'σ=1: {new_urban:,}, σ=0: {lost_urban:,}, Δ={sign}{net_change:,}, {pct_change:+.1f}%'
    ax3.set_title(title, fontsize=11, fontweight='bold', color='green' if net_change >= 0 else 'red')
    ax3.axis('off')
    
    # Leyenda
    red_patch = mpatches.Patch(color=[0.6, 0, 0], label='Nuevas urbanas')
    blue_patch = mpatches.Patch(color=[0, 0, 0.6], label='Pérdidas urbanas')
    ax3.legend(handles=[red_patch, blue_patch], loc='upper right', fontsize=9)
    
    # Título general
    fig.suptitle(f'Predicción {year_prev}→{year_current} (Validación Quinquenal)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


print('=' * 80)
print('🎯 VALIDACIÓN QUINQUENAL 2011 → 2016')
print('=' * 80)
print()

# ============================================================================
# 1. CARGAR DATOS
# ============================================================================

print('📂 Cargando datos...')

woe_path = Path('data/processed/woe_pooled_1984_2010.pkl')
print(f'  • WoE: {woe_path}')
with open(woe_path, 'rb') as f:
    woe_data = pickle.load(f)
    woe_calculator = woe_data['woe_calculator']
    metadata = woe_data['metadata']

print(f'    ✓ Entrenado con {metadata["n_periods"]} períodos')
print(f'    ✓ Variables: {len(metadata["variables"])} features')
print(f'    ✓ Transiciones: {metadata["total_transitions"]:,}')

CALIBRATED_THRESHOLD = 0.75
params_path = Path('data/processed/ga_calibrated_params.json')
print(f'  • Parámetros AC: {params_path}')
with open(params_path, 'r') as f:
    params_data = json.load(f)
    ac_params_base = params_data['configuration']
ac_params = {
    'threshold': CALIBRATED_THRESHOLD,
    'neighbor_weight': ac_params_base['neighbor_weight'],
    'distance_weight': ac_params_base['distance_weight']
}
print(f'    ✓ Parámetros óptimos cargados')
print(f'    ✓ threshold: {ac_params["threshold"]:.4f} (⚡ CALIBRADO - evita sobre-predicción)')
print(f'    ✓ neighbor_weight: {ac_params["neighbor_weight"]:.4f}')
print(f'    ✓ distance_weight: {ac_params["distance_weight"]:.4f}')

maps_dir = Path('data/processed/standardized_maps')
print(f'  • Mapas: {maps_dir}')
map_2011 = np.load(maps_dir / '2011.npy')
map_2016 = np.load(maps_dir / '2016.npy')
print(f'    ✓ 2011: {map_2011.shape}')
print(f'    ✓ 2016: {map_2016.shape}')
print()

# ============================================================================
# 2. SIMULAR CON AUTÓMATA CELULAR (5 PASOS) - RECALCULANDO FEATURES
# ============================================================================

print('🤖 Ejecutando simulación AC (5 pasos)...')
print(f'  Parámetros:')
print(f'    • threshold: {ac_params["threshold"]:.4f}')
print(f'    • neighbor_weight: {ac_params["neighbor_weight"]:.4f}')
print(f'    • distance_weight: {ac_params["distance_weight"]:.4f}')
print()
print('  Ponderación WoE por Information Value (IV):')
for var_name in sorted(woe_calculator.woe_results.keys()):
    iv = woe_calculator.woe_results[var_name].iv_total
    total_iv = sum(r.iv_total for r in woe_calculator.woe_results.values())
    weight = iv / total_iv
    print(f'    • {var_name:25s}: IV={iv:.4f} (peso={weight:.1%})')
print()
print('  NOTA: Recalculando features espaciales en cada paso para máxima precisión')
print()

output_dir = Path('data/processed/validation_quinquenal_2011_2016_v3_weighted')
output_dir.mkdir(exist_ok=True)
maps_dir = output_dir / 'yearly_predictions'
maps_dir.mkdir(exist_ok=True)
viz_dir = output_dir / 'visualizations'
viz_dir.mkdir(exist_ok=True)

# Guardar mapa inicial
np.save(maps_dir / '2011_initial.npy', map_2011)
print(f'💾 Guardado: {maps_dir / "2011_initial.npy"}')
print()

# Estado inicial
current_state = map_2011.copy()
prev_state = map_2011.copy()  # Para visualizaciones

# Simular 5 pasos (2011→2016)
print('  Paso  |  Urbano  | Nuevas | Tasa    | Tiempo')
print('  ------|----------|--------|---------|--------')

for step in range(1, 6):
    step_start = time.time()
    year = 2011 + step
    
    # ==== RECALCULAR TODAS LAS FEATURES ESPACIALES ====
    # Esto es necesario porque el mapa cambió en el paso anterior
    features_current = create_default_spatial_variables(current_state.shape, current_state)
    
    # Transformar cada feature a WoE scores con PONDERACIÓN por IV
    woe_maps = {}
    iv_weights = {}
    
    for feature_name, feature_map in features_current.items():
        # Aplicar transformación WoE usando bins pre-calculados
        woe_transformed = woe_calculator.apply_woe_transform(
            feature_map.flatten(), 
            feature_name
        )
        woe_maps[feature_name] = woe_transformed.reshape(current_state.shape)
        
        # Obtener Information Value como peso
        iv_weights[feature_name] = woe_calculator.woe_results[feature_name].iv_total
    
    # Normalizar pesos por IV
    total_iv = sum(iv_weights.values())
    normalized_weights = {k: v/total_iv for k, v in iv_weights.items()}
    
    # Combinar scores WoE PONDERADOS (suma ponderada de evidencias)
    total_woe = np.zeros_like(current_state, dtype=float)
    for feature_name, woe_map in woe_maps.items():
        total_woe += normalized_weights[feature_name] * woe_map
    
    # Convertir WoE combinado a probabilidad
    prob_map = 1 / (1 + np.exp(-total_woe))
    
    # ==== CALCULAR VECINDAD ====
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0  # Excluir celda central
    neighbor_count = convolve(current_state.astype(float), kernel, mode='constant', cval=0)
    neighbor_density = neighbor_count / 8.0  # Normalizar
    
    # ==== COMBINAR PROBABILIDADES ====
    # Usar probabilidad WoE directamente (ya incluye todas las features espaciales)
    # y agregar influencia de vecindad inmediata
    combined_prob = prob_map + ac_params['neighbor_weight'] * neighbor_density
    
    # Normalizar para mantener en rango [0,1]
    combined_prob = np.clip(combined_prob, 0, 1)
    
    # ==== APLICAR TRANSICIONES ====
    # Aplicar threshold y estocasticidad
    random_values = np.random.random(current_state.shape)
    transitions = (combined_prob > ac_params['threshold']) & (random_values < combined_prob)
    
    # Actualizar estado (solo celdas no-urbanas pueden transicionar)
    new_urban = transitions & (current_state == 0)
    current_state[new_urban] = 1
    
    # ==== ESTADÍSTICAS ====
    urban_count = np.sum(current_state == 1)
    urban_prev = np.sum(map_2011 == 1) if step == 1 else urban_count_prev
    new_urban_count = urban_count - urban_prev
    urban_pct = 100 * urban_count / current_state.size
    
    step_elapsed = time.time() - step_start
    print(f'  {year} | {urban_count:8d} | {new_urban_count:6d} | {urban_pct:5.2f}% | {step_elapsed:6.1f}s')
    
    # ==== GUARDAR MAPA PREDICHO ====
    map_filename = maps_dir / f'{year}_predicted.npy'
    np.save(map_filename, current_state)
    
    # ==== CREAR VISUALIZACIÓN ====
    prev_map = map_2011 if step == 1 else prev_state
    prev_year = 2011 + step - 1
    viz_filename = viz_dir / f'{prev_year}_to_{year}.png'
    visualize_prediction(prev_year, year, prev_map, current_state, viz_filename)
    print(f'       💾 Guardado: {map_filename.name} | 📊 Viz: {viz_filename.name}')
    
    # Guardar para siguiente iteración
    prev_state = current_state.copy()
    urban_count_prev = urban_count

# Mapa final predicho
predicted_2016 = current_state

print()
print('✅ Simulación completada')
print(f'   📁 Mapas anuales guardados en: {maps_dir}')
print(f'   📊 Visualizaciones en: {viz_dir}')
print()

# ============================================================================
# 3. CALCULAR MÉTRICAS DE VALIDACIÓN
# ============================================================================

print('📊 Calculando métricas de validación...')
print()

# Preparar datos
observed = map_2016.flatten()
predicted = predicted_2016.flatten()

# ----- Matriz de Confusión -----
TP = np.sum((predicted == 1) & (observed == 1))
TN = np.sum((predicted == 0) & (observed == 0))
FP = np.sum((predicted == 1) & (observed == 0))
FN = np.sum((predicted == 0) & (observed == 1))

print('🔢 Matriz de Confusión:')
print(f'  TP (hit urbano):     {TP:8d}')
print(f'  TN (hit no-urbano):  {TN:8d}')
print(f'  FP (falsa alarma):   {FP:8d}')
print(f'  FN (omisión):        {FN:8d}')
print()

# ----- Métricas Píxel a Píxel -----
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# IoU
iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

# Kappa
po = accuracy
pe = ((TP + FP) * (TP + FN) + (TN + FN) * (TN + FP)) / ((TP + TN + FP + FN) ** 2)
kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0

print('✅ Métricas Píxel-a-Píxel:')
print(f'  Accuracy:   {accuracy:.4f}')
print(f'  IoU:        {iou:.4f}')
print(f'  Precision:  {precision:.4f}')
print(f'  Recall:     {recall:.4f}')
print(f'  F1-Score:   {f1:.4f}')
print(f'  Kappa:      {kappa:.4f}')
print()

# ----- Figure of Merit (FoM) -----
# Detectar transiciones
trans_obs = (map_2016 == 1) & (map_2011 == 0)  # Nuevas urbanas observadas
trans_pred = (predicted_2016 == 1) & (map_2011 == 0)  # Nuevas urbanas predichas

B = np.sum(trans_obs & trans_pred)  # Hits
C = np.sum(trans_obs & ~trans_pred)  # Misses
A = np.sum(~trans_obs & trans_pred)  # False alarms

fom = B / (A + B + C) if (A + B + C) > 0 else 0

print('🎯 Figure of Merit (FoM):')
print(f'  B (hits):          {B:8d}')
print(f'  C (misses):        {C:8d}')
print(f'  A (false alarms):  {A:8d}')
print(f'  FoM:               {fom:.4f}')
print()

# ----- Métricas de Cambio -----
change_precision = B / (A + B) if (A + B) > 0 else 0
change_recall = B / (B + C) if (B + C) > 0 else 0
change_f1 = 2 * B / (2 * B + A + C) if (2 * B + A + C) > 0 else 0

print('🔄 Métricas de Detección de Cambio:')
print(f'  Change Precision:  {change_precision:.4f}')
print(f'  Change Recall:     {change_recall:.4f}')
print(f'  Change F1:         {change_f1:.4f}')
print()

# ----- Descomposición de Error -----
quantity_disagreement = abs(FP - FN)
allocation_disagreement = 2 * min(FP, FN)

print('📐 Descomposición de Error:')
print(f'  Quantity Disagreement:    {quantity_disagreement:8d}')
print(f'  Allocation Disagreement:  {allocation_disagreement:8d}')
print()

# ----- Estadísticas de Crecimiento -----
urban_2011 = np.sum(map_2011 == 1)
urban_2016_obs = np.sum(map_2016 == 1)
urban_2016_pred = np.sum(predicted_2016 == 1)

growth_obs = urban_2016_obs - urban_2011
growth_pred = urban_2016_pred - urban_2011

growth_rate_obs = 100 * growth_obs / urban_2011
growth_rate_pred = 100 * growth_pred / urban_2011

print('📈 Crecimiento Urbano:')
print(f'  2011 (inicial):      {urban_2011:8d} píxeles')
print(f'  2016 observado:      {urban_2016_obs:8d} píxeles (+{growth_obs:d}, +{growth_rate_obs:.1f}%)')
print(f'  2016 predicho:       {urban_2016_pred:8d} píxeles (+{growth_pred:d}, +{growth_rate_pred:.1f}%)')
print(f'  Error cantidad:      {abs(growth_obs - growth_pred):8d} píxeles')
print()

# ============================================================================
# 4. GUARDAR RESULTADOS
# ============================================================================

print('💾 Guardando resultados...')

# Crear directorio (usar mismo que arriba)
output_dir = Path('data/processed/validation_quinquenal_2011_2016_v3_weighted')
output_dir.mkdir(exist_ok=True)

# Guardar mapa predicho
np.save(output_dir / 'predicted_2016.npy', predicted_2016)
print(f'  ✓ Mapa predicho: {output_dir / "predicted_2016.npy"}')

# Guardar mapa de probabilidades
np.save(output_dir / 'probabilities_2011.npy', prob_map)
print(f'  ✓ Probabilidades: {output_dir / "probabilities_2011.npy"}')

# Guardar métricas
results = {
    'metadata': {
        'validation_period': '2011→2016',
        'training_period': '1984-2014',
        'n_steps': 5,
        'timestamp': datetime.now().isoformat()
    },
    'ac_parameters': {
        'threshold': float(ac_params['threshold']),
        'neighbor_weight': float(ac_params['neighbor_weight']),
        'distance_weight': float(ac_params['distance_weight']),
        'source': ac_params.get('source', 'unknown')
    },
    'confusion_matrix': {
        'TP': int(TP),
        'TN': int(TN),
        'FP': int(FP),
        'FN': int(FN)
    },
    'pixel_metrics': {
        'accuracy': float(accuracy),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'kappa': float(kappa)
    },
    'fom_metrics': {
        'fom': float(fom),
        'B_hits': int(B),
        'C_misses': int(C),
        'A_false_alarms': int(A)
    },
    'change_metrics': {
        'change_precision': float(change_precision),
        'change_recall': float(change_recall),
        'change_f1': float(change_f1)
    },
    'error_decomposition': {
        'quantity_disagreement': int(quantity_disagreement),
        'allocation_disagreement': int(allocation_disagreement)
    },
    'growth_statistics': {
        'urban_2011': int(urban_2011),
        'urban_2016_observed': int(urban_2016_obs),
        'urban_2016_predicted': int(urban_2016_pred),
        'growth_observed': int(growth_obs),
        'growth_predicted': int(growth_pred),
        'growth_rate_observed_pct': float(growth_rate_obs),
        'growth_rate_predicted_pct': float(growth_rate_pred)
    }
}

results_path = output_dir / 'validation_results.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f'  ✓ Resultados: {results_path}')
print()

# ============================================================================
# 5. RESUMEN FINAL
# ============================================================================

print('=' * 80)
print('✅ VALIDACIÓN QUINQUENAL COMPLETADA')
print('=' * 80)
print()
print('📊 Resumen de Desempeño:')
print(f'  • Accuracy:         {accuracy:.4f}')
print(f'  • IoU:              {iou:.4f}')
print(f'  • F1-Score:         {f1:.4f}')
print(f'  • Kappa:            {kappa:.4f}')
print(f'  • FoM:              {fom:.4f}')
print(f'  • Change F1:        {change_f1:.4f}')
print()
print('🎯 Interpretación:')
if fom > 0.20:
    print('  ✅ FoM excelente para validación quinquenal')
elif fom > 0.15:
    print('  ✅ FoM bueno - captura patrones estructurales')
elif fom > 0.10:
    print('  ⚠️  FoM moderado - revisa parámetros AC')
else:
    print('  ❌ FoM bajo - requiere calibración')

if accuracy > 0.90:
    print('  ✅ Accuracy alto - buena concordancia global')

if kappa > 0.60:
    print('  ✅ Kappa sustancial - acuerdo significativo')
elif kappa > 0.40:
    print('  ✅ Kappa moderado - acuerdo aceptable')

print()
print(f'📁 Resultados guardados en: {output_dir}')
print('=' * 80)
