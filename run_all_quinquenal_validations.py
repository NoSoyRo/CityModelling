#!/usr/bin/env python3
"""
Ejecuta validaciones quinquenales para múltiples períodos:
  2011→2016, 2012→2017, 2013→2018, 2014→2019, 2015→2020

Usa los mismos parámetros calibrados (threshold=0.75, WoE pooled 1984-2010).
Guarda resultados en data/processed/validation_quinquenal_YYYY_ZZZZ_v3_weighted/
y un resumen en data/processed/quinquenal_all_periods_summary.json

Para ver progreso en tiempo real:
  python -u run_all_quinquenal_validations.py 2>&1 | tee data/processed/quinquenal_run_log.txt
  tail -f data/processed/quinquenal_run_log.txt
"""

import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
import time
from scipy.ndimage import convolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import sys
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from tesis_ac.ca.rules import create_default_spatial_variables

# Forzar salida sin buffer para que tail -f muestre progreso en tiempo real
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# ============================================================================
# LOGGING (flush para ver output en tiempo real con tail -f)
# ============================================================================

def log(msg: str):
    """Imprime y hace flush para que tail -f muestre el output inmediatamente."""
    print(msg, flush=True)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

PERIODS = [
    (2011, 2016),
    (2012, 2017),
    (2013, 2018),
    (2014, 2019),
    (2015, 2020),
]

CALIBRATED_THRESHOLD = 0.75
WOE_PATH = Path('data/processed/woe_pooled_1984_2010.pkl')
PARAMS_PATH = Path('data/processed/ga_calibrated_params.json')
MAPS_DIR = Path('data/processed/standardized_maps')


def visualize_prediction(year_prev, year_current, map_prev, map_current, output_path):
    """Crea visualización de 3 columnas: anterior, actual, diferencia."""
    change_map = np.zeros_like(map_current, dtype=int)
    change_map[(map_prev == 0) & (map_current == 1)] = 1
    change_map[(map_prev == 1) & (map_current == 0)] = -1

    urban_prev = np.sum(map_prev == 1)
    urban_curr = np.sum(map_current == 1)
    new_urban = np.sum(change_map == 1)
    lost_urban = np.sum(change_map == -1)
    net_change = new_urban - lost_urban
    pct_change = 100 * net_change / urban_prev if urban_prev > 0 else 0
    pct_urban_curr = 100 * urban_curr / map_current.size

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    img_prev = np.zeros((*map_prev.shape, 3))
    img_prev[map_prev == 0] = [0, 0.5, 0]
    img_prev[map_prev == 1] = [0.8, 0, 0]
    axes[0].imshow(img_prev, aspect='auto')
    axes[0].set_title(f'{year_prev}\n{urban_prev:,} urbano', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    img_curr = np.zeros((*map_current.shape, 3))
    img_curr[map_current == 0] = [0, 0.5, 0]
    img_curr[map_current == 1] = [0.8, 0, 0]
    axes[1].imshow(img_curr, aspect='auto')
    axes[1].set_title(f'{year_current}\n{urban_curr:,} urbano', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    img_diff = np.ones((*change_map.shape, 3))
    img_diff[change_map == 1] = [0.6, 0, 0]
    img_diff[change_map == -1] = [0, 0, 0.6]
    axes[2].imshow(img_diff, aspect='auto')
    sign = '+' if net_change >= 0 else ''
    title = f'Cambio: {pct_change:.1f}%\nσ=1: {new_urban:,}, σ=0: {lost_urban:,}, Δ={sign}{net_change:,}'
    axes[2].set_title(title, fontsize=11, fontweight='bold')
    axes[2].axis('off')

    red_patch = mpatches.Patch(color=[0.6, 0, 0], label='Nuevas urbanas')
    blue_patch = mpatches.Patch(color=[0, 0, 0.6], label='Pérdidas urbanas')
    axes[2].legend(handles=[red_patch, blue_patch], loc='upper right', fontsize=9)

    fig.suptitle(f'Predicción {year_prev}→{year_current} (Validación Quinquenal)', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_summary(map_start, map_end_obs, map_end_pred, start_year, end_year, output_path, TP, FP, FN):
    """Crea figura resumen: inicio | final observado | final predicho | diferencia (TP/FP/FN)."""
    def to_rgb(m):
        rgb = np.zeros((*m.shape, 3))
        rgb[m == 0] = [0.18, 0.49, 0.20]
        rgb[m == 1] = [0.78, 0.08, 0.08]
        return rgb

    diff = np.zeros_like(map_end_obs, dtype=int)
    diff[(map_end_obs == 0) & (map_end_pred == 1)] = 1   # FP
    diff[(map_end_obs == 1) & (map_end_pred == 0)] = 2   # FN
    diff[(map_end_obs == 1) & (map_end_pred == 1)] = 3   # TP
    diff_rgb = np.ones((*diff.shape, 3))
    diff_rgb[diff == 1] = [0.85, 0.33, 0.10]
    diff_rgb[diff == 2] = [0.13, 0.47, 0.71]
    diff_rgb[diff == 3] = [0.17, 0.63, 0.17]

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    axes[0].imshow(to_rgb(map_start), aspect='auto')
    axes[0].set_title(f'{start_year}\n(Inicio)\n{np.sum(map_start==1):,} px', fontsize=11, fontweight='bold')
    axes[0].axis('off')
    axes[1].imshow(to_rgb(map_end_obs), aspect='auto')
    axes[1].set_title(f'{end_year} Observado\n(Ground truth)\n{np.sum(map_end_obs==1):,} px', fontsize=11, fontweight='bold')
    axes[1].axis('off')
    axes[2].imshow(to_rgb(map_end_pred), aspect='auto')
    axes[2].set_title(f'{end_year} Predicho\n(AC WoE, 5 pasos)\n{np.sum(map_end_pred==1):,} px', fontsize=11, fontweight='bold')
    axes[2].axis('off')
    axes[3].imshow(diff_rgb, aspect='auto')
    axes[3].set_title('Diferencia\n(Obs vs Pred)', fontsize=11, fontweight='bold')
    axes[3].axis('off')
    p_tp = mpatches.Patch(color=[0.17, 0.63, 0.17], label=f'TP = {TP:,}')
    p_fp = mpatches.Patch(color=[0.85, 0.33, 0.10], label=f'FP = {FP:,}')
    p_fn = mpatches.Patch(color=[0.13, 0.47, 0.71], label=f'FN = {FN:,}')
    axes[3].legend(handles=[p_tp, p_fp, p_fn], loc='lower right', fontsize=8.5, framealpha=0.9)
    fig.suptitle(f'Simulación quinquenal AC-WoE: {start_year} → {end_year}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()


def run_single_validation(start_year: int, end_year: int, woe_calculator, ac_params: dict) -> dict:
    """Ejecuta validación quinquenal para un período y retorna métricas."""
    log(f'  [1/6] Cargando mapas {start_year}.npy y {end_year}.npy...')
    map_start = np.load(MAPS_DIR / f'{start_year}.npy')
    map_end = np.load(MAPS_DIR / f'{end_year}.npy')
    log(f'  [1/6] ✓ Mapas cargados ({map_start.shape[0]}x{map_start.shape[1]})')

    output_dir = Path(f'data/processed/validation_quinquenal_{start_year}_{end_year}_v3_weighted')
    output_dir.mkdir(exist_ok=True)
    maps_dir = output_dir / 'yearly_predictions'
    maps_dir.mkdir(exist_ok=True)
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)

    current_state = map_start.copy()
    prev_state = map_start.copy()
    urban_count_prev = np.sum(map_start == 1)

    for step in range(1, 6):
        step_start = time.time()
        year = start_year + step
        log(f'  [2/6] Paso {step}/5: simulando {start_year + step - 1}→{year}...')
        features_current = create_default_spatial_variables(current_state.shape, current_state)
        woe_maps = {}
        total_iv = sum(r.iv_total for r in woe_calculator.woe_results.values())

        for feature_name, feature_map in features_current.items():
            woe_transformed = woe_calculator.apply_woe_transform(
                feature_map.flatten(), feature_name
            )
            woe_maps[feature_name] = woe_transformed.reshape(current_state.shape)
            iv = woe_calculator.woe_results[feature_name].iv_total
            woe_maps[feature_name] *= (iv / total_iv)

        total_woe = np.zeros_like(current_state, dtype=float)
        for wm in woe_maps.values():
            total_woe += wm

        prob_map = 1 / (1 + np.exp(-total_woe))

        kernel = np.ones((3, 3))
        kernel[1, 1] = 0
        neighbor_count = convolve(current_state.astype(float), kernel, mode='constant', cval=0)
        neighbor_density = neighbor_count / 8.0

        combined_prob = prob_map + ac_params['neighbor_weight'] * neighbor_density
        combined_prob = np.clip(combined_prob, 0, 1)

        random_values = np.random.random(current_state.shape)
        transitions = (combined_prob > ac_params['threshold']) & (random_values < combined_prob)
        new_urban = transitions & (current_state == 0)
        current_state[new_urban] = 1

        prev_map = map_start if step == 1 else prev_state
        prev_year = start_year + step - 1
        viz_filename = viz_dir / f'{prev_year}_to_{year}.png'
        visualize_prediction(prev_year, year, prev_map, current_state, viz_filename)

        np.save(maps_dir / f'{year}_predicted.npy', current_state)
        elapsed = time.time() - step_start
        urban_now = np.sum(current_state == 1)
        log(f'  [2/6] ✓ Paso {step}/5 listo: {year} ({urban_now:,} px urbano) en {elapsed:.1f}s')

        prev_state = current_state.copy()
        urban_count_prev = np.sum(current_state == 1)

    log(f'  [3/6] Calculando métricas (TP, TN, FP, FN, FoM)...')
    predicted = current_state
    observed = map_end.flatten()
    pred_flat = predicted.flatten()

    TP = int(np.sum((pred_flat == 1) & (observed == 1)))
    TN = int(np.sum((pred_flat == 0) & (observed == 0)))
    FP = int(np.sum((pred_flat == 1) & (observed == 0)))
    FN = int(np.sum((pred_flat == 0) & (observed == 1)))

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    pe = ((TP + FP) * (TP + FN) + (TN + FN) * (TN + FP)) / ((TP + TN + FP + FN) ** 2)
    kappa = (accuracy - pe) / (1 - pe) if (1 - pe) > 0 else 0

    trans_obs = (map_end == 1) & (map_start == 0)
    trans_pred = (predicted == 1) & (map_start == 0)
    B = int(np.sum(trans_obs & trans_pred))
    C = int(np.sum(trans_obs & ~trans_pred))
    A = int(np.sum(~trans_obs & trans_pred))
    fom = B / (A + B + C) if (A + B + C) > 0 else 0

    urban_start = int(np.sum(map_start == 1))
    urban_end_obs = int(np.sum(map_end == 1))
    urban_end_pred = int(np.sum(predicted == 1))

    np.save(output_dir / f'predicted_{end_year}.npy', predicted)

    log(f'  [4/6] Generando mapa resumen (inicio|obs|pred|diff)...')
    summary_path = output_dir / f'summary_{start_year}_to_{end_year}.png'
    visualize_summary(map_start, map_end, predicted, start_year, end_year, summary_path, TP, FP, FN)
    log(f'  [4/6] ✓ {summary_path.name}')

    log(f'  [5/6] Guardando validation_results.json...')
    results = {
        'metadata': {'validation_period': f'{start_year}→{end_year}', 'timestamp': datetime.now().isoformat()},
        'confusion_matrix': {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN},
        'pixel_metrics': {'accuracy': accuracy, 'iou': iou, 'precision': precision, 'recall': recall, 'f1_score': f1, 'kappa': kappa},
        'fom_metrics': {'fom': fom, 'B_hits': B, 'C_misses': C, 'A_false_alarms': A},
        'growth_statistics': {
            'urban_start': urban_start,
            'urban_end_observed': urban_end_obs,
            'urban_end_predicted': urban_end_pred,
            'growth_observed': urban_end_obs - urban_start,
            'growth_predicted': urban_end_pred - urban_start,
        }
    }
    with open(output_dir / 'validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    log(f'  [5/6] ✓ validation_results.json (FoM={fom:.3f})')

    log(f'  [6/6] ✓ Período {start_year}→{end_year} completado.')
    return {
        'period': f'{start_year}→{end_year}',
        'accuracy': accuracy,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'kappa': kappa,
        'fom': fom,
        'urban_start': urban_start,
        'urban_end_observed': urban_end_obs,
        'urban_end_predicted': urban_end_pred,
    }


def main():
    log('=' * 80)
    log('🎯 VALIDACIONES QUINQUENALES MÚLTIPLES')
    log('   Períodos: 2011→2016, 2012→2017, 2013→2018, 2014→2019, 2015→2020')
    log('=' * 80)

    log('Cargando WoE y parámetros AC...')
    with open(WOE_PATH, 'rb') as f:
        woe_data = pickle.load(f)
        woe_calculator = woe_data['woe_calculator']
    log('✓ WoE cargado')

    with open(PARAMS_PATH, 'r') as f:
        params_data = json.load(f)
        ac_params_base = params_data['configuration']
    ac_params = {
        'threshold': CALIBRATED_THRESHOLD,
        'neighbor_weight': ac_params_base['neighbor_weight'],
        'distance_weight': ac_params_base['distance_weight']
    }
    log('✓ Parámetros AC cargados (threshold=0.75)')
    log('')

    all_results = []
    for idx, (start_year, end_year) in enumerate(PERIODS, 1):
        log(f'')
        log(f'📂 [{idx}/5] Validación {start_year}→{end_year}...')
        res = run_single_validation(start_year, end_year, woe_calculator, ac_params)
        all_results.append(res)
        log(f'   → FoM={res["fom"]:.3f}  Accuracy={res["accuracy"]:.3f}  F1={res["f1_score"]:.3f}')

    log('')
    log('Guardando resumen consolidado...')
    summary_path = Path('data/processed/quinquenal_all_periods_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'periods': all_results,
            'ac_parameters': ac_params,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    log(f'✅ Resumen guardado en: {summary_path}')
    log('=' * 80)
    log('✅ TODAS LAS VALIDACIONES COMPLETADAS')


if __name__ == '__main__':
    main()
