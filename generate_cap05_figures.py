#!/usr/bin/env python3
"""
Genera todas las figuras necesarias para el Capítulo 5 de la tesis.

Figuras generadas:
  1. urban_growth_timeline.png    — evolución urbana 1984-2020
  2. woe_weights.png              — Information Values y WoE ranges por variable
  3. quinquenal_2011_to_2012.png  — mapa comparativo año 1 (copia de visualizaciones)
  4. quinquenal_2012_to_2013.png  — mapa comparativo año 2
  5. quinquenal_2013_to_2014.png  — mapa comparativo año 3
  6. quinquenal_2014_to_2015.png  — mapa comparativo año 4
  7. quinquenal_2015_to_2016.png  — mapa comparativo año 5
  8. ac_simulation_quinquenal.png — resumen visual de la simulación completa
  9. validation_metrics.png       — dashboard de métricas de validación
"""

import numpy as np
import pickle
import json
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import sys

sys.path.insert(0, 'src')

# ── Rutas ────────────────────────────────────────────────────────────────────
MAPS_DIR   = Path('data/processed/standardized_maps')
WOE_PKL    = Path('data/processed/woe_pooled_1984_2010.pkl')
QUINQ_DIR  = Path('data/processed/validation_quinquenal_2011_2016_v3_weighted')
VAL_JSON   = QUINQ_DIR / 'validation_results.json'
FIGS_OUT   = Path('report/tesis/tesis_indice_nuevo/caps_larraga/figures')
FIGS_OUT.mkdir(parents=True, exist_ok=True)
(FIGS_OUT / 'comparisons').mkdir(exist_ok=True)

print('=' * 65)
print('GENERANDO FIGURAS PARA CAPÍTULO 5')
print('=' * 65)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 1: Urban Growth Timeline 1984-2020
# ─────────────────────────────────────────────────────────────────────────────
print('\n[1/6] urban_growth_timeline.png ...')

years = sorted([int(p.stem) for p in MAPS_DIR.glob('*.npy')
                if p.stem.isdigit()])
urban_counts = []
for y in years:
    m = np.load(MAPS_DIR / f'{y}.npy')
    urban_counts.append(int(np.sum(m == 1)))

fig, ax = plt.subplots(figsize=(12, 5))

# Área bajo la curva
ax.fill_between(years, urban_counts, alpha=0.15, color='#c0392b')
ax.plot(years, urban_counts, '-o', color='#c0392b', lw=2, ms=5, zorder=3)

# Puntos clave anotados
key_years = {1984: None, 1990: None, 2000: None, 2010: None, 2014: None, 2020: None}
for y in key_years:
    if y in years:
        idx = years.index(y)
        ax.annotate(f'{y}\n{urban_counts[idx]:,}',
                    xy=(y, urban_counts[idx]),
                    xytext=(0, 14), textcoords='offset points',
                    ha='center', fontsize=8.5, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

# Período de validación sombreado
ax.axvspan(2011, 2016, alpha=0.08, color='blue', label='Período validación (2011-2016)')
ax.axvline(2011, color='blue', lw=1.2, ls='--', alpha=0.6)
ax.axvline(2016, color='blue', lw=1.2, ls='--', alpha=0.6)

ax.set_xlabel('Año', fontsize=12)
ax.set_ylabel('Píxeles urbanos', fontsize=12)
ax.set_title('Evolución del crecimiento urbano de Querétaro (1984–2020)', fontsize=13, fontweight='bold')
ax.set_xlim(1983, 2021)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.grid(True, alpha=0.3, ls='--')
ax.legend(fontsize=10)

plt.tight_layout()
out = FIGS_OUT / 'urban_growth_timeline.png'
plt.savefig(out, dpi=180, bbox_inches='tight')
plt.close()
print(f'   ✓ {out}')

# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 2: WoE Information Values + ranges
# ─────────────────────────────────────────────────────────────────────────────
print('\n[2/6] woe_weights.png ...')

with open(WOE_PKL, 'rb') as f:
    woe_data = pickle.load(f)
woe_calc = woe_data['woe_calculator']

var_labels = {
    'distance_urban':        'Distancia a zona\nurbana',
    'neighbor_density_3x3':  'Densidad vecinos\n3×3',
    'neighbor_density_5x5':  'Densidad vecinos\n5×5',
    'neighbor_density_7x7':  'Densidad vecinos\n7×7',
    'local_fragmentation':   'Fragmentación\nlocal',
    'urban_gradient':        'Gradiente\nurbano',
    'nearest_cluster_size':  'Tamaño cluster\nmás cercano',
}

total_iv = sum(r.iv_total for r in woe_calc.woe_results.values())
sorted_vars = sorted(woe_calc.woe_results.items(), key=lambda x: -x[1].iv_total)

names  = [var_labels.get(v, v) for v, _ in sorted_vars]
ivs    = [r.iv_total for _, r in sorted_vars]
weights= [r.iv_total / total_iv * 100 for _, r in sorted_vars]
woe_min= [min(r.woe_values) for _, r in sorted_vars]
woe_max= [max(r.woe_values) for _, r in sorted_vars]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Panel izquierdo: IV + peso
colors = plt.cm.RdYlGn([w/max(weights) for w in weights])
bars = ax1.barh(names[::-1], ivs[::-1], color=colors[::-1], edgecolor='white', height=0.6)
for bar, iv, w in zip(bars, ivs[::-1], weights[::-1]):
    ax1.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
             f'{iv:.3f}  ({w:.1f}%)', va='center', fontsize=9)
ax1.set_xlabel('Information Value (IV)', fontsize=11)
ax1.set_title('Information Value por variable\n(ponderación en el modelo)', fontsize=11, fontweight='bold')
ax1.set_xlim(0, max(ivs) * 1.35)
ax1.grid(True, axis='x', alpha=0.3)

# Panel derecho: WoE range como barras de error
y_pos = range(len(sorted_vars))
ax2.barh(list(y_pos)[::-1],
         [mx - mn for mn, mx in zip(woe_min, woe_max)],
         left=woe_min[::-1],
         color='steelblue', alpha=0.65, height=0.55, edgecolor='white')
ax2.axvline(0, color='black', lw=1.2, ls='--')
ax2.set_yticks(list(y_pos))
ax2.set_yticklabels(names[::-1], fontsize=9)
ax2.set_xlabel('Rango de WoE [min, max]', fontsize=11)
ax2.set_title('Rango WoE por variable\n(evidencia negativa ↔ positiva)', fontsize=11, fontweight='bold')
ax2.grid(True, axis='x', alpha=0.3)

fig.suptitle('Análisis de Variables — WoE Pooled 1984–2010', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
out = FIGS_OUT / 'woe_weights.png'
plt.savefig(out, dpi=180, bbox_inches='tight')
plt.close()
print(f'   ✓ {out}')

# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 3-7: Copiar visualizaciones año a año del quinquenal
# ─────────────────────────────────────────────────────────────────────────────
print('\n[3/6] Copiando visualizaciones quinquenales ...')

viz_src = QUINQ_DIR / 'visualizations'
pairs = ['2011_to_2012', '2012_to_2013', '2013_to_2014', '2014_to_2015', '2015_to_2016']
for pair in pairs:
    src = viz_src / f'{pair}.png'
    dst = FIGS_OUT / 'comparisons' / f'quinquenal_{pair}.png'
    if src.exists():
        shutil.copy2(src, dst)
        print(f'   ✓ {dst.name}')
    else:
        print(f'   ⚠ No encontrado: {src}')

# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 4: Resumen simulación quinquenal (mosaico 2011 obs vs 2016 obs vs 2016 pred)
# ─────────────────────────────────────────────────────────────────────────────
print('\n[4/6] ac_simulation_quinquenal.png ...')

map_2011 = np.load(MAPS_DIR / '2011.npy')
map_2016_obs = np.load(MAPS_DIR / '2016.npy')
map_2016_pred = np.load(QUINQ_DIR / 'predicted_2016.npy')

def to_rgb(m):
    rgb = np.zeros((*m.shape, 3))
    rgb[m == 0] = [0.18, 0.49, 0.20]  # Verde oscuro
    rgb[m == 1] = [0.78, 0.08, 0.08]  # Rojo
    return rgb

# Diferencia observada vs predicha
diff = np.zeros_like(map_2016_obs, dtype=int)
diff[(map_2016_obs == 0) & (map_2016_pred == 1)] = 1   # FP: sobre-predicción
diff[(map_2016_obs == 1) & (map_2016_pred == 0)] = 2   # FN: sub-predicción
diff[(map_2016_obs == 1) & (map_2016_pred == 1)] = 3   # TP: correcto
diff_rgb = np.ones((*diff.shape, 3))
diff_rgb[diff == 1] = [0.85, 0.33, 0.10]   # Naranja = FP
diff_rgb[diff == 2] = [0.13, 0.47, 0.71]   # Azul = FN
diff_rgb[diff == 3] = [0.17, 0.63, 0.17]   # Verde = TP

fig, axes = plt.subplots(1, 4, figsize=(20, 6))

axes[0].imshow(to_rgb(map_2011), aspect='auto')
axes[0].set_title(f'2011\n(Estado inicial)\n{np.sum(map_2011==1):,} px urbanos', fontsize=11, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(to_rgb(map_2016_obs), aspect='auto')
axes[1].set_title(f'2016 Observado\n(Ground truth)\n{np.sum(map_2016_obs==1):,} px urbanos', fontsize=11, fontweight='bold')
axes[1].axis('off')

axes[2].imshow(to_rgb(map_2016_pred), aspect='auto')
axes[2].set_title(f'2016 Predicho\n(AC WoE, 5 pasos)\n{np.sum(map_2016_pred==1):,} px urbanos', fontsize=11, fontweight='bold')
axes[2].axis('off')

axes[3].imshow(diff_rgb, aspect='auto')
axes[3].set_title('Diferencia\n(Obs vs Pred)', fontsize=11, fontweight='bold')
axes[3].axis('off')

# Leer TP/FP/FN de validation_results.json (no hardcodear)
with open(VAL_JSON) as f:
    val_data = json.load(f)
cm = val_data.get('confusion_matrix', {})
tp_val = cm.get('TP', int(np.sum((map_2016_obs == 1) & (map_2016_pred == 1))))
fp_val = cm.get('FP', int(np.sum((map_2016_obs == 0) & (map_2016_pred == 1))))
fn_val = cm.get('FN', int(np.sum((map_2016_obs == 1) & (map_2016_pred == 0))))
p_tp = mpatches.Patch(color=[0.17, 0.63, 0.17], label=f'TP (acierto) = {tp_val:,}')
p_fp = mpatches.Patch(color=[0.85, 0.33, 0.10], label=f'FP (sobre-pred) = {fp_val:,}')
p_fn = mpatches.Patch(color=[0.13, 0.47, 0.71], label=f'FN (omisión) = {fn_val:,}')
axes[3].legend(handles=[p_tp, p_fp, p_fn], loc='lower right', fontsize=8.5,
               framealpha=0.9)

fig.suptitle('Simulación quinquenal AC-WoE: 2011 → 2016', fontsize=14, fontweight='bold')
plt.tight_layout()
out = FIGS_OUT / 'ac_simulation_quinquenal.png'
plt.savefig(out, dpi=180, bbox_inches='tight')
plt.close()
print(f'   ✓ {out}')

# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 5: Dashboard de métricas de validación
# ─────────────────────────────────────────────────────────────────────────────
print('\n[5/6] validation_metrics.png ...')

with open(VAL_JSON) as f:
    val = json.load(f)

pm   = val['pixel_metrics']
fom  = val['fom_metrics']
cm   = val['confusion_matrix']
grow = val['growth_statistics']

fig = plt.figure(figsize=(15, 8))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# ── Panel 1: Barras de métricas pixel ──
ax1 = fig.add_subplot(gs[0, 0])
metrics_names  = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'IoU', 'Kappa']
metrics_values = [pm['accuracy'], pm['precision'], pm['recall'],
                  pm['f1_score'], pm['iou'], pm['kappa']]
colors_m = ['#2ecc71' if v >= 0.7 else '#e67e22' if v >= 0.5 else '#e74c3c'
            for v in metrics_values]
bars = ax1.bar(metrics_names, metrics_values, color=colors_m, edgecolor='white', width=0.6)
for bar, val_m in zip(bars, metrics_values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val_m:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax1.set_ylim(0, 1.15)
ax1.set_title('Métricas pixel a pixel', fontweight='bold', fontsize=11)
ax1.axhline(0.7, color='gray', ls='--', lw=1, alpha=0.5)
ax1.tick_params(axis='x', labelrotation=30, labelsize=9)
ax1.grid(True, axis='y', alpha=0.3)

# ── Panel 2: FoM descompuesto ──
ax2 = fig.add_subplot(gs[0, 1])
fom_labels = ['Hits (B)\nAciertos', 'Misses (C)\nOmisiones', 'False alarms (A)\nSobre-pred.']
fom_values = [fom['B_hits'], fom['C_misses'], fom['A_false_alarms']]
fom_colors = ['#27ae60', '#3498db', '#e74c3c']
wedges, texts, autotexts = ax2.pie(
    fom_values, labels=fom_labels, colors=fom_colors,
    autopct='%1.1f%%', startangle=90,
    textprops={'fontsize': 9}, pctdistance=0.75)
for at in autotexts:
    at.set_fontweight('bold')
ax2.set_title(f'Figure of Merit = {fom["fom"]:.3f}\n(descomposición de transiciones)',
              fontweight='bold', fontsize=11)

# ── Panel 3: Matriz de confusión ──
ax3 = fig.add_subplot(gs[0, 2])
conf = np.array([[cm['TP'], cm['FN']], [cm['FP'], cm['TN']]])
conf_norm = conf / conf.sum()
im = ax3.imshow(conf_norm, cmap='RdYlGn', vmin=0, vmax=0.5)
ax3.set_xticks([0, 1])
ax3.set_yticks([0, 1])
ax3.set_xticklabels(['Pred Urbano', 'Pred No-Urb'], fontsize=9)
ax3.set_yticklabels(['Obs Urbano', 'Obs No-Urb'], fontsize=9)
labels_conf = [['TP', 'FN'], ['FP', 'TN']]
for i in range(2):
    for j in range(2):
        ax3.text(j, i, f'{labels_conf[i][j]}\n{conf[i,j]:,}\n({conf_norm[i,j]:.1%})',
                 ha='center', va='center', fontsize=9, fontweight='bold',
                 color='black')
ax3.set_title('Matriz de confusión\n(píxel a píxel)', fontweight='bold', fontsize=11)

# ── Panel 4: Crecimiento predicho vs observado ──
ax4 = fig.add_subplot(gs[1, 0])
categories = ['Urbano\n2011\n(inicial)', 'Urbano\n2016\nobservado', 'Urbano\n2016\npredicho']
values     = [grow['urban_2011'], grow['urban_2016_observed'], grow['urban_2016_predicted']]
bar_colors = ['#7f8c8d', '#27ae60', '#e74c3c']
bars4 = ax4.bar(categories, values, color=bar_colors, edgecolor='white', width=0.5)
for bar, v in zip(bars4, values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15000,
             f'{v:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax4.set_title('Crecimiento urbano\nobservado vs predicho', fontweight='bold', fontsize=11)
ax4.set_ylabel('Píxeles urbanos')
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax4.grid(True, axis='y', alpha=0.3)

# ── Panel 5: Métricas de cambio ──
ax5 = fig.add_subplot(gs[1, 1])
cm_names  = ['Change\nPrecision', 'Change\nRecall', 'Change\nF1']
cm_values = [val['change_metrics']['change_precision'],
             val['change_metrics']['change_recall'],
             val['change_metrics']['change_f1']]
cm_colors = ['#e67e22' if v < 0.5 else '#2ecc71' for v in cm_values]
bars5 = ax5.bar(cm_names, cm_values, color=cm_colors, edgecolor='white', width=0.5)
for bar, v in zip(bars5, cm_values):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax5.set_ylim(0, 1.15)
ax5.set_title('Métricas de detección\nde cambio urbano', fontweight='bold', fontsize=11)
ax5.axhline(0.5, color='gray', ls='--', lw=1, alpha=0.5)
ax5.grid(True, axis='y', alpha=0.3)

# ── Panel 6: Resumen de parámetros ──
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
summary_text = (
    "Configuración del experimento\n"
    "─────────────────────────────\n"
    "WoE: pooled 1984–2010\n"
    "  Períodos: 26 | Trans: 8,196,264\n"
    "  Variables: 7 (pond. por IV)\n\n"
    "Parámetros AC:\n"
    f"  threshold       = 0.750\n"
    f"  neighbor_weight = 0.500\n"
    f"  distance_weight = 1.963\n"
    f"  pasos           = 5 (2011→2016)\n\n"
    "Resultados clave:\n"
    f"  FoM   = {fom['fom']:.3f}\n"
    f"  F1    = {pm['f1_score']:.3f}\n"
    f"  Kappa = {pm['kappa']:.3f}\n"
    f"  Recall= {pm['recall']:.3f}"
)
ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
         fontsize=9.5, va='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
ax6.set_title('Resumen del experimento', fontweight='bold', fontsize=11)

fig.suptitle('Dashboard de Validación Quinquenal AC-WoE — 2011 → 2016',
             fontsize=14, fontweight='bold', y=1.01)

out = FIGS_OUT / 'validation_metrics.png'
plt.savefig(out, dpi=180, bbox_inches='tight')
plt.close()
print(f'   ✓ {out}')

# ─────────────────────────────────────────────────────────────────────────────
# RESUMEN FINAL
# ─────────────────────────────────────────────────────────────────────────────
print('\n[6/6] Verificando archivos generados...')
expected = [
    FIGS_OUT / 'urban_growth_timeline.png',
    FIGS_OUT / 'woe_weights.png',
    FIGS_OUT / 'ac_simulation_quinquenal.png',
    FIGS_OUT / 'validation_metrics.png',
] + [FIGS_OUT / 'comparisons' / f'quinquenal_{p}.png' for p in pairs]

all_ok = True
for f in expected:
    exists = f.exists()
    size   = f'{f.stat().st_size/1024:.0f} KB' if exists else '—'
    mark   = '✓' if exists else '✗'
    print(f'   {mark} {f.relative_to(FIGS_OUT.parent.parent.parent.parent.parent)} ({size})')
    if not exists:
        all_ok = False

print()
print('=' * 65)
if all_ok:
    print('✅ Todas las figuras generadas correctamente.')
else:
    print('⚠️  Algunas figuras no se generaron. Revisar errores arriba.')
print('=' * 65)
