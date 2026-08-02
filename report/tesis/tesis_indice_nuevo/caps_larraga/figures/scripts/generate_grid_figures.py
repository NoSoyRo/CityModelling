#!/usr/bin/env python3
"""
Figuras de cuadrícula — Capítulo 5: Modelo WoE-AC
Alta prioridad: F1 (mapa binario), F2 (transición), F3 (Moore),
                F6 (paso AC), F8 (componentes FoM)
Salida: report/tesis/.../figures/grid_diagrams/
v2: traslapes corregidos en F2, F6, F8
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

np.random.seed(42)

# ── Rutas ────────────────────────────────────────────────────────────────────
OUTPUT = Path('report/tesis/tesis_indice_nuevo/caps_larraga/figures/grid_diagrams')
OUTPUT.mkdir(parents=True, exist_ok=True)

DPI = 300

# ── Paleta ───────────────────────────────────────────────────────────────────
BG       = '#FAFBFC'
C_NU     = '#EEF1F5'
C_U      = '#1B2631'
C_BORDER = '#C8CDD0'
C_NEW    = '#C0392B'
C_MO     = '#2471A3'
C_CTR    = '#D35400'
C_HIT    = '#1D6A39'
C_FAL    = '#B7770D'
C_MIS    = '#154360'
C_PU     = '#717D7E'

plt.rcParams.update({
    'font.family':       'sans-serif',
    'figure.facecolor':  BG,
    'axes.facecolor':    BG,
    'savefig.facecolor': BG,
})

# ── Patrón base (10×10) ──────────────────────────────────────────────────────
BASE = np.array([
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,1,1,1,0,0,0],
    [0,0,0,1,1,1,1,1,0,0],
    [0,0,1,1,1,1,1,1,0,0],
    [0,0,1,1,1,1,1,1,1,0],
    [0,0,1,1,1,1,1,1,0,0],
    [0,0,0,1,1,1,1,0,0,0],
    [0,0,0,0,1,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
], dtype=int)

# ── Utilidades ───────────────────────────────────────────────────────────────

def render(ax, grid, color_fn, text_fn=None, fs=9, bw=0.8, highlight=None):
    R, C = grid.shape
    for i in range(R):
        for j in range(C):
            v  = grid[i, j]
            fc = color_fn(i, j, v)
            is_hi  = highlight and (i, j) in highlight
            bord   = '#E74C3C' if is_hi else C_BORDER
            bw_use = 3.0       if is_hi else bw
            ax.add_patch(mpatches.Rectangle(
                (j, R - 1 - i), 1, 1,
                linewidth=bw_use, edgecolor=bord, facecolor=fc, zorder=3))
            if text_fn:
                t = text_fn(i, j, v)
                if t:
                    dark = fc in {C_U, C_NEW, C_MO, C_CTR,
                                  C_HIT, C_FAL, C_MIS, C_PU}
                    ax.text(j + .5, R - .5 - i, t,
                            ha='center', va='center', fontsize=fs,
                            color='white' if dark else '#1B2631',
                            fontweight='bold', fontfamily='monospace', zorder=4)
    ax.set_xlim(0, C); ax.set_ylim(0, R)
    ax.set_aspect('equal'); ax.axis('off')


def LP(label, color, edgecolor=C_BORDER):
    return mpatches.Patch(facecolor=color, edgecolor=edgecolor, label=label)


# ═════════════════════════════════════════════════════════════════════════════
# F1 — Mapa binario
# ═════════════════════════════════════════════════════════════════════════════
def f1_mapa_binario():
    fig, ax = plt.subplots(figsize=(5.8, 6.2))

    render(ax, BASE,
           lambda i, j, v: C_U if v else C_NU,
           lambda i, j, v: str(v), fs=14)

    ax.set_title(r'Mapa binario:  $U_t(i,j)\in\{0,\,1\}$',
                 fontsize=15, fontweight='bold', pad=14, color='#1B2631')

    ax.legend(handles=[
        LP('Urbano  $(U_t = 1)$', C_U),
        LP('No urbano  $(U_t = 0)$', C_NU),
    ], loc='upper right', fontsize=11, framealpha=0.95,
       edgecolor='#BDC3C7', facecolor='white')

    ax.text(0.5, -0.03,
            'Cada celda representa ≈ 30 m × 30 m sobre el territorio',
            ha='center', va='top', transform=ax.transAxes,
            fontsize=9.5, color='#7F8C8D', style='italic')

    plt.tight_layout()
    fig.savefig(OUTPUT / 'F1_mapa_binario.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print('✅  F1_mapa_binario.png')


# ═════════════════════════════════════════════════════════════════════════════
# F2 — Definición de transición  ·  sin traslapes
# ═════════════════════════════════════════════════════════════════════════════
def f2_transicion():
    TI, TJ = 3, 2   # celda que transita
    R = BASE.shape[0]

    # figura más alta para dejar espacio a callouts debajo de cada grid
    fig = plt.figure(figsize=(13, 7.2), facecolor=BG)

    # ── panel t (deja margen inferior para callout) ──────────────
    ax1 = fig.add_axes([0.03, 0.22, 0.39, 0.66])
    render(ax1, BASE,
           lambda i, j, v: C_U if v else C_NU,
           lambda i, j, v: str(v), fs=13, highlight={(TI, TJ)})
    ax1.set_title(r'Año $t$', fontsize=16, fontweight='bold',
                  pad=10, color='#1B2631')

    # callout F2-izquierdo: en coordenadas de FIGURA, debajo del panel
    fig.text(0.215, 0.185,
             'Borde rojo  →  celda candidata  $(U_t = 0)$',
             ha='center', va='top', fontsize=10, color=C_NEW, style='italic',
             bbox=dict(facecolor='white', edgecolor=C_NEW,
                       boxstyle='round,pad=0.38', linewidth=1.4, alpha=0.97))

    # ── flecha central ───────────────────────────────────────────
    ax_arr = fig.add_axes([0.42, 0.34, 0.16, 0.38])
    ax_arr.set_facecolor(BG)
    ax_arr.axis('off')
    ax_arr.annotate('', xy=(0.90, 0.5), xytext=(0.10, 0.5),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='-|>', color='#1B2631',
                                   lw=2.5, mutation_scale=28))
    for dy, txt, fs_ in [
        (0.90, 'Regla de transición',            9.5),
        (0.68, r'$P_{i,j} > \theta = 0{,}75$',  13),
        (0.36, r'$+$  componente estocástica',    8.8),
    ]:
        ax_arr.text(0.5, dy, txt, ha='center', va='center',
                    fontsize=fs_, color='#2C3E50',
                    transform=ax_arr.transAxes)

    # ── panel t+1 (igual margen inferior) ───────────────────────
    ax2 = fig.add_axes([0.58, 0.22, 0.39, 0.66])

    def cmap2(i, j, v):
        return C_NEW if (i == TI and j == TJ) else (C_U if v else C_NU)

    def txt2(i, j, v):
        return '1' if (i == TI and j == TJ) else str(v)

    render(ax2, BASE, cmap2, txt2, fs=13)
    ax2.set_title(r'Año $t+1$', fontsize=16, fontweight='bold',
                  pad=10, color='#1B2631')

    # callout F2-derecho
    fig.text(0.775, 0.185,
             'Celda roja  →  nueva celda urbana  $(U_{t+1} = 1)$',
             ha='center', va='top', fontsize=10, color='#922B21', style='italic',
             bbox=dict(facecolor='white', edgecolor='#922B21',
                       boxstyle='round,pad=0.38', linewidth=1.4, alpha=0.97))

    # ── leyenda global ───────────────────────────────────────────
    fig.legend(handles=[
        LP('Urbano  $(U = 1)$',             C_U),
        LP('No urbano  $(U = 0)$',          C_NU),
        LP(r'Transición  $0\rightarrow 1$', C_NEW),
    ], loc='lower center', ncol=3, fontsize=11,
       framealpha=0.95, edgecolor='#BDC3C7', facecolor='white',
       bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        r'Transición: $U_t(i,j)=0 \;\rightarrow\; U_{t+1}(i,j)=1$',
        fontsize=15, fontweight='bold', y=0.99, color='#1B2631')

    fig.savefig(OUTPUT / 'F2_transicion.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print('✅  F2_transicion.png')


# ═════════════════════════════════════════════════════════════════════════════
# F3 — Vecindad de Moore
# ═════════════════════════════════════════════════════════════════════════════
def f3_moore():
    CI, CJ = 3, 3
    grid7  = np.zeros((7, 7), dtype=int)
    for r, c in [(1, 5), (2, 4), (2, 5), (3, 4), (4, 4), (4, 5), (5, 5)]:
        grid7[r, c] = 1

    moore = {(CI + di, CJ + dj)
             for di in (-1, 0, 1) for dj in (-1, 0, 1)
             if (di, dj) != (0, 0)}
    vn    = {(CI-1, CJ), (CI+1, CJ), (CI, CJ-1), (CI, CJ+1)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))

    # ── izquierdo: Moore ─────────────────────────────────────────
    ax = axes[0]
    def cmap_m(i, j, v):
        if (i, j) == (CI, CJ): return C_CTR
        if (i, j) in moore:    return C_MO
        return C_U if v else C_NU

    render(ax, grid7, cmap_m, fs=9)
    R = 7
    ax.text(CJ + .5, R - .5 - CI, 'Celda\nobjetivo',
            ha='center', va='center', fontsize=8.5,
            color='white', fontweight='bold')
    for n, (r, c) in enumerate(sorted(moore), 1):
        ax.text(c + .5, R - .5 - r, str(n),
                ha='center', va='center', fontsize=11,
                color='white', fontweight='bold')

    ax.set_title('Vecindad de Moore\n(8 celdas adyacentes)',
                 fontsize=14, fontweight='bold', pad=12, color='#1B2631')
    ax.legend(handles=[
        LP('Celda objetivo', C_CTR),
        LP('Vecinos de Moore (8)', C_MO),
        LP('Urbano', C_U), LP('No urbano', C_NU),
    ], fontsize=10, loc='lower right',
       framealpha=0.95, edgecolor='#BDC3C7', facecolor='white')

    # ── derecho: Moore vs Von Neumann ────────────────────────────
    ax2 = axes[1]
    diag_only = moore - vn

    def cmap_vn(i, j, v):
        if (i, j) == (CI, CJ):  return C_CTR
        if (i, j) in vn:         return '#117A65'
        if (i, j) in diag_only:  return C_MO
        return C_U if v else C_NU

    render(ax2, grid7, cmap_vn, fs=9)
    ax2.text(CJ + .5, R - .5 - CI, 'Celda\nobjetivo',
             ha='center', va='center', fontsize=8.5,
             color='white', fontweight='bold')
    ax2.set_title('Moore vs. Von Neumann\n(ortogonales + diagonales)',
                  fontsize=14, fontweight='bold', pad=12, color='#1B2631')
    ax2.legend(handles=[
        LP('Celda objetivo', C_CTR),
        LP('Von Neumann — ortogonales (4)', '#117A65'),
        LP('Solo Moore — diagonales (4)', C_MO),
        LP('Urbano', C_U), LP('No urbano', C_NU),
    ], fontsize=9.5, loc='lower right',
       framealpha=0.95, edgecolor='#BDC3C7', facecolor='white')

    ax2.text(0.5, -0.04,
             'Moore captura frentes diagonales del crecimiento urbano',
             ha='center', va='top', transform=ax2.transAxes,
             fontsize=9.5, color='#7F8C8D', style='italic')

    plt.tight_layout()
    fig.savefig(OUTPUT / 'F3_vecindad_moore.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print('✅  F3_vecindad_moore.png')


# ═════════════════════════════════════════════════════════════════════════════
# F6 — Un paso del AC  ·  sin traslapes
# ═════════════════════════════════════════════════════════════════════════════
def f6_paso_ac():
    grid_t = BASE.copy()
    R, C = grid_t.shape

    new_cells = set()
    for i in range(R):
        for j in range(C):
            if grid_t[i, j] == 0:
                nbrs = sum(
                    grid_t[i + di, j + dj]
                    for di in (-1, 0, 1) for dj in (-1, 0, 1)
                    if (di, dj) != (0, 0) and 0 <= i+di < R and 0 <= j+dj < C)
                if nbrs >= 3:
                    new_cells.add((i, j))
    new_cells = set(list(sorted(new_cells))[:5])

    # ── gridspec: 3 columnas [grid | flecha | grid] ──────────────
    fig = plt.figure(figsize=(14, 6.8), facecolor=BG)
    gs  = fig.add_gridspec(
        1, 3,
        width_ratios=[10, 4, 10],
        left=0.02, right=0.98,
        top=0.88,  bottom=0.14,
        wspace=0.08)

    ax1  = fig.add_subplot(gs[0])
    ax_a = fig.add_subplot(gs[1])
    ax2  = fig.add_subplot(gs[2])

    # ── panel t ─────────────────────────────────────────────────
    render(ax1, grid_t,
           lambda i, j, v: C_U if v else C_NU,
           highlight=new_cells)
    ax1.set_title('Estado  $U_t$', fontsize=14, fontweight='bold',
                  pad=10, color='#1B2631')
    # Subtítulo bajo el grid (transAxes → no choca con el grid)
    ax1.text(0.5, -0.05,
             'Borde rojo: celdas candidatas a transitar',
             ha='center', va='top', transform=ax1.transAxes,
             fontsize=9.5, color='#C0392B', style='italic',
             bbox=dict(facecolor='white', edgecolor='#C0392B',
                       boxstyle='round,pad=0.3', linewidth=1.2, alpha=0.95))

    # ── flecha + pasos de la regla ───────────────────────────────
    ax_a.set_facecolor(BG)
    ax_a.axis('off')
    ax_a.set_xlim(0, 1); ax_a.set_ylim(0, 1)

    # flecha horizontal
    ax_a.annotate('', xy=(0.88, 0.5), xytext=(0.12, 0.5),
                  xycoords='axes fraction', textcoords='axes fraction',
                  arrowprops=dict(arrowstyle='-|>', color='#1B2631',
                                  lw=2.5, mutation_scale=26))

    # pasos de la regla: separados claramente, sin solapamiento
    pasos = [
        (0.93, 'Recalcular',              9.5, True),
        (0.82, '7 variables espaciales',   9.0, False),
        (0.68, '↓',                       12,  False),
        (0.57, 'Aplicar WoE × IV',         9.0, False),
        (0.43, '↓',                       12,  False),
        (0.32, r'$P_{i,j} > \theta$',     12,  False),
        (0.18, r'($\theta = 0{,}75$)',      8.5, False),
    ]
    for y_, t_, fs_, b_ in pasos:
        ax_a.text(0.5, y_, t_, ha='center', va='center',
                  fontsize=fs_,
                  color='#1B2631' if b_ else '#2C3E50',
                  fontweight='bold' if b_ else 'normal',
                  transform=ax_a.transAxes)

    # ── panel t+1 ────────────────────────────────────────────────
    def cmap2(i, j, v):
        return C_NEW if (i, j) in new_cells else (C_U if v else C_NU)

    def txt2(i, j, v):
        return '↑' if (i, j) in new_cells else ''

    render(ax2, grid_t, cmap2, txt2, fs=13)
    ax2.set_title('Estado  $U_{t+1}$', fontsize=14, fontweight='bold',
                  pad=10, color='#1B2631')
    ax2.text(0.5, -0.05,
             'Nuevas celdas urbanas (rojo)',
             ha='center', va='top', transform=ax2.transAxes,
             fontsize=9.5, color=C_NEW, style='italic',
             bbox=dict(facecolor='white', edgecolor=C_NEW,
                       boxstyle='round,pad=0.3', linewidth=1.2, alpha=0.95))

    # ── suptitle y leyenda ───────────────────────────────────────
    fig.suptitle('Un paso del Autómata Celular: propagación del frente urbano',
                 fontsize=14, fontweight='bold', y=0.97, color='#1B2631')

    fig.legend(handles=[
        LP('Urbano', C_U),
        LP('No urbano', C_NU),
        LP('Candidata (borde rojo)', C_NU, '#C0392B'),
        LP('Nueva celda urbana', C_NEW),
    ], loc='lower center', ncol=4, fontsize=10.5,
       framealpha=0.95, edgecolor='#BDC3C7', facecolor='white',
       bbox_to_anchor=(0.5, 0.0))

    fig.savefig(OUTPUT / 'F6_paso_ac.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print('✅  F6_paso_ac.png')


# ═════════════════════════════════════════════════════════════════════════════
# F8 — Componentes del FoM  ·  sin traslapes
# ═════════════════════════════════════════════════════════════════════════════
def f8_fom():
    grid_init = BASE.copy()
    R, C = grid_init.shape

    grid_obs = BASE.copy()
    for r, c in [(2,1),(3,1),(3,2),(6,1),(7,3),(7,4)]:
        grid_obs[r, c] = 1

    grid_pred = BASE.copy()
    for r, c in [(2,1),(3,1),(4,1),(5,1),(6,2),(6,1)]:
        grid_pred[r, c] = 1

    hits, false_a, miss, perm_u = set(), set(), set(), set()
    for i in range(R):
        for j in range(C):
            if grid_init[i, j] == 1:
                perm_u.add((i, j))
            else:
                o = grid_obs[i, j]
                p = grid_pred[i, j]
                if   p == 1 and o == 1: hits.add((i, j))
                elif p == 1 and o == 0: false_a.add((i, j))
                elif p == 0 and o == 1: miss.add((i, j))

    B, A, Cn = len(hits), len(false_a), len(miss)
    FoM = B / (A + B + Cn) if (A + B + Cn) > 0 else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    # ── izquierdo: cuadrícula de diferencia ──────────────────────
    ax = axes[0]

    def cmap_fom(i, j, v):
        if (i, j) in perm_u:  return C_PU
        if (i, j) in hits:    return C_HIT
        if (i, j) in false_a: return C_FAL
        if (i, j) in miss:    return C_MIS
        return C_NU

    def txt_fom(i, j, v):
        if (i, j) in hits:    return 'B'
        if (i, j) in false_a: return 'A'
        if (i, j) in miss:    return 'C'
        return ''

    render(ax, grid_init, cmap_fom, txt_fom, fs=13)
    ax.set_title('Mapa de diferencia — predicho vs. observado',
                 fontsize=13, fontweight='bold', pad=12, color='#1B2631')
    ax.legend(handles=[
        LP(f'B  Acierto        (predijo y ocurrió)    n={B}',   C_HIT),
        LP(f'A  Falsa alarma  (predijo, no ocurrió)  n={A}',    C_FAL),
        LP(f'C  Omisión        (ocurrió, no predijo)  n={Cn}',  C_MIS),
        LP('Urbano permanente (no entra al FoM)',                C_PU),
        LP('No urbano permanente',                               C_NU),
    ], fontsize=9.5, loc='lower right', framealpha=0.95,
       edgecolor='#BDC3C7', facecolor='white',
       title='Componentes', title_fontsize=10)

    # ── derecho: diagrama A|B|C horizontal, sin solapamiento ─────
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    ax2.axis('off')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect('auto')

    ax2.set_title('Figure of Merit — Pontius et al.',
                  fontsize=13, fontweight='bold', pad=12, color='#1B2631')

    # Tres cajas horizontales: A (0–3) | B (3–6.5) | C (6.5–10)
    # Cada caja ocupa su propia franja sin traslaparse
    cajas = [
        # (x_ini, ancho, letra, color_borde, color_fondo, titulo_arriba,
        #  linea1_abajo, linea2_abajo)
        (0.2, 2.8, 'A', C_FAL, '#FEF9E7',
         'Falsa alarma', 'Predijo,', f'no ocurrió   n={A}'),
        (3.2, 3.4, 'B', C_HIT, '#EAFAF1',
         'Acierto', 'Predijo y', f'ocurrió   n={B}'),
        (6.8, 3.0, 'C', C_MIS, '#EAF2FB',
         'Omisión', 'Ocurrió,', f'no predijo   n={Cn}'),
    ]

    y_box_bot = 3.2
    y_box_top = 8.5
    box_h     = y_box_top - y_box_bot

    for x0, w, letra, col_borde, col_fondo, titulo, l1, l2 in cajas:
        cx = x0 + w / 2

        # Caja de fondo
        ax2.add_patch(mpatches.FancyBboxPatch(
            (x0, y_box_bot), w, box_h,
            boxstyle='round,pad=0.15', linewidth=2,
            facecolor=col_fondo, edgecolor=col_borde, zorder=2))

        # Círculo con letra (centrado verticalmente en la caja)
        cy = (y_box_bot + y_box_top) / 2 + 0.6
        ax2.add_patch(mpatches.Circle(
            (cx, cy), 0.7,
            facecolor=col_borde, edgecolor='white', linewidth=1.5, zorder=5))
        ax2.text(cx, cy, letra,
                 ha='center', va='center', fontsize=15,
                 color='white', fontweight='bold', zorder=6)

        # Texto bajo el círculo (dentro de la caja, espacio propio)
        ax2.text(cx, y_box_bot + 0.25, l1,
                 ha='center', va='bottom', fontsize=9, color='#2C3E50', zorder=4)
        ax2.text(cx, y_box_bot + 0.25 + 0.72, l2,
                 ha='center', va='bottom', fontsize=9, color='#2C3E50', zorder=4)

        # Título encima de la caja (espacio libre entre caja y borde superior)
        ax2.text(cx, y_box_top + 0.20, titulo,
                 ha='center', va='bottom', fontsize=11,
                 color=col_borde, fontweight='bold', zorder=4)

    # Indicadores "Solo predicho" / "Predicho y observado" / "Solo observado"
    etiquetas_zona = [
        (0.2 + 2.8/2, 3.05, 'Solo predicho',       C_FAL),
        (3.2 + 3.4/2, 3.05, 'Predicho\ny observado', C_HIT),
        (6.8 + 3.0/2, 3.05, 'Solo observado',        C_MIS),
    ]
    for xe, ye, txt, col in etiquetas_zona:
        ax2.text(xe, ye, txt, ha='center', va='top',
                 fontsize=8, color=col, style='italic')

    # Fórmula en la parte inferior, sin interferir con las cajas
    ax2.text(5.0, 1.7,
             rf'$\mathrm{{FoM}} = \dfrac{{B}}{{A+B+C}}'
             rf'= \dfrac{{{B}}}{{{A}+{B}+{Cn}}} = {FoM:.3f}$',
             ha='center', va='center', fontsize=14,
             color='#1B2631', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5',
                       facecolor='white', edgecolor='#BDC3C7', linewidth=1.5))

    plt.tight_layout()
    fig.savefig(OUTPUT / 'F8_componentes_fom.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print('✅  F8_componentes_fom.png')


# ═════════════════════════════════════════════════════════════════════════════
# Ejecutar
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generando figuras de cuadrícula — Capítulo 5  (v2, sin traslapes)\n')
    f1_mapa_binario()
    f2_transicion()
    f3_moore()
    f6_paso_ac()
    f8_fom()
    print(f'\nListo. Figuras en: {OUTPUT.resolve()}')
