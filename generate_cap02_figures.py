#!/usr/bin/env python3
"""
Genera los esquemas ilustrativos del Capitulo 2 (Marco teorico).

Se dibujan por codigo, no con IA, de modo que no requieren la leyenda de imagen
generativa y cualquier lector puede reproducirlos.

Figuras generadas:
  1. esquema_espacio_celular.png   - dimensiones y teselaciones del espacio celular
  2. esquema_fronteras.png         - condiciones de frontera periodica, fija y reflejante
  3. esquema_matriz_confusion.png  - matriz de confusion y componentes del FoM
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

FIGS_OUT = Path('report/tesis/tesis_indice_nuevo/caps_larraga/figures')
FIGS_OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 10,
    'mathtext.fontset': 'dejavuserif',
    'figure.dpi': 200,
    'savefig.bbox': 'tight',
})

URBANO = '#1b3a5c'
NO_URBANO = '#d9d9d9'
OBJETIVO = '#d94f1e'
VECINO = '#2f7cb5'
EXTERIOR = '#f2f2f2'
BORDE = '#5a5a5a'


def _limpiar(ax):
    ax.set_aspect('equal')
    ax.axis('off')


# ---------------------------------------------------------------------------
# 1. Espacio celular: dimensiones y teselaciones
# ---------------------------------------------------------------------------
def espacio_celular():
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.9))

    # Marco identico en los cuatro paneles: mismo rango de datos y aspecto igual,
    # de modo que las cajas fisicas coinciden y los titulos quedan a una altura.
    ANCHO_VIS, ALTO_VIS = 5.8, 4.8

    def encuadrar(ax, x0, x1, y0, y1):
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ax.set_xlim(cx - ANCHO_VIS / 2, cx + ANCHO_VIS / 2)
        ax.set_ylim(cy - ALTO_VIS / 2, cy + ALTO_VIS / 2)

    # (a) unidimensional
    ax = axes[0]
    estados = [0, 1, 1, 0, 1]
    for i, s in enumerate(estados):
        ax.add_patch(Rectangle((i, 0), 1, 1,
                               facecolor=URBANO if s else NO_URBANO,
                               edgecolor=BORDE, linewidth=0.7))
    encuadrar(ax, 0, len(estados), 0, 1)
    ax.set_title('(a) Unidimensional')
    _limpiar(ax)

    # (b) rejilla cuadrada
    ax = axes[1]
    rng = np.random.default_rng(7)
    malla = rng.integers(0, 2, size=(4, 4))
    for i in range(4):
        for j in range(4):
            ax.add_patch(Rectangle((j, 3 - i), 1, 1,
                                   facecolor=URBANO if malla[i, j] else NO_URBANO,
                                   edgecolor=BORDE, linewidth=0.7))
    encuadrar(ax, 0, 4, 0, 4)
    ax.set_title('(b) Rejilla cuadrada')
    _limpiar(ax)

    # (c) teselacion hexagonal
    ax = axes[2]
    r = 0.62
    rng = np.random.default_rng(3)
    for fila in range(4):
        for col in range(4):
            cx = col * np.sqrt(3) * r + (np.sqrt(3) / 2 * r if fila % 2 else 0)
            cy = fila * 1.5 * r
            ang = np.pi / 180 * (np.arange(6) * 60 + 30)
            verts = np.column_stack([cx + r * np.cos(ang), cy + r * np.sin(ang)])
            ax.add_patch(Polygon(verts,
                                 facecolor=URBANO if rng.integers(0, 2) else NO_URBANO,
                                 edgecolor=BORDE, linewidth=0.7))
    encuadrar(ax, -r, 3 * np.sqrt(3) * r + np.sqrt(3) / 2 * r + r, -r, 3 * 1.5 * r + r)
    ax.set_title('(c) Teselación hexagonal')
    _limpiar(ax)

    # (d) teselacion triangular
    # Cada fila alterna triangulos con vertice arriba y vertice abajo, de modo
    # que comparten arista y cubren el plano sin huecos ni solapes.
    ax = axes[3]
    rng = np.random.default_rng(11)
    lado, alto = 1.0, np.sqrt(3) / 2
    n_filas, n_col = 4, 7
    for fila in range(n_filas):
        y0 = fila * alto
        for col in range(n_col):
            xa = col * lado / 2
            if col % 2 == 0:
                verts = [(xa, y0), (xa + lado, y0), (xa + lado / 2, y0 + alto)]
            else:
                verts = [(xa, y0 + alto), (xa + lado, y0 + alto), (xa + lado / 2, y0)]
            ax.add_patch(Polygon(verts,
                                 facecolor=URBANO if rng.integers(0, 2) else NO_URBANO,
                                 edgecolor=BORDE, linewidth=0.7))
    encuadrar(ax, 0, (n_col - 1) * lado / 2 + lado, 0, n_filas * alto)
    ax.set_title('(d) Teselación triangular')
    _limpiar(ax)

    fig.savefig(FIGS_OUT / 'esquema_espacio_celular.png')
    plt.close(fig)
    print('  esquema_espacio_celular.png')


# ---------------------------------------------------------------------------
# 2. Condiciones de frontera
# ---------------------------------------------------------------------------
def fronteras():
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.4))
    n = 5
    rng = np.random.default_rng(19)
    malla = rng.integers(0, 2, size=(n, n))

    def dibujar_malla(ax):
        for i in range(n):
            for j in range(n):
                ax.add_patch(Rectangle((j, n - 1 - i), 1, 1,
                                       facecolor=URBANO if malla[i, j] else NO_URBANO,
                                       edgecolor=BORDE, linewidth=0.8))
        ax.add_patch(Rectangle((0, 0), n, n, fill=False,
                               edgecolor='black', linewidth=1.6))

    # Marco identico en los tres paneles, para que los titulos queden a una altura.
    def encuadrar(ax):
        ax.set_xlim(n / 2 - 5.1, n / 2 + 5.1)
        ax.set_ylim(n / 2 - 4.7, n / 2 + 4.7)

    # (a) periodica
    ax = axes[0]
    dibujar_malla(ax)
    for j in range(n):
        ax.add_patch(Rectangle((j, n), 1, 1, facecolor=URBANO if malla[n - 1, j] else NO_URBANO,
                               edgecolor=BORDE, linewidth=0.5, alpha=0.45))
        ax.add_patch(Rectangle((j, -1), 1, 1, facecolor=URBANO if malla[0, j] else NO_URBANO,
                               edgecolor=BORDE, linewidth=0.5, alpha=0.45))
    ax.add_patch(FancyArrowPatch((n + 0.35, 0.5), (n + 0.35, n - 0.5),
                                 connectionstyle='arc3,rad=0.5',
                                 arrowstyle='<->', color=OBJETIVO, linewidth=1.3))
    ax.text(n / 2, n + 1.9, 'el borde superior se une al inferior',
            ha='center', fontsize=7.8, color=OBJETIVO)
    encuadrar(ax)
    ax.set_title('(a) Periódica (toro)')
    _limpiar(ax)

    # (b) fija
    ax = axes[1]
    dibujar_malla(ax)
    for j in range(n):
        for (yy) in (n, -1):
            ax.add_patch(Rectangle((j, yy), 1, 1, facecolor=EXTERIOR,
                                   edgecolor=BORDE, linewidth=0.5, hatch='///'))
    for i in range(n):
        for xx in (n, -1):
            ax.add_patch(Rectangle((xx, i), 1, 1, facecolor=EXTERIOR,
                                   edgecolor=BORDE, linewidth=0.5, hatch='///'))
    ax.text(n / 2, n + 1.9, 'exterior fijo en no urbano',
            ha='center', fontsize=7.8, color=OBJETIVO)
    encuadrar(ax)
    ax.set_title('(b) Fija o absorbente')
    _limpiar(ax)

    # (c) reflejante
    ax = axes[2]
    dibujar_malla(ax)
    for j in range(n):
        ax.add_patch(Rectangle((j, n), 1, 1, facecolor=URBANO if malla[0, j] else NO_URBANO,
                               edgecolor=BORDE, linewidth=0.5, alpha=0.45))
    for i in range(n):
        ax.add_patch(Rectangle((n, i), 1, 1,
                               facecolor=URBANO if malla[n - 1 - i, n - 1] else NO_URBANO,
                               edgecolor=BORDE, linewidth=0.5, alpha=0.45))
    ax.add_patch(FancyArrowPatch((n / 2, n - 0.55), (n / 2, n + 0.55),
                                 arrowstyle='-|>', mutation_scale=12,
                                 color=OBJETIVO, linewidth=1.5))
    ax.text(n / 2, n + 1.9, 'la fila interior se replica hacia fuera',
            ha='center', fontsize=7.8, color=OBJETIVO)
    encuadrar(ax)
    ax.set_title('(c) Reflejante')
    _limpiar(ax)

    fig.savefig(FIGS_OUT / 'esquema_fronteras.png')
    plt.close(fig)
    print('  esquema_fronteras.png')


# ---------------------------------------------------------------------------
# 3. Matriz de confusion y componentes del FoM
# ---------------------------------------------------------------------------
def matriz_confusion():
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))

    # (a) matriz de confusion 2x2
    ax = axes[0]
    # Notacion identica a la de las ecuaciones del capitulo.
    celdas = [
        (0, 1, 'TP\nverdadero\npositivo', '#2f7cb5'),
        (1, 1, 'FP\nfalso\npositivo', '#e8b04b'),
        (0, 0, 'FN\nfalso\nnegativo', '#c9503a'),
        (1, 0, 'TN\nverdadero\nnegativo', '#b8c4cc'),
    ]
    for cx, cy, txt, color in celdas:
        ax.add_patch(Rectangle((cx, cy), 1, 1, facecolor=color,
                               edgecolor='white', linewidth=2.2))
        ax.text(cx + 0.5, cy + 0.5, txt, ha='center', va='center',
                fontsize=8.6, color='white', weight='bold')
    ax.text(0.5, 2.16, 'urbano', ha='center', fontsize=9)
    ax.text(1.5, 2.16, 'no urbano', ha='center', fontsize=9)
    ax.text(1.0, 2.52, 'Observado', ha='center', fontsize=9.5, weight='bold')
    ax.text(-0.14, 1.5, 'urbano', ha='right', va='center', fontsize=9)
    ax.text(-0.14, 0.5, 'no urbano', ha='right', va='center', fontsize=9)
    ax.text(-0.95, 1.0, 'Predicho', ha='center', va='center',
            fontsize=9.5, weight='bold', rotation=90)
    ax.set_xlim(-1.3, 2.3)
    ax.set_ylim(-0.35, 2.8)
    ax.set_title('(a) Matriz de confusión')
    _limpiar(ax)

    # (b) componentes del FoM
    ax = axes[1]
    r_obs, r_pred = 0.95, 0.95
    c_obs, c_pred = (1.35, 1.0), (2.25, 1.0)
    ax.add_patch(Circle(c_obs, r_obs, facecolor='#c9503a', alpha=0.55,
                        edgecolor='#8f3325', linewidth=1.4))
    ax.add_patch(Circle(c_pred, r_pred, facecolor='#e8b04b', alpha=0.55,
                        edgecolor='#a87c22', linewidth=1.4))
    ax.text(0.85, 1.0, 'C', ha='center', va='center', fontsize=13, weight='bold')
    ax.text(1.80, 1.0, 'B', ha='center', va='center', fontsize=13, weight='bold')
    ax.text(2.75, 1.0, 'A', ha='center', va='center', fontsize=13, weight='bold')
    ax.text(0.85, 0.62, 'omisión', ha='center', fontsize=7.8)
    ax.text(1.80, 0.62, 'acierto', ha='center', fontsize=7.8)
    ax.text(2.75, 0.62, 'falsa\nalarma', ha='center', fontsize=7.8)
    ax.text(1.35, 2.22, 'cambio observado', ha='center', fontsize=8.6, color='#8f3325')
    ax.text(2.25, -0.28, 'cambio predicho', ha='center', fontsize=8.6, color='#a87c22')
    ax.text(1.80, -0.95, r'$\mathrm{FoM} = \dfrac{B}{A + B + C}$',
            ha='center', fontsize=11.5)
    ax.set_xlim(-0.15, 3.75)
    ax.set_ylim(-1.45, 2.6)
    ax.set_title('(b) Componentes del Figure of Merit')
    _limpiar(ax)

    fig.savefig(FIGS_OUT / 'esquema_matriz_confusion.png')
    plt.close(fig)
    print('  esquema_matriz_confusion.png')


if __name__ == '__main__':
    print('Generando esquemas del Capitulo 2 en', FIGS_OUT)
    espacio_celular()
    fronteras()
    matriz_confusion()
    print('Listo.')
