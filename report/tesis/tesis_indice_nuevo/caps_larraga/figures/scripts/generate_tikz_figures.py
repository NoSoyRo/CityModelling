#!/usr/bin/env python3
"""
Genera figuras TikZ para el Capítulo 5.
Produce .tex compilables standalone → PDF con fuente de LaTeX (Latin Modern).
"""

from pathlib import Path
import subprocess, sys

OUTPUT = Path('report/tesis/tesis_indice_nuevo/caps_larraga/figures/grid_diagrams')
OUTPUT.mkdir(parents=True, exist_ok=True)

# ── Patrón base 10×10 ─────────────────────────────────────────────────────────
BASE = [
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
]
NR, NC = 10, 10
CS = 0.52  # tamaño de celda en cm

# Celda (j, y) en TikZ:  y = NR-1-i  (origen abajo-izquierda)
def cells(grid=BASE):
    u, n = [], []
    for i in range(NR):
        for j in range(NC):
            (u if grid[i][j] == 1 else n).append((j, NR-1-i))
    return u, n

URBAN, NONU = cells()

# ── Preamble / footer comunes ─────────────────────────────────────────────────
def preamble():
    return r"""\documentclass[tikz,border=8pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{calc,positioning,arrows.meta}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
%% Paleta
\definecolor{CU}{RGB}{27,38,49}
\definecolor{CNU}{RGB}{238,241,245}
\definecolor{CBRD}{RGB}{200,205,208}
\definecolor{CNEW}{RGB}{192,57,43}
\definecolor{CMO}{RGB}{36,113,163}
\definecolor{CCTR}{RGB}{211,84,0}
\definecolor{CHIT}{RGB}{29,106,57}
\definecolor{CFAL}{RGB}{183,119,13}
\definecolor{CMIS}{RGB}{21,67,96}
\definecolor{CPU}{RGB}{113,125,126}
\begin{document}
"""

FOOTER = "\\end{document}\n"

# ── Utilidades ────────────────────────────────────────────────────────────────
def f(v):  return f"{v:.4f}cm"

def draw_grid(lines, urban, nonu, x0=0, y0=0,
              special=None, highlight=None, show01=True):
    """
    Dibuja una cuadrícula 10×10 en TikZ.
    special  : {(j,y): (fill, textcol, label)}
    highlight: {(j,y)}  → borde rojo grueso
    """
    def X(j):  return f(x0 + j*CS)
    def Y(y):  return f(y0 + y*CS)
    def X1(j): return f(x0 + (j+1)*CS)
    def Y1(y): return f(y0 + (y+1)*CS)
    def CX(j): return f(x0 + (j+0.5)*CS)
    def CY(y): return f(y0 + (y+0.5)*CS)

    lines += [
        f"  % — grid background",
        f"  \\fill[CNU] ({X(0)},{Y(0)}) rectangle ({X(NC)},{Y(NR)});",
    ]
    for j, y in urban:
        lines.append(f"  \\fill[CU] ({X(j)},{Y(y)}) rectangle ({X1(j)},{Y1(y)});")
    if special:
        for (j, y), (fc, _, _) in special.items():
            lines.append(f"  \\fill[{fc}] ({X(j)},{Y(y)}) rectangle ({X1(j)},{Y1(y)});")
    lines.append(
        f"  \\draw[CBRD,line width=0.35pt,xstep={f(CS)},ystep={f(CS)}]"
        f" ({X(0)},{Y(0)}) grid ({X(NC)},{Y(NR)});"
    )
    if highlight:
        for j, y in highlight:
            lines.append(
                f"  \\draw[CNEW,line width=2.2pt]"
                f" ({X(j)},{Y(y)}) rectangle ({X1(j)},{Y1(y)});"
            )
    if show01:
        fs = "\\fontsize{5.5}{6.5}\\selectfont\\ttfamily\\bfseries"
        for j, y in nonu:
            if special and (j, y) in special: continue
            lines.append(f"  \\node[font={fs},text=CU!55] at ({CX(j)},{CY(y)}) {{0}};")
        for j, y in urban:
            if special and (j, y) in special: continue
            lines.append(f"  \\node[font={fs},text=white] at ({CX(j)},{CY(y)}) {{1}};")
        if special:
            for (j, y), (_, tc, lbl) in special.items():
                lines.append(f"  \\node[font={fs},text={tc}] at ({CX(j)},{CY(y)}) {{{lbl}}};")


def callout(lines, cx, y, text, color, above=False, width="4.2cm"):
    """Caja de texto con borde de color, debajo (o arriba) de una posición."""
    anchor = "north" if not above else "south"
    lines.append(
        f"  \\node[draw={color},fill=white,rounded corners=2pt,"
        f"  text={color},font=\\small\\itshape,"
        f"  text width={width},align=center,inner sep=4pt,"
        f"  anchor={anchor}] at ({f(cx)},{f(y)}) {{{text}}};"
    )


def legend_entry(lines, x, y, color, label, edgecolor=None):
    ec = edgecolor or color
    lines += [
        f"  \\fill[{color}] ({f(x)},{f(y)}) rectangle ({f(x+0.35)},{f(y+0.35)});",
        f"  \\draw[{ec},line width=0.6pt] ({f(x)},{f(y)}) rectangle ({f(x+0.35)},{f(y+0.35)});",
        f"  \\node[font=\\small,anchor=west] at ({f(x+0.48)},{f(y+0.175)}) {{{label}}};",
    ]


# ═════════════════════════════════════════════════════════════════════════════
# F1 — Mapa binario
# ═════════════════════════════════════════════════════════════════════════════
def make_f1():
    lines = [preamble(), "\\begin{tikzpicture}[font=\\sffamily]"]

    # Grid centrado en (0,0)
    draw_grid(lines, URBAN, NONU, x0=0, y0=0, show01=True)

    W = NC * CS  # ancho total

    # Título (encima)
    lines.append(
        f"  \\node[font=\\large\\bfseries,anchor=south] at ({f(W/2)},{f(NR*CS+0.2)})"
        r" {Mapa binario: $U_t(i,j)\in\{0,\,1\}$};"
    )

    # Leyenda (debajo, una sola fila, centrada)
    lx = W/2 - 2.8
    ly = -0.9
    legend_entry(lines, lx,       ly, "CU",  r"Urbano~$(U_t=1)$")
    legend_entry(lines, lx + 2.9, ly, "CNU", r"No~urbano~$(U_t=0)$", edgecolor="CBRD")

    # Nota al pie
    lines.append(
        f"  \\node[font=\\footnotesize\\itshape,text=gray,anchor=north]"
        f" at ({f(W/2)},{f(ly-0.1)})"
        r" {Cada celda representa $\approx30\,\text{m}\times30\,\text{m}$};"
    )

    lines += ["\\end{tikzpicture}", FOOTER]
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# F2 — Definición de transición
# ═════════════════════════════════════════════════════════════════════════════
def make_f2():
    TI, TJ = 3, 2   # fila/col en BASE → TikZ: (TJ, NR-1-TI)
    TY = NR - 1 - TI   # y-coord en TikZ

    GW   = NC * CS      # 5.2 cm ancho de cada grid
    GAP  = 3.0          # cm entre grids (espacio para flecha)
    G2X  = GW + GAP     # origen x del segundo grid
    TW   = G2X + GW     # ancho total

    lines = [preamble(), "\\begin{tikzpicture}[font=\\sffamily]"]

    # ── Grid izquierdo (año t) ─────────────────────────────────────────────
    draw_grid(lines, URBAN, NONU, x0=0, y0=0,
              highlight={(TJ, TY)}, show01=True)

    # ── Grid derecho (año t+1) ────────────────────────────────────────────
    special_r = {(TJ, TY): ("CNEW", "white", "1")}
    draw_grid(lines, URBAN, NONU, x0=G2X, y0=0,
              special=special_r, show01=True)

    # ── Panel central: flecha + regla ─────────────────────────────────────
    ax = GW + GAP/2   # centro x del panel de flecha
    ay = NR*CS/2      # centro y

    # flecha horizontal
    lines.append(
        f"  \\draw[-{{Latex[length=7pt,width=5pt]}},line width=1.8pt,CU]"
        f" ({f(GW+0.35)},{f(ay)}) -- ({f(G2X-0.35)},{f(ay)});"
    )

    # textos en el panel (sin solapamiento: y-coords separadas ≥ 0.55cm)
    panel_items = [
        (ay + 1.30, r"Regla de transición",              "\\small\\itshape",   "CU!70"),
        (ay + 0.60, r"$P_{i,j}>\theta=0{,}75$",         "\\normalsize",       "CU"),
        (ay - 0.55, r"$+$ comp.\ estocástica",           "\\small\\itshape",   "CU!70"),
    ]
    for yp, txt, fs, col in panel_items:
        lines.append(
            f"  \\node[font={fs},text={col},align=center] at ({f(ax)},{f(yp)}) {{{txt}}};"
        )

    # ── Títulos de cada panel ─────────────────────────────────────────────
    TH = NR*CS + 0.35
    lines.append(
        f"  \\node[font=\\large\\bfseries,anchor=south] at ({f(GW/2)},{f(TH)})"
        r" {A\~{n}o $t$};"
    )
    lines.append(
        f"  \\node[font=\\large\\bfseries,anchor=south] at ({f(G2X+GW/2)},{f(TH)})"
        r" {A\~{n}o $t+1$};"
    )

    # ── Título general ────────────────────────────────────────────────────
    lines.append(
        f"  \\node[font=\\large\\bfseries,anchor=south] at ({f(TW/2)},{f(TH+0.55)})"
        r" {Transición: $U_t(i,j)=0\;\rightarrow\;U_{t+1}(i,j)=1$};"
    )

    # ── Callouts debajo de cada grid (y < 0, sin interferir con la cuadrícula) ─
    CY_BOX = -0.55  # y del borde superior del callout

    callout(lines,
            cx=GW/2, y=CY_BOX,
            text=r"Borde rojo $\rightarrow$ celda candidata $(U_t=0)$",
            color="CNEW", width="3.8cm")

    callout(lines,
            cx=G2X + GW/2, y=CY_BOX,
            text=r"Celda roja $\rightarrow$ nueva celda urbana $(U_{t+1}=1)$",
            color="CNEW!80!black", width="3.8cm")

    # ── Leyenda debajo de los callouts ────────────────────────────────────
    LY = -2.05
    lx = TW/2 - 4.5
    legend_entry(lines, lx,       LY, "CU",   r"Urbano~$(U=1)$")
    legend_entry(lines, lx + 2.4, LY, "CNU",  r"No~urbano~$(U=0)$", edgecolor="CBRD")
    legend_entry(lines, lx + 5.0, LY, "CNEW", r"Transición~$0\rightarrow1$")

    lines += ["\\end{tikzpicture}", FOOTER]
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# F3 — Vecindad de Moore
# ═════════════════════════════════════════════════════════════════════════════
def make_f3():
    NR7, NC7 = 7, 7
    CI, CJ = 3, 3   # centro en BASE-7

    # Contexto urbano en la cuadrícula 7×7
    BASE7 = [[0]*7 for _ in range(7)]
    for r, c in [(1,5),(2,4),(2,5),(3,4),(4,4),(4,5),(5,5)]:
        BASE7[r][c] = 1

    urban7 = [(j, NR7-1-i) for i in range(NR7) for j in range(NC7) if BASE7[i][j]==1]
    nonu7  = [(j, NR7-1-i) for i in range(NR7) for j in range(NC7) if BASE7[i][j]==0]
    CJ_TZ, CI_TZ = CJ, NR7-1-CI   # TikZ coords del centro

    moore = {(CJ+dj, CI_TZ+(-di)) for di in (-1,0,1) for dj in (-1,0,1) if (di,dj)!=(0,0)}
    vn    = {(CJ, CI_TZ+1),(CJ, CI_TZ-1),(CJ-1, CI_TZ),(CJ+1, CI_TZ)}
    diag  = moore - vn

    GW  = NC7 * CS
    GAP = 1.8
    G2X = GW + GAP
    TW  = G2X + GW

    lines = [preamble(), "\\begin{tikzpicture}[font=\\sffamily]"]

    def draw7(lines, x0, spec_dict):
        """Dibuja un grid 7×7 con special cells."""
        def X(j):  return f(x0 + j*CS)
        def Y(y):  return f(y*CS)
        def X1(j): return f(x0 + (j+1)*CS)
        def Y1(y): return f((y+1)*CS)
        def CX(j): return f(x0 + (j+0.5)*CS)
        def CY(y): return f((y+0.5)*CS)

        lines.append(f"  \\fill[CNU] ({X(0)},{Y(0)}) rectangle ({X(NC7)},{Y(NR7)});")
        for j, y in urban7:
            lines.append(f"  \\fill[CU] ({X(j)},{Y(y)}) rectangle ({X1(j)},{Y1(y)});")
        for (j, y), (fc, tc, lbl) in spec_dict.items():
            lines.append(f"  \\fill[{fc}] ({X(j)},{Y(y)}) rectangle ({X1(j)},{Y1(y)});")
        lines.append(
            f"  \\draw[CBRD,line width=0.35pt,xstep={f(CS)},ystep={f(CS)}]"
            f" ({X(0)},{Y(0)}) grid ({X(NC7)},{Y(NR7)});"
        )
        # Labels en celdas especiales
        for (j, y), (fc, tc, lbl) in spec_dict.items():
            lines.append(
                f"  \\node[font=\\small\\bfseries,text={tc}]"
                f" at ({CX(j)},{CY(y)}) {{{lbl}}};"
            )

    # ── Panel izquierdo: Moore ────────────────────────────────────────────
    spec_l = {(CJ_TZ, CI_TZ): ("CCTR", "white", "\\footnotesize\\bfseries obj.")}
    for n, (j, y) in enumerate(sorted(moore), 1):
        spec_l[(j, y)] = ("CMO", "white", str(n))
    draw7(lines, x0=0, spec_dict=spec_l)

    # Título panel izquierdo
    TH = NR7*CS + 0.35
    lines += [
        f"  \\node[font=\\normalsize\\bfseries,anchor=south,align=center]"
        f" at ({f(GW/2)},{f(TH)})"
        r" {Vecindad de Moore\\(8 celdas adyacentes)};",
    ]

    # ── Panel derecho: Moore vs. Von Neumann ─────────────────────────────
    spec_r = {(CJ_TZ, CI_TZ): ("CCTR", "white", "\\footnotesize\\bfseries obj.")}
    for j, y in vn:
        spec_r[(j, y)] = ("CMO!50!black!60!teal", "white", "VN")
    for j, y in diag:
        spec_r[(j, y)] = ("CMO", "white", "M")
    draw7(lines, x0=G2X, spec_dict=spec_r)

    lines += [
        f"  \\node[font=\\normalsize\\bfseries,anchor=south,align=center]"
        f" at ({f(G2X+GW/2)},{f(TH)})"
        r" {Moore vs.\ Von~Neumann\\(ortogonales + diagonales)};",
    ]

    # Título general
    lines.append(
        f"  \\node[font=\\large\\bfseries,anchor=south]"
        f" at ({f(TW/2)},{f(TH+0.7)})"
        r" {Vecindad de Moore: $N_{i,j}$ en el AC};"
    )

    # ── Leyenda ───────────────────────────────────────────────────────────
    LY = -0.8
    lx = TW/2 - 5.2
    entries = [
        ("CCTR",                   r"Celda objetivo"),
        ("CMO",                    r"Vecinos Moore (8)"),
        ("CMO!50!black!60!teal",   r"Von Neumann — ortog. (4)"),
        ("CU",                     r"Urbano"),
        ("CNU",                    r"No urbano"),
    ]
    gap = 2.55
    for k, (col, lbl) in enumerate(entries):
        legend_entry(lines, lx + k*gap, LY, col, lbl,
                     edgecolor="CBRD" if col=="CNU" else col)

    lines += ["\\end{tikzpicture}", FOOTER]
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# F6 — Un paso del AC
# ═════════════════════════════════════════════════════════════════════════════
def make_f6():
    # Celdas candidatas: no-urbanas con ≥3 vecinos urbanos
    cands = []
    for i in range(NR):
        for j in range(NC):
            if BASE[i][j] == 0:
                nbrs = sum(
                    BASE[i+di][j+dj]
                    for di in (-1,0,1) for dj in (-1,0,1)
                    if (di,dj)!=(0,0) and 0<=i+di<NR and 0<=j+dj<NC)
                if nbrs >= 3:
                    cands.append((j, NR-1-i))
    new_cells = set(sorted(cands)[:5])

    GW  = NC*CS
    GAP = 3.2
    G2X = GW + GAP
    TW  = G2X + GW
    AX  = GW + GAP/2   # centro x del panel de regla

    lines = [preamble(), "\\begin{tikzpicture}[font=\\sffamily]"]

    # ── Grid t ────────────────────────────────────────────────────────────
    draw_grid(lines, URBAN, NONU, x0=0, y0=0,
              highlight=new_cells, show01=False)

    # ── Grid t+1 ──────────────────────────────────────────────────────────
    spec2 = {(j, y): ("CNEW", "white", r"$\uparrow$") for j, y in new_cells}
    draw_grid(lines, URBAN, NONU, x0=G2X, y0=0,
              special=spec2, show01=False)

    # ── Flecha ────────────────────────────────────────────────────────────
    ay = NR*CS/2
    lines.append(
        f"  \\draw[-{{Latex[length=7pt,width=5pt]}},line width=1.8pt,CU]"
        f" ({f(GW+0.3)},{f(ay)}) -- ({f(G2X-0.3)},{f(ay)});"
    )

    # ── Pasos de la regla (sin solapamiento, separados 0.6 cm) ────────────
    steps = [
        (ay + 1.50, r"\textbf{Recalcular}",         "\\small"),
        (ay + 0.90, r"7 variables espaciales",       "\\small\\itshape"),
        (ay + 0.30, r"$\downarrow$",                 "\\normalsize"),
        (ay - 0.25, r"Aplicar WoE $\times$ IV",      "\\small\\itshape"),
        (ay - 0.80, r"$\downarrow$",                 "\\normalsize"),
        (ay - 1.35, r"$P_{i,j}>\theta=0{,}75$",     "\\small"),
    ]
    for yp, txt, fs in steps:
        lines.append(
            f"  \\node[font={fs},text=CU,align=center] at ({f(AX)},{f(yp)}) {{{txt}}};"
        )

    # ── Títulos ───────────────────────────────────────────────────────────
    TH = NR*CS + 0.35
    lines += [
        f"  \\node[font=\\large\\bfseries,anchor=south] at ({f(GW/2)},{f(TH)})"
        r" {Estado $U_t$};",
        f"  \\node[font=\\large\\bfseries,anchor=south] at ({f(G2X+GW/2)},{f(TH)})"
        r" {Estado $U_{t+1}$};",
        f"  \\node[font=\\large\\bfseries,anchor=south] at ({f(TW/2)},{f(TH+0.6)})"
        r" {Un paso del Aut\'{o}mata Celular: propagaci\'{o}n del frente urbano};",
    ]

    # ── Callouts debajo de cada grid ──────────────────────────────────────
    callout(lines, cx=GW/2, y=-0.5,
            text=r"Borde rojo: celdas candidatas a transitar",
            color="CNEW", width="3.8cm")
    callout(lines, cx=G2X+GW/2, y=-0.5,
            text=r"Nuevas celdas urbanas (rojo)",
            color="CNEW!80!black", width="3.2cm")

    # ── Leyenda ───────────────────────────────────────────────────────────
    LY = -1.9
    lx = TW/2 - 5.0
    legend_entry(lines, lx,       LY, "CU",   r"Urbano")
    legend_entry(lines, lx + 1.9, LY, "CNU",  r"No urbano",     edgecolor="CBRD")
    legend_entry(lines, lx + 4.0, LY, "CNU",  r"Candidata (borde rojo)", edgecolor="CNEW")
    legend_entry(lines, lx + 7.3, LY, "CNEW", r"Nueva celda urbana")

    lines += ["\\end{tikzpicture}", FOOTER]
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# F8 — Componentes del FoM
# ═════════════════════════════════════════════════════════════════════════════
def make_f8():
    obs_new  = [(2,1),(3,1),(3,2),(6,1),(7,3),(7,4)]
    pred_new = [(2,1),(3,1),(4,1),(5,1),(6,2),(6,1)]

    grid_obs  = [row[:] for row in BASE]
    grid_pred = [row[:] for row in BASE]
    for r, c in obs_new:  grid_obs[r][c]  = 1
    for r, c in pred_new: grid_pred[r][c] = 1

    hits, false_a, miss, perm = set(), set(), set(), set()
    for i in range(NR):
        for j in range(NC):
            tj = (j, NR-1-i)
            if BASE[i][j] == 1:
                perm.add(tj)
            else:
                o = grid_obs[i][j]; p = grid_pred[i][j]
                if   p==1 and o==1: hits.add(tj)
                elif p==1 and o==0: false_a.add(tj)
                elif p==0 and o==1: miss.add(tj)

    B, A, Cn = len(hits), len(false_a), len(miss)
    FoM = B/(A+B+Cn) if (A+B+Cn)>0 else 0.0

    GW   = NC*CS
    GAP  = 1.8
    RX   = GW + GAP   # origen x del diagrama derecho
    RW   = 8.0        # ancho del diagrama derecho
    TW   = RX + RW

    lines = [preamble(), "\\begin{tikzpicture}[font=\\sffamily]"]

    # ── Grid izquierdo: mapa de diferencia ────────────────────────────────
    urban_perm = list(perm)
    nonu_rest  = [c for c in NONU if c not in hits|false_a|miss]
    spec = {}
    for c in perm:    spec[c] = ("CPU", "white", "")
    for c in hits:    spec[c] = ("CHIT","white","B")
    for c in false_a: spec[c] = ("CFAL","white","A")
    for c in miss:    spec[c] = ("CMIS","white","C")

    draw_grid(lines, [], NONU, x0=0, y0=0, special=spec, show01=False)

    # Título panel izquierdo
    TH = NR*CS + 0.35
    lines.append(
        f"  \\node[font=\\normalsize\\bfseries,anchor=south]"
        f" at ({f(GW/2)},{f(TH)})"
        r" {Mapa de diferencia --- predicho vs.\ observado};"
    )

    # ── Diagrama derecho: 3 cajas A | B | C ──────────────────────────────
    # Cajas horizontales, cada una en su propia banda sin solapamiento
    BOX_Y0 = 0.6        # y inferior de las cajas
    BOX_H  = NR*CS - 1.2   # altura de cada caja
    BOX_TOP = BOX_Y0 + BOX_H

    box_defs = [
        # (x0_rel, width, letra, fill, border, titulo,  desc1,              desc2)
        (0.0, 2.3, "A", "CFAL!18",  "CFAL",  "Falsa alarma",
         "Predijo,",        f"no ocurrió\\quad $n={A}$"),
        (2.5, 3.0, "B", "CHIT!18",  "CHIT",  "Acierto",
         "Predijo y",       f"ocurrió\\quad $n={B}$"),
        (5.7, 2.3, "C", "CMIS!18",  "CMIS",  "Omisión",
         "Ocurrió,",        f"no predijo\\quad $n={Cn}$"),
    ]

    for x0r, w, letra, fill, bord, titulo, d1, d2 in box_defs:
        x0 = RX + x0r
        cx = x0 + w/2
        # Caja de fondo
        lines.append(
            f"  \\draw[{bord},fill={fill},rounded corners=3pt,line width=1.5pt]"
            f" ({f(x0)},{f(BOX_Y0)}) rectangle ({f(x0+w)},{f(BOX_TOP)});"
        )
        # Título encima de la caja (fuera del rectángulo → no choca)
        lines.append(
            f"  \\node[font=\\small\\bfseries,text={bord},anchor=south]"
            f" at ({f(cx)},{f(BOX_TOP+0.08)}) {{{titulo}}};"
        )
        # Círculo con letra (centrado en la mitad superior de la caja)
        cy_circle = BOX_Y0 + BOX_H * 0.68
        lines.append(
            f"  \\fill[{bord}] ({f(cx)},{f(cy_circle)}) circle (0.42cm);"
        )
        lines.append(
            f"  \\node[font=\\large\\bfseries,text=white]"
            f" at ({f(cx)},{f(cy_circle)}) {{{letra}}};"
        )
        # Descripción en la mitad inferior (separada del círculo ≥ 0.5cm)
        y_d1 = BOX_Y0 + 0.80
        y_d2 = BOX_Y0 + 0.28
        lines.append(
            f"  \\node[font=\\small,text=CU,anchor=base]"
            f" at ({f(cx)},{f(y_d1)}) {{{d1}}};"
        )
        lines.append(
            f"  \\node[font=\\small,text=CU,anchor=base]"
            f" at ({f(cx)},{f(y_d2)}) {{{d2}}};"
        )

    # Título panel derecho
    lines.append(
        f"  \\node[font=\\normalsize\\bfseries,anchor=south]"
        f" at ({f(RX+RW/2)},{f(TH)})"
        r" {Figure of Merit --- Pontius et al.};"
    )

    # Fórmula debajo de las cajas (espacio libre garantizado)
    FY = BOX_Y0 - 0.9
    lines.append(
        f"  \\node[draw=CBRD,fill=white,rounded corners=2pt,inner sep=5pt,"
        f"  font=\\normalsize,anchor=north]"
        f" at ({f(RX+RW/2)},{f(FY)})"
        r" {$\mathrm{FoM}=\dfrac{B}{A+B+C}="
        f"\\dfrac{{{B}}}{{{A}+{B}+{Cn}}}={FoM:.3f}$}};"
    )

    # ── Leyenda del panel izquierdo (debajo del grid) ─────────────────────
    LY = -0.8
    lx = 0.0
    entries_l = [
        ("CHIT", f"B Acierto~(predicho y ocurrió)~$n={B}$"),
        ("CFAL", f"A Falsa alarma~(predicho, no ocurrió)~$n={A}$"),
        ("CMIS", f"C Omisión~(ocurrió, no predicho)~$n={Cn}$"),
        ("CPU",  r"Urbano permanente (no entra al FoM)"),
        ("CNU",  r"No urbano permanente"),
    ]
    for k, (col, lbl) in enumerate(entries_l):
        ky = LY - k*0.55
        legend_entry(lines, lx, ky, col, lbl,
                     edgecolor="CBRD" if col=="CNU" else col)

    lines += ["\\end{tikzpicture}", FOOTER]
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# Compilar con pdflatex
# ═════════════════════════════════════════════════════════════════════════════
def compile_tex(name, content):
    tex_path = OUTPUT / f"{name}.tex"
    tex_path.write_text(content, encoding="utf-8")
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode",
         "-output-directory", str(OUTPUT), str(tex_path)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ⚠ Error compilando {name}:")
        # Mostrar solo líneas con '!' o 'Error'
        for line in result.stdout.splitlines():
            if line.startswith('!') or 'error' in line.lower():
                print(f"    {line}")
        return False
    # Limpiar auxiliares
    for ext in (".aux", ".log"):
        (OUTPUT / f"{name}{ext}").unlink(missing_ok=True)
    print(f"  ✅  {name}.pdf")
    return True


if __name__ == "__main__":
    print("Generando figuras TikZ — Capítulo 5\n")
    figures = [
        ("F1_mapa_binario",     make_f1),
        ("F2_transicion",       make_f2),
        ("F3_vecindad_moore",   make_f3),
        ("F6_paso_ac",          make_f6),
        ("F8_componentes_fom",  make_f8),
    ]
    ok = 0
    for name, maker in figures:
        print(f"  Generando {name}...")
        content = maker()
        if compile_tex(name, content):
            ok += 1

    print(f"\n{ok}/{len(figures)} figuras generadas.")
    print(f"PDFs en: {OUTPUT.resolve()}")
