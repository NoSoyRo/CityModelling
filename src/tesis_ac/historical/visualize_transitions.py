"""
Visualización y análisis de transiciones históricas extraídas.

Script para explorar patrones espaciales y temporales de crecimiento urbano.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
import argparse
import logging
import sys

# Asegurar que podemos importar desde el módulo correcto
from src.tesis_ac.historical.extract_transitions import HistoricalTransitionsDataset

logger = logging.getLogger(__name__)


def plot_transition_timeline(dataset: HistoricalTransitionsDataset, output_dir: Optional[Path] = None):
    """Visualiza una línea temporal de transiciones (conteos, crecimiento y área).

    Argumentos:
        dataset: Dataset con transiciones.
        output_dir: Directorio para guardar figuras. Si es ``None``, muestra en pantalla.
    """
    df = dataset.get_summary_dataframe()
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 1. Número de transiciones por año
    ax1 = axes[0]
    ax1.bar(df['year_from'], df['n_transitions'], color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_title('Transiciones Urbanas por Año', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Año')
    ax1.set_ylabel('Número de Transiciones (0→1)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Tasa de crecimiento
    ax2 = axes[1]
    colors = ['green' if x > 0 else 'red' for x in df['growth_rate_%']]
    ax2.bar(df['year_from'], df['growth_rate_%'], color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_title('Tasa de Crecimiento Urbano Anual', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Año')
    ax2.set_ylabel('Crecimiento (%)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Área urbana total
    ax3 = axes[2]
    ax3.plot(df['year_from'], df['urban_t0'], 'o-', linewidth=2, markersize=6, 
             color='darkgreen', label='Área urbana inicial')
    ax3.fill_between(df['year_from'], df['urban_t0'], alpha=0.3, color='darkgreen')
    ax3.set_title('Evolución del Área Urbana Total', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Año')
    ax3.set_ylabel('Píxeles Urbanos')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'transition_timeline.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Guardado: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_spatial_transition_map(
    dataset: HistoricalTransitionsDataset,
    year_from: int,
    year_to: int,
    output_dir: Optional[Path] = None
):
    """Visualiza el mapa espacial de transiciones para un período específico.

    Argumentos:
        dataset: Dataset con transiciones.
        year_from: Año inicial.
        year_to: Año final.
        output_dir: Directorio para guardar. Si es ``None``, muestra en pantalla.
    """
    # Buscar el período
    transition = None
    for t in dataset.transitions:
        if t.year_from == year_from and t.year_to == year_to:
            transition = t
            break
    
    if transition is None:
        raise ValueError(f"Período {year_from}→{year_to} no encontrado")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Estado inicial (t0)
    ax1 = axes[0]
    ax1.imshow(transition.grid_t0, cmap='RdYlGn_r', vmin=0, vmax=1)
    ax1.set_title(f'Estado {year_from}\n({transition.n_urban_t0:,} píxeles urbanos)', 
                  fontweight='bold')
    ax1.axis('off')
    
    # 2. Transiciones
    ax2 = axes[1]
    # Overlay: urbano en gris, transiciones en rojo
    overlay = np.zeros((*transition.grid_t0.shape, 3))
    overlay[transition.grid_t0 == 1] = [0.7, 0.7, 0.7]  # Urbano existente en gris
    overlay[transition.transitions] = [1.0, 0.0, 0.0]   # Nuevas transiciones en rojo
    
    ax2.imshow(overlay)
    ax2.set_title(f'Transiciones {year_from}→{year_to}\n({transition.n_transitions:,} nuevas celdas)', 
                  fontweight='bold', color='red')
    ax2.axis('off')
    
    # 3. Estado final (t1)
    ax3 = axes[2]
    ax3.imshow(transition.grid_t1, cmap='RdYlGn_r', vmin=0, vmax=1)
    ax3.set_title(f'Estado {year_to}\n({transition.n_urban_t1:,} píxeles urbanos)', 
                  fontweight='bold')
    ax3.axis('off')
    
    plt.suptitle(f'Análisis Espacial de Transiciones Urbanas: {year_from}→{year_to}',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / f'spatial_transitions_{year_from}_{year_to}.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        logger.info(f"Guardado: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_aggregated_transitions_heatmap(
    dataset: HistoricalTransitionsDataset,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    output_dir: Optional[Path] = None
):
    """Crea un heatmap de frecuencia de transiciones agregadas.

    Muestra dónde ocurrieron más transiciones a lo largo del tiempo.

    Argumentos:
        dataset: Dataset con transiciones.
        min_year: Año mínimo a incluir (opcional).
        max_year: Año máximo a incluir (opcional).
        output_dir: Directorio para guardar. Si es ``None``, muestra en pantalla.
    """
    # Crear mapa de frecuencia de transiciones
    transition_frequency = np.zeros(dataset.grid_shape, dtype=int)
    
    for t in dataset.transitions:
        if min_year is not None and t.year_from < min_year:
            continue
        if max_year is not None and t.year_to > max_year:
            continue
        
        transition_frequency[t.transitions] += 1
    
    # Visualizar
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    im = ax.imshow(transition_frequency, cmap='hot', interpolation='bilinear')
    ax.set_title(f'Frecuencia de Transiciones Urbanas ({min_year or "inicio"}→{max_year or "fin"})',
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Número de veces que ocurrió transición', rotation=270, labelpad=20)
    
    # Estadísticas
    n_transitions_total = np.sum(transition_frequency > 0)
    max_freq = np.max(transition_frequency)
    
    textstr = f'Celdas con transiciones: {n_transitions_total:,}\n'
    textstr += f'Frecuencia máxima: {max_freq}\n'
    textstr += f'Frecuencia media: {np.mean(transition_frequency[transition_frequency > 0]):.2f}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'transition_frequency_heatmap.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Guardado: {output_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_transition_patterns(dataset: HistoricalTransitionsDataset) -> dict:
    """Analiza patrones estadísticos agregados de las transiciones.

    Argumentos:
        dataset: Dataset con transiciones.

    Retorna:
        Diccionario con estadísticas agregadas.
    """
    df = dataset.get_summary_dataframe()
    
    stats = {
        'total_periods': len(dataset.transitions),
        'total_transitions': dataset.total_transitions,
        'avg_transitions_per_year': df['n_transitions'].mean(),
        'std_transitions_per_year': df['n_transitions'].std(),
        'max_transitions_year': df.loc[df['n_transitions'].idxmax(), 'year_from'],
        'max_transitions_count': df['n_transitions'].max(),
        'min_transitions_year': df.loc[df['n_transitions'].idxmin(), 'year_from'],
        'min_transitions_count': df['n_transitions'].min(),
        'avg_growth_rate': df['growth_rate_%'].mean(),
        'growth_rate_volatility': df['growth_rate_%'].std(),
        'positive_growth_periods': (df['growth_rate_%'] > 0).sum(),
        'negative_growth_periods': (df['growth_rate_%'] < 0).sum(),
    }
    
    return stats


def main():
    """CLI para visualización de transiciones."""
    parser = argparse.ArgumentParser(
        description="Visualizar transiciones urbanas históricas"
    )
    parser.add_argument(
        '--dataset',
        type=Path,
        required=True,
        help='Archivo .pkl con dataset de transiciones'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('analysis/visualizations'),
        help='Directorio para guardar visualizaciones'
    )
    parser.add_argument(
        '--specific-year',
        type=int,
        nargs=2,
        metavar=('YEAR_FROM', 'YEAR_TO'),
        help='Visualizar período específico (ej: --specific-year 2000 2010)'
    )
    
    args = parser.parse_args()
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Cargar dataset
    logger.info(f"Cargando dataset: {args.dataset}")
    dataset = HistoricalTransitionsDataset.load(args.dataset)
    
    logger.info(f"Dataset cargado: {len(dataset.transitions)} períodos")
    
    # Crear directorio de salida
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generar visualizaciones
    logger.info("Generando línea temporal...")
    plot_transition_timeline(dataset, args.output_dir)
    
    logger.info("Generando heatmap de frecuencia...")
    plot_aggregated_transitions_heatmap(dataset, output_dir=args.output_dir)
    
    # Visualizaciones específicas
    if args.specific_year:
        year_from, year_to = args.specific_year
        logger.info(f"Generando mapa espacial {year_from}→{year_to}...")
        plot_spatial_transition_map(dataset, year_from, year_to, args.output_dir)
    else:
        # Visualizar algunos períodos interesantes
        df = dataset.get_summary_dataframe()
        
        # Año con más transiciones
        max_trans_idx = df['n_transitions'].idxmax()
        year_from = int(df.loc[max_trans_idx, 'year_from'])
        year_to = int(df.loc[max_trans_idx, 'year_to'])
        logger.info(f"Visualizando período con más transiciones: {year_from}→{year_to}")
        plot_spatial_transition_map(dataset, year_from, year_to, args.output_dir)
        
        # Visualizar década reciente
        logger.info("Visualizando período reciente: 2010→2020")
        plot_aggregated_transitions_heatmap(dataset, min_year=2010, max_year=2020, 
                                            output_dir=args.output_dir)
    
    # Análisis estadístico
    logger.info("\nAnalizando patrones...")
    stats = analyze_transition_patterns(dataset)
    
    print("\n" + "="*60)
    print("ESTADÍSTICAS DE TRANSICIONES")
    print("="*60)
    for key, value in stats.items():
        print(f"{key:.<40} {value}")
    print("="*60)
    
    logger.info(f"\n✅ Visualizaciones guardadas en: {args.output_dir}")


if __name__ == "__main__":
    main()
