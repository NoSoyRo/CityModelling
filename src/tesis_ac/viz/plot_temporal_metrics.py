"""
Graficar métricas de validación temporal (2016-2020) con WoE histórico.

Uso:
  python -m tesis_ac.viz.plot_temporal_metrics \
    --csv reports/validation_hist_all_years.csv \
    --output figures/temporal_metrics_hist.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_temporal_metrics(csv_path: Path, output: Path, figsize=(14, 10)):
    """
    Graficar evolución temporal de métricas con barras de error.
    
    Args:
        csv_path: CSV con columnas [initial_year, target_year, replica, accuracy, kappa, iou, fom, ...]
        output: Ruta de salida PNG
        figsize: Tamaño de figura
    """
    df = pd.read_csv(csv_path)
    
    # Crear etiqueta de período
    df['period'] = df['initial_year'].astype(str) + '→' + df['target_year'].astype(str)
    
    # Agregar por período
    metrics = ['accuracy', 'kappa', 'iou', 'fom']
    agg_df = df.groupby(['initial_year', 'target_year', 'period'])[metrics].agg(['mean', 'std']).reset_index()
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    colors = sns.color_palette("husl", len(agg_df))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Extraer medias y stds
        means = agg_df[(metric, 'mean')]
        stds = agg_df[(metric, 'std')]
        periods = agg_df['period']
        x_pos = range(len(periods))
        
        # Barras con error
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black', linewidth=1.2)
        
        # Línea de tendencia
        ax.plot(x_pos, means, 'o-', color='darkred', linewidth=2, markersize=8, label='Tendencia', zorder=10)
        
        # Etiquetas
        ax.set_xlabel('Período', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
        ax.set_title(f'{metric.upper()} por período', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(periods, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend()
        
        # Añadir valores en las barras
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 0.01, f'{m:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('Validación Temporal con WoE Histórico (1984-2014)', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Guardar
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfica guardada: {output}")
    plt.close()
    
    # Tabla resumen
    print("\nRESUMEN DE MÉTRICAS:")
    print("="*70)
    for _, row in agg_df.iterrows():
        period = row['period']
        print(f"\n{period}:")
        for metric in metrics:
            mean_val = row[(metric, 'mean')]
            std_val = row[(metric, 'std')]
            print(f"  {metric.upper():12s}: {mean_val:.4f} ± {std_val:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Graficar métricas temporales")
    parser.add_argument('--csv', type=Path, required=True,
                        help='CSV con resultados consolidados')
    parser.add_argument('--output', type=Path, 
                        default=Path('figures/temporal_metrics_hist.png'),
                        help='Ruta de salida PNG')
    parser.add_argument('--figsize', type=int, nargs=2, default=[14, 10],
                        help='Tamaño de figura (ancho alto)')
    
    args = parser.parse_args()
    
    plot_temporal_metrics(
        csv_path=args.csv,
        output=args.output,
        figsize=tuple(args.figsize),
    )


if __name__ == '__main__':
    main()
