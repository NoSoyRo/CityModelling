"""
Visualización de métricas comprehensivas para comparar modelos.

Genera gráficos comparativos de:
- Accuracy, Precision, Recall, F1, Kappa
- Figure of Merit (FoM)
- Tasas de cambio
- Evolución temporal
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

def plot_spatial_metrics(df: pd.DataFrame, output_file: Optional[Path] = None):
    """
    Gráfico de métricas espaciales globales.
    
    Args:
        df: DataFrame con métricas
        output_file: Archivo de salida (opcional)
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Métricas Espaciales Globales - Validación 2016-2020', 
                 fontsize=16, fontweight='bold')
    
    metrics = [
        ('accuracy', 'Accuracy (Aproximación)', axes[0, 0]),
        ('precision', 'Precision', axes[0, 1]),
        ('recall', 'Recall (Sensibilidad)', axes[0, 2]),
        ('f1_score', 'F1-Score', axes[1, 0]),
        ('kappa', 'Kappa de Cohen', axes[1, 1])
    ]
    
    for metric, title, ax in metrics:
        values = df[metric].values
        periods = df['period'].values
        
        ax.plot(range(len(periods)), values, 
               marker='o', linewidth=2, markersize=8, 
               color='#2E86AB', label=metric)
        ax.set_xticks(range(len(periods)))
        ax.set_xticklabels(periods, rotation=45)
        ax.set_ylabel(title, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=df[metric].mean(), color='red', 
                  linestyle='--', alpha=0.5, label=f'Promedio: {df[metric].mean():.3f}')
        ax.legend(fontsize=9)
    
    # Remover subplot vacío
    fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico guardado: {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_fom_analysis(df: pd.DataFrame, output_file: Optional[Path] = None):
    """
    Gráfico de análisis de Figure of Merit (FoM).
    
    Args:
        df: DataFrame con métricas
        output_file: Archivo de salida (opcional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Figure of Merit (FoM) - Métrica Crítica para Expansión Urbana', 
                 fontsize=14, fontweight='bold')
    
    periods = df['period'].values
    x_pos = np.arange(len(periods))
    
    # Panel 1: FoM con barras y línea
    ax1 = axes[0]
    bars = ax1.bar(x_pos, df['fom'].values, color='#A23B72', alpha=0.7, 
                   label='FoM por periodo')
    ax1.axhline(y=df['fom'].mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Promedio: {df["fom"].mean():.4f}')
    
    # Zonas de interpretación
    ax1.axhspan(0.50, 1.0, alpha=0.1, color='green', label='Excelente (>0.50)')
    ax1.axhspan(0.40, 0.50, alpha=0.1, color='yellow', label='Muy bueno (0.40-0.50)')
    ax1.axhspan(0.25, 0.40, alpha=0.1, color='orange', label='Aceptable (0.25-0.40)')
    ax1.axhspan(0.0, 0.25, alpha=0.1, color='red', label='Mejorable (<0.25)')
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(periods, rotation=45)
    ax1.set_ylabel('FoM', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(loc='upper right', fontsize=9)
    
    # Agregar valores sobre las barras
    for i, (bar, val) in enumerate(zip(bars, df['fom'].values)):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Panel 2: Componentes A, B, C
    ax2 = axes[1]
    width = 0.25
    
    ax2.bar(x_pos - width, df['A_correct_change'].values, width, 
           label='A: Cambio correcto', color='green', alpha=0.7)
    ax2.bar(x_pos, df['B_false_change'].values, width, 
           label='B: Falso cambio', color='red', alpha=0.7)
    ax2.bar(x_pos + width, df['C_missed_change'].values, width, 
           label='C: Cambio perdido', color='orange', alpha=0.7)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(periods, rotation=45)
    ax2.set_ylabel('Número de píxeles', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico guardado: {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_change_rates(df: pd.DataFrame, output_file: Optional[Path] = None):
    """
    Gráfico de tasas de cambio real vs predicha.
    
    Args:
        df: DataFrame con métricas
        output_file: Archivo de salida (opcional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Tasas de Cambio Urbano: Real vs Predicho', 
                 fontsize=14, fontweight='bold')
    
    periods = df['period'].values
    x_pos = np.arange(len(periods))
    
    ax.plot(x_pos, df['change_rate_real'].values * 100, 
           marker='o', linewidth=2, markersize=10, 
           color='#2E86AB', label='Cambio Real')
    ax.plot(x_pos, df['change_rate_predicted'].values * 100, 
           marker='s', linewidth=2, markersize=10, 
           color='#A23B72', label='Cambio Predicho')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(periods, rotation=45)
    ax.set_ylabel('Tasa de Cambio (%)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Agregar valores
    for i in range(len(periods)):
        real_val = df['change_rate_real'].values[i] * 100
        pred_val = df['change_rate_predicted'].values[i] * 100
        ax.text(i, real_val + 0.02, f'{real_val:.2f}%', 
               ha='center', va='bottom', fontsize=9)
        ax.text(i, pred_val - 0.05, f'{pred_val:.2f}%', 
               ha='center', va='top', fontsize=9)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico guardado: {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_heatmap(df: pd.DataFrame, output_file: Optional[Path] = None):
    """
    Heatmap de matriz de confusión promedio.
    
    Args:
        df: DataFrame con métricas
        output_file: Archivo de salida (opcional)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Matriz de Confusión Promedio - Validación 2016-2020', 
                 fontsize=14, fontweight='bold')
    
    # Calcular promedios
    tp_avg = df['TP'].mean()
    tn_avg = df['TN'].mean()
    fp_avg = df['FP'].mean()
    fn_avg = df['FN'].mean()
    
    total = tp_avg + tn_avg + fp_avg + fn_avg
    
    # Matriz normalizada
    confusion_matrix = np.array([
        [tn_avg/total * 100, fp_avg/total * 100],
        [fn_avg/total * 100, tp_avg/total * 100]
    ])
    
    im = ax.imshow(confusion_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Etiquetas
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicho: No Urbano', 'Predicho: Urbano'], fontsize=11)
    ax.set_yticklabels(['Real: No Urbano', 'Real: Urbano'], fontsize=11)
    
    # Valores en celdas
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f'{confusion_matrix[i, j]:.2f}%',
                          ha="center", va="center", color="black", 
                          fontweight='bold', fontsize=14)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Porcentaje (%)', rotation=270, labelpad=20, fontsize=11)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico guardado: {output_file}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualización de métricas comprehensivas"
    )
    parser.add_argument(
        '--metrics-csv',
        type=Path,
        required=True,
        help='Archivo CSV con métricas calculadas'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('figures/metrics'),
        help='Directorio de salida para gráficos'
    )
    
    args = parser.parse_args()
    
    if not args.metrics_csv.exists():
        raise FileNotFoundError(f"Archivo no existe: {args.metrics_csv}")
    
    # Cargar métricas
    print(f"Cargando métricas desde {args.metrics_csv}")
    df = pd.read_csv(args.metrics_csv)
    
    # Crear directorio de salida
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generar gráficos
    print("\nGenerando gráficos...")
    
    print("  1. Métricas espaciales globales...")
    plot_spatial_metrics(df, args.output_dir / 'spatial_metrics.png')
    
    print("  2. Análisis de Figure of Merit (FoM)...")
    plot_fom_analysis(df, args.output_dir / 'fom_analysis.png')
    
    print("  3. Tasas de cambio...")
    plot_change_rates(df, args.output_dir / 'change_rates.png')
    
    print("  4. Matriz de confusión...")
    plot_confusion_heatmap(df, args.output_dir / 'confusion_matrix.png')
    
    print(f"\n✓ Todos los gráficos guardados en {args.output_dir}")


if __name__ == '__main__':
    main()
