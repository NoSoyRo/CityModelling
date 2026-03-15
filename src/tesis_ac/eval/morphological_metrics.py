"""
Métricas morfológicas para análisis de patrones espaciales urbanos.

Evalúa la ESTRUCTURA del crecimiento urbano, no solo la cantidad.
Incluye:
- Número de parches urbanos (fragmentación)
- Tamaño promedio y distribución de parches
- Índice de agregación (compacidad)
- Fragmentación del paisaje
- Dimensión fractal (opcional)
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import distance
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


def identify_urban_patches(grid: np.ndarray) -> Tuple[np.ndarray, int]:
    """Identifica parches urbanos conectados en un grid binario.

    Nota:
        La función fuerza conectividad de 8 vecinos (estructura 3x3 de unos)
        para el etiquetado de componentes conectados.

    Argumentos:
        grid: Grid binario (1=urbano, 0=no urbano) o numérico (se binariza con
            ``grid > 0``).

    Retorna:
        labeled: Grid donde cada parche tiene un ID entero único.
        num_patches: Número total de parches urbanos.
    """
    # Binarizar
    binary = (grid > 0).astype(int)
    
    # Estructura de conectividad (8-vecinos: horizontal, vertical, diagonal)
    structure = np.ones((3, 3), dtype=int)
    
    # Connected Component Labeling
    labeled, num_patches = ndimage.label(binary, structure=structure)
    
    logger.info(f"Identificados {num_patches:,} parches urbanos")
    
    return labeled, num_patches


def calculate_patch_sizes(labeled: np.ndarray, num_patches: int) -> np.ndarray:
    """Calcula el tamaño (número de celdas) de cada parche.

    Argumentos:
        labeled: Grid con parches etiquetados (salida de :func:`identify_urban_patches`).
        num_patches: Número de parches (actualmente no se usa, se conserva por
            claridad/compatibilidad).

    Retorna:
        Array con tamaños de parches (se excluye el background/etiqueta 0).
    """
    # Contar píxeles por etiqueta
    patch_sizes = np.bincount(labeled.ravel())
    
    # Remover background (label=0)
    if len(patch_sizes) > 1:
        patch_sizes = patch_sizes[1:]  # Excluir background
    
    return patch_sizes


def calculate_morphological_metrics(grid: np.ndarray) -> Dict[str, float]:
    """Calcula un conjunto de métricas morfológicas para un patrón urbano.

    Argumentos:
        grid: Grid binario (1=urbano, 0=no urbano) o numérico (se binariza con
            ``grid > 0``).

    Retorna:
        Diccionario con métricas (densidad, fragmentación, índices de forma,
        agregación y complejidad).
    """
    logger.info("Calculando métricas morfológicas...")
    
    # 1. Identificar parches
    labeled, num_patches = identify_urban_patches(grid)
    
    if num_patches == 0:
        logger.warning("No se encontraron parches urbanos")
        return {
            'num_patches': 0,
            'mean_patch_size': 0.0,
            'median_patch_size': 0.0,
            'std_patch_size': 0.0,
            'largest_patch_size': 0,
            'smallest_patch_size': 0,
            'aggregation_index': 0.0,
            'fragmentation_index': 0.0,
            'urban_density': 0.0,
            'perimeter_area_ratio': 0.0
        }
    
    # 2. Tamaños de parches
    patch_sizes = calculate_patch_sizes(labeled, num_patches)
    
    mean_size = float(np.mean(patch_sizes))
    median_size = float(np.median(patch_sizes))
    std_size = float(np.std(patch_sizes))
    largest = int(np.max(patch_sizes))
    smallest = int(np.min(patch_sizes))
    
    # 3. Índice de agregación (AI)
    # AI mide qué tan agrupadas están las celdas urbanas
    # AI = 1 → máxima agregación (un solo parche)
    # AI → 0 → máxima dispersión (píxeles aislados)
    total_urban = np.sum(grid > 0)
    ai = calculate_aggregation_index(grid, labeled, num_patches)
    
    # 4. Índice de fragmentación
    # Fragmentación = número de parches / área urbana total
    # Alto valor → muchos parches pequeños (disperso)
    # Bajo valor → pocos parches grandes (compacto)
    fragmentation = num_patches / total_urban if total_urban > 0 else 0.0
    
    # 5. Densidad urbana global
    total_pixels = grid.size
    urban_density = total_urban / total_pixels
    
    # 6. Ratio perímetro/área (compacidad)
    # Bajo ratio → formas compactas (círculos)
    # Alto ratio → formas alargadas o fragmentadas
    pa_ratio = calculate_perimeter_area_ratio(grid)
    
    metrics = {
        'num_patches': int(num_patches),
        'mean_patch_size': float(mean_size),
        'median_patch_size': float(median_size),
        'std_patch_size': float(std_size),
        'largest_patch_size': int(largest),
        'smallest_patch_size': int(smallest),
        'aggregation_index': float(ai),
        'fragmentation_index': float(fragmentation * 1000),  # Por mil píxeles
        'urban_density': float(urban_density),
        'perimeter_area_ratio': float(pa_ratio)
    }
    
    logger.info(f"  Parches: {num_patches:,}")
    logger.info(f"  Tamaño promedio: {mean_size:.1f} píxeles")
    logger.info(f"  Agregación: {ai:.4f}")
    logger.info(f"  Fragmentación: {fragmentation*1000:.4f} parches/1000px")
    
    return metrics


def calculate_aggregation_index(grid: np.ndarray, 
                                labeled: np.ndarray, 
                                num_patches: int) -> float:
    """
    Calcula índice de agregación (AI).
    
    AI mide la tendencia de las celdas urbanas a agruparse.
    
    AI = (g_ii / max_g_ii)
    
    Donde:
    - g_ii = número de adyacencias urbano-urbano observadas
    - max_g_ii = máximo posible de adyacencias
    
    Argumentos:
        grid: Grid binario (1=urbano, 0=no urbano) o numérico (se binariza con
            ``grid > 0``).
        labeled: Grid etiquetado (no se usa; se conserva por compatibilidad).
        num_patches: Número de parches (no se usa; se conserva por compatibilidad).

    Retorna:
        AI ∈ [0, 1], donde 1 indica máxima agregación.
    """
    binary = (grid > 0).astype(int)
    total_urban = np.sum(binary)
    
    if total_urban == 0:
        return 0.0
    
    # Contar adyacencias urbano-urbano (vecinos de 4)
    # Desplazar grid en 4 direcciones y contar coincidencias
    g_ii = 0
    
    # Arriba
    g_ii += np.sum(binary[:-1, :] & binary[1:, :])
    # Abajo (equivalente, ya contado)
    # Izquierda
    g_ii += np.sum(binary[:, :-1] & binary[:, 1:])
    # Derecha (equivalente, ya contado)
    
    # Máximo teórico (todos los píxeles urbanos en un cuadrado perfecto)
    # Para n píxeles: max_g_ii ≈ 2n - 2√n (perímetro de cuadrado)
    # Aproximación: max_g_ii ≈ 2 * total_urban
    max_g_ii = 2 * total_urban
    
    ai = g_ii / max_g_ii if max_g_ii > 0 else 0.0
    
    return ai


def calculate_perimeter_area_ratio(grid: np.ndarray) -> float:
    """Calcula el ratio perímetro/área como indicador de complejidad de forma.

    Argumentos:
        grid: Grid binario (1=urbano, 0=no urbano) o numérico (se binariza con
            ``grid > 0``).

    Retorna:
        Ratio perímetro/área (0 si no hay área urbana).
    """
    binary = (grid > 0).astype(int)
    area = np.sum(binary)
    
    if area == 0:
        return 0.0
    
    # Calcular perímetro usando gradiente
    # Perímetro = píxeles urbanos que tienen al menos un vecino no urbano
    from scipy.ndimage import generic_filter
    
    def has_nonurban_neighbor(values):
        """Retorna 1 si el píxel central es urbano Y tiene vecino no urbano."""
        center = values[len(values)//2]
        if center == 0:
            return 0
        # Verificar si algún vecino es 0
        return 1 if np.any(values == 0) else 0
    
    perimeter_mask = generic_filter(binary, has_nonurban_neighbor, size=3)
    perimeter = np.sum(perimeter_mask)
    
    pa_ratio = perimeter / area if area > 0 else 0.0
    
    return pa_ratio


def calculate_fractal_dimension(grid: np.ndarray) -> float:
    """Calcula dimensión fractal aproximada usando box-counting.

    Argumentos:
        grid: Grid binario (1=urbano, 0=no urbano) o numérico (se binariza con
            ``grid > 0``).

    Retorna:
        Dimensión fractal estimada. Si no es estimable, retorna 0.0.
    """
    binary = (grid > 0).astype(int)
    
    # Encontrar píxeles urbanos
    points = np.argwhere(binary > 0)
    
    if len(points) < 10:
        return 0.0
    
    # Box-counting: contar cajas necesarias a diferentes escalas
    scales = []
    counts = []
    
    # Probar diferentes tamaños de caja
    for box_size in [2, 4, 8, 16, 32, 64]:
        if box_size > min(grid.shape) // 4:
            break
        
        # Cuantizar puntos a grilla de cajas
        boxes = set()
        for point in points:
            box = tuple(point // box_size)
            boxes.add(box)
        
        scales.append(np.log(1.0 / box_size))
        counts.append(np.log(len(boxes)))
    
    if len(scales) < 2:
        return 0.0
    
    # Dimensión fractal = pendiente de log-log plot
    coeffs = np.polyfit(scales, counts, 1)
    fractal_dim = coeffs[0]
    
    return float(fractal_dim)


def compare_morphology(grid_real: np.ndarray, 
                       grid_pred: np.ndarray) -> Dict[str, any]:
    """Compara morfología entre grids real y predicho.

    Argumentos:
        grid_real: Grid real.
        grid_pred: Grid predicho.

    Retorna:
        Diccionario con métricas de cada grid y diferencias porcentuales.
    """
    logger.info("\n" + "="*70)
    logger.info("COMPARACIÓN MORFOLÓGICA: Real vs Predicho")
    logger.info("="*70)
    
    # Métricas para cada grid
    logger.info("\nCalculando morfología REAL...")
    metrics_real = calculate_morphological_metrics(grid_real)
    
    logger.info("\nCalculando morfología PREDICHA...")
    metrics_pred = calculate_morphological_metrics(grid_pred)
    
    # Calcular diferencias relativas
    differences = {}
    for key in metrics_real.keys():
        real_val = metrics_real[key]
        pred_val = metrics_pred[key]
        
        if real_val != 0:
            diff_pct = ((pred_val - real_val) / real_val) * 100
        else:
            diff_pct = 0.0 if pred_val == 0 else 100.0
        
        differences[f'{key}_diff_pct'] = diff_pct
    
    comparison = {
        'real': metrics_real,
        'predicted': metrics_pred,
        'differences': differences
    }
    
    # Reporte
    logger.info("\n" + "="*70)
    logger.info("RESULTADOS COMPARATIVOS")
    logger.info("="*70)
    
    logger.info(f"\n{'Métrica':<30} {'Real':>15} {'Predicho':>15} {'Dif %':>10}")
    logger.info("-"*70)
    
    for key in metrics_real.keys():
        real_val = metrics_real[key]
        pred_val = metrics_pred[key]
        diff_pct = differences[f'{key}_diff_pct']
        
        logger.info(f"{key:<30} {real_val:>15.2f} {pred_val:>15.2f} {diff_pct:>9.1f}%")
    
    logger.info("="*70 + "\n")
    
    return comparison


def print_morphology_report(metrics: Dict[str, float], title: str = ""):
    """Imprime un reporte formateado de métricas morfológicas.

    Argumentos:
        metrics: Diccionario con métricas (salida de
            :func:`calculate_morphological_metrics`).
        title: Título opcional del reporte.
    """
    print("\n" + "="*70)
    if title:
        print(f"MÉTRICAS MORFOLÓGICAS: {title}")
    else:
        print("MÉTRICAS MORFOLÓGICAS")
    print("="*70)
    
    print("\n📦 ESTRUCTURA DE PARCHES")
    print(f"  Número de parches:        {metrics['num_patches']:>12,}")
    print(f"  Tamaño promedio:          {metrics['mean_patch_size']:>12,.1f} píxeles")
    print(f"  Tamaño mediano:           {metrics['median_patch_size']:>12,.1f} píxeles")
    print(f"  Desv. estándar:           {metrics['std_patch_size']:>12,.1f} píxeles")
    print(f"  Parche más grande:        {metrics['largest_patch_size']:>12,} píxeles")
    print(f"  Parche más pequeño:       {metrics['smallest_patch_size']:>12,} píxeles")
    
    print("\n🔗 AGREGACIÓN Y COMPACIDAD")
    print(f"  Índice de agregación:     {metrics['aggregation_index']:>12.4f}")
    ai_status = "🟢 Alta" if metrics['aggregation_index'] > 0.7 else "🟡 Media" if metrics['aggregation_index'] > 0.4 else "🔴 Baja"
    print(f"    Interpretación:         {ai_status:>12}")
    
    print(f"  Ratio perímetro/área:     {metrics['perimeter_area_ratio']:>12.4f}")
    pa_status = "🟢 Compacto" if metrics['perimeter_area_ratio'] < 0.3 else "🟡 Medio" if metrics['perimeter_area_ratio'] < 0.5 else "🔴 Irregular"
    print(f"    Interpretación:         {pa_status:>12}")
    
    print("\n🧩 FRAGMENTACIÓN")
    print(f"  Índice de fragmentación:  {metrics['fragmentation_index']:>12.4f} parches/1000px")
    frag_status = "🔴 Alta" if metrics['fragmentation_index'] > 0.5 else "🟡 Media" if metrics['fragmentation_index'] > 0.1 else "🟢 Baja"
    print(f"    Interpretación:         {frag_status:>12}")
    
    print("\n🏙️ DENSIDAD")
    print(f"  Densidad urbana global:   {metrics['urban_density']:>11.2%}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # Test con datos sintéticos
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("Test de métricas morfológicas\n")
    
    # Crear grid sintético: crecimiento compacto
    size = 200
    grid_compact = np.zeros((size, size))
    grid_compact[80:120, 80:120] = 1  # Parche grande y compacto
    
    # Grid disperso
    grid_dispersed = np.zeros((size, size))
    np.random.seed(42)
    for _ in range(50):
        x, y = np.random.randint(0, size, 2)
        grid_dispersed[x:x+5, y:y+5] = 1  # Muchos parches pequeños
    
    # Calcular métricas
    print("="*70)
    print("GRID COMPACTO")
    print("="*70)
    metrics_compact = calculate_morphological_metrics(grid_compact)
    print_morphology_report(metrics_compact, "Grid Compacto")
    
    print("\n" + "="*70)
    print("GRID DISPERSO")
    print("="*70)
    metrics_dispersed = calculate_morphological_metrics(grid_dispersed)
    print_morphology_report(metrics_dispersed, "Grid Disperso")
    
    # Comparación
    comparison = compare_morphology(grid_compact, grid_dispersed)
