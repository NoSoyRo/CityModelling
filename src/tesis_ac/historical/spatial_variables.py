"""
Cálculo de variables espaciales para análisis de transiciones.

Este módulo calcula variables explicativas que pueden predecir
dónde ocurrirán transiciones urbanas basándose en el estado actual.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from scipy.ndimage import generic_filter, distance_transform_edt, binary_erosion, convolve
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SpatialVariablesConfig:
    """Configuración para cálculo de variables espaciales."""
    neighborhood_radius: int = 3
    edge_erosion_iterations: int = 2
    calculate_slope: bool = True
    calculate_distance_roads: bool = False  # Requiere datos externos
    calculate_recent_growth: bool = False  # Requiere grid de crecimiento reciente
    
    # Normalización
    normalize_distances: bool = True
    max_distance_clip: float = 5000.0  # metros (asumiendo resolución 100m)
    recent_growth_clip: float = 2000.0  # metros para distancia a crecimiento reciente


class SpatialVariableCalculator:
    """
    Calculadora de variables espaciales explicativas.
    
    Variables calculadas del histórico:
    1. neighborhood_density: Densidad de vecinos urbanos
    2. distance_to_urban: Distancia a celda urbana más cercana
    3. urban_edge_distance: Distancia al borde urbano
    4. urban_core_distance: Distancia al núcleo urbano (área consolidada)
    
    Variables opcionales (requieren datos externos):
    5. slope: Pendiente del terreno (requiere DEM)
    6. distance_to_roads: Distancia a carreteras (requiere mapa vial)
    """
    
    def __init__(self, config: Optional[SpatialVariablesConfig] = None):
        """Inicializa la calculadora.

        Argumentos:
            config: Configuración de cálculo. Si es ``None``, usa
                :class:`SpatialVariablesConfig` con valores por defecto.
        """
        self.config = config or SpatialVariablesConfig()
    
    def calculate_all_variables(
        self,
        urban_grid: np.ndarray,
        dem: Optional[np.ndarray] = None,
        roads: Optional[np.ndarray] = None,
        recent_growth_grid: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> Dict[str, np.ndarray]:
        """Calcula todas las variables espaciales disponibles.

        Argumentos:
            urban_grid: Grid binario (0=no urbano, 1=urbano).
            dem: Modelo Digital de Elevación (opcional).
            roads: Mapa de carreteras (opcional).
            recent_growth_grid: Grid binario de crecimiento reciente (opcional).
            verbose: Si ``True``, imprime tiempos de cálculo.

        Retorna:
            Diccionario con todas las variables calculadas.
        """
        import time
        
        if verbose:
            print(f"        • Calculando variables para grid {urban_grid.shape}")
        
        logger.info(f"Calculando variables espaciales para grid {urban_grid.shape}")
        
        variables = {}
        
        # Variables del histórico (siempre disponibles)
        t0 = time.time()
        variables['neighborhood_density'] = calculate_neighborhood_density(
            urban_grid, 
            radius=self.config.neighborhood_radius
        )
        t1 = time.time()
        if verbose:
            print(f"          - neighborhood_density: {t1-t0:.2f}s")
        logger.info("  ✅ neighborhood_density")
        
        t0 = time.time()
        variables['distance_to_urban'] = calculate_distance_to_urban(
            urban_grid,
            normalize=self.config.normalize_distances,
            max_clip=self.config.max_distance_clip
        )
        t1 = time.time()
        if verbose:
            print(f"          - distance_to_urban: {t1-t0:.2f}s")
        logger.info("  ✅ distance_to_urban")
        
        t0 = time.time()
        variables['urban_edge_distance'] = calculate_urban_edge_distance(
            urban_grid,
            erosion_iterations=self.config.edge_erosion_iterations,
            normalize=self.config.normalize_distances
        )
        t1 = time.time()
        if verbose:
            print(f"          - urban_edge_distance: {t1-t0:.2f}s")
        logger.info("  ✅ urban_edge_distance")
        
        t0 = time.time()
        variables['urban_core_distance'] = calculate_urban_core_distance(
            urban_grid,
            core_erosion=5,
            normalize=self.config.normalize_distances
        )
        t1 = time.time()
        if verbose:
            print(f"          - urban_core_distance: {t1-t0:.2f}s")
        logger.info("  ✅ urban_core_distance")
        
        # Variables opcionales
        if dem is not None and self.config.calculate_slope:
            variables['slope'] = calculate_slope(dem)
            logger.info("  ✅ slope")
        
        if roads is not None and self.config.calculate_distance_roads:
            variables['distance_to_roads'] = calculate_distance_to_roads(
                roads,
                normalize=self.config.normalize_distances
            )
            logger.info("  ✅ distance_to_roads")
        
        if recent_growth_grid is not None and self.config.calculate_recent_growth:
            t0 = time.time()
            variables['recent_growth_proximity'] = calculate_recent_growth_proximity(
                recent_growth_grid,
                normalize=self.config.normalize_distances,
                max_clip=self.config.recent_growth_clip
            )
            t1 = time.time()
            if verbose:
                print(f"          - recent_growth_proximity: {t1-t0:.2f}s")
            logger.info("  ✅ recent_growth_proximity")
        
        logger.info(f"✅ {len(variables)} variables calculadas")
        
        return variables


def calculate_neighborhood_density(
    urban_grid: np.ndarray,
    radius: int = 3
) -> np.ndarray:
    """Calcula densidad de vecinos urbanos (vecindario de Moore).

    Argumentos:
        urban_grid: Grid binario urbano.
        radius: Radio del vecindario.

    Retorna:
        Densidad de urbanización en cada celda (rango $[0, 1]$).
    """
    # OPTIMIZACIÓN: Usar convolución en lugar de generic_filter (10x más rápido)
    from scipy.ndimage import convolve
    
    size = 2 * radius + 1
    
    # Crear kernel de unos
    kernel = np.ones((size, size), dtype=float)
    # Excluir el centro (no contar la celda misma)
    kernel[radius, radius] = 0
    
    # Contar vecinos urbanos
    urban_float = urban_grid.astype(float)
    n_urban_neighbors = convolve(urban_float, kernel, mode='constant', cval=0)
    
    # Total de vecinos (kernel sum)
    n_total = np.sum(kernel)
    
    # Densidad = vecinos urbanos / total de vecinos
    density = n_urban_neighbors / n_total
    
    return density


def calculate_distance_to_urban(
    urban_grid: np.ndarray,
    normalize: bool = True,
    max_clip: float = 5000.0
) -> np.ndarray:
    """
    Calcular distancia euclidiana a celda urbana más cercana.
    
    Parameters:
    -----------
    urban_grid : np.ndarray
        Grid binario urbano
    normalize : bool
        Si True, normaliza distancias a [0, 1]
    max_clip : float
        Distancia máxima para clipping
        
    Returns:
    --------
    np.ndarray
        Distancia a zona urbana en cada celda
    """
    urban_mask = (urban_grid == 1)
    
    # Distancia euclidiana
    distances = distance_transform_edt(~urban_mask)
    
    # Clip a máximo
    if max_clip is not None:
        distances = np.clip(distances, 0, max_clip)
    
    # Normalizar
    if normalize:
        max_dist = np.max(distances)
        if max_dist > 0:
            distances = distances / max_dist
    
    return distances


def calculate_urban_edge_distance(
    urban_grid: np.ndarray,
    erosion_iterations: int = 2,
    normalize: bool = True
) -> np.ndarray:
    """
    Calcular distancia al borde urbano (frontera de expansión).
    
    El borde urbano es la diferencia entre área urbana y su erosión.
    
    Parameters:
    -----------
    urban_grid : np.ndarray
        Grid binario urbano
    erosion_iterations : int
        Iteraciones de erosión para definir borde
    normalize : bool
        Si True, normaliza a [0, 1]
        
    Returns:
    --------
    np.ndarray
        Distancia al borde urbano
    """
    urban_mask = (urban_grid == 1)
    
    # Crear máscara del núcleo urbano (área erosionada)
    urban_core = binary_erosion(urban_mask, iterations=erosion_iterations)
    
    # Borde es la diferencia: área urbana - núcleo
    urban_edge = urban_mask & ~urban_core
    
    # Distancia al borde
    distances = distance_transform_edt(~urban_edge)
    
    # Normalizar
    if normalize:
        max_dist = np.max(distances)
        if max_dist > 0:
            distances = distances / max_dist
    
    return distances


def calculate_urban_core_distance(
    urban_grid: np.ndarray,
    core_erosion: int = 5,
    normalize: bool = True
) -> np.ndarray:
    """
    Calcular distancia al núcleo urbano consolidado.
    
    El núcleo es el área urbana después de erosiones significativas,
    representando centros urbanos consolidados.
    
    Parameters:
    -----------
    urban_grid : np.ndarray
        Grid binario urbano
    core_erosion : int
        Iteraciones de erosión para identificar núcleo
    normalize : bool
        Si True, normaliza a [0, 1]
        
    Returns:
    --------
    np.ndarray
        Distancia a núcleos urbanos consolidados
    """
    urban_mask = (urban_grid == 1)
    
    # Identificar núcleos urbanos (áreas muy erosionadas)
    urban_cores = binary_erosion(urban_mask, iterations=core_erosion)
    
    if not np.any(urban_cores):
        # No hay núcleos identificables, usar área urbana completa
        logger.warning(f"No se identificaron núcleos con erosión={core_erosion}, usando área urbana")
        urban_cores = urban_mask
    
    # Distancia a núcleos
    distances = distance_transform_edt(~urban_cores)
    
    # Normalizar
    if normalize:
        max_dist = np.max(distances)
        if max_dist > 0:
            distances = distances / max_dist
    
    return distances


def calculate_slope(dem: np.ndarray) -> np.ndarray:
    """
    Calcular pendiente del terreno desde DEM.
    
    Parameters:
    -----------
    dem : np.ndarray
        Modelo Digital de Elevación
        
    Returns:
    --------
    np.ndarray
        Pendiente en cada celda (magnitud del gradiente)
    """
    # Calcular gradiente
    gy, gx = np.gradient(dem)
    
    # Magnitud del gradiente = pendiente
    slope = np.sqrt(gx**2 + gy**2)
    
    return slope


def calculate_distance_to_roads(
    roads: np.ndarray,
    normalize: bool = True,
    max_clip: float = 2000.0
) -> np.ndarray:
    """
    Calcular distancia a carreteras.
    
    Parameters:
    -----------
    roads : np.ndarray
        Mapa binario de carreteras (1=carretera, 0=no carretera)
    normalize : bool
        Si True, normaliza a [0, 1]
    max_clip : float
        Distancia máxima para clipping
        
    Returns:
    --------
    np.ndarray
        Distancia a carretera más cercana
    """
    road_mask = (roads > 0)
    
    # Distancia a carreteras
    distances = distance_transform_edt(~road_mask)
    
    # Clip
    if max_clip is not None:
        distances = np.clip(distances, 0, max_clip)
    
    # Normalizar
    if normalize:
        max_dist = np.max(distances)
        if max_dist > 0:
            distances = distances / max_dist
    
    return distances


def calculate_recent_growth_proximity(
    recent_growth_grid: np.ndarray,
    normalize: bool = True,
    max_clip: float = 2000.0
) -> np.ndarray:
    """
    Calcular proximidad a zonas de crecimiento reciente.
    
    Esta variable captura la tendencia de que el crecimiento urbano
    tiende a ocurrir cerca de áreas que crecieron recientemente,
    ayudando a concentrar predicciones en hotspots activos.
    
    Parameters:
    -----------
    recent_growth_grid : np.ndarray
        Grid binario donde 1 indica crecimiento reciente
    normalize : bool
        Si True, normaliza distancias a [0, 1]
    max_clip : float
        Distancia máxima para clipping (metros)
        
    Returns:
    --------
    np.ndarray
        Distancia inversa a crecimiento reciente (valores altos = cerca de crecimiento)
    """
    growth_mask = (recent_growth_grid > 0)
    
    # Si no hay crecimiento reciente, retornar zeros
    if not np.any(growth_mask):
        return np.zeros_like(recent_growth_grid, dtype=float)
    
    # Distancia euclidiana a zonas de crecimiento reciente
    distances = distance_transform_edt(~growth_mask)
    
    # Clip a máximo
    if max_clip is not None:
        distances = np.clip(distances, 0, max_clip)
    
    # Invertir: queremos que cerca = alto valor
    # Usamos max_distance - distance para invertir
    max_dist = np.max(distances) if not normalize else max_clip
    if max_dist > 0:
        proximity = (max_dist - distances) / max_dist
    else:
        proximity = np.ones_like(distances)
    
    return proximity


def visualize_spatial_variables(
    variables: Dict[str, np.ndarray],
    urban_grid: np.ndarray,
    output_dir: Optional[str] = None
) -> None:
    """
    Visualizar variables espaciales calculadas.
    
    Parameters:
    -----------
    variables : Dict[str, np.ndarray]
        Variables espaciales calculadas
    urban_grid : np.ndarray
        Grid urbano original para contexto
    output_dir : str, optional
        Directorio para guardar visualizaciones
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    n_vars = len(variables)
    ncols = min(3, n_vars)
    nrows = (n_vars + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    
    if n_vars == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if nrows > 1 else axes
    
    for idx, (var_name, var_data) in enumerate(variables.items()):
        ax = axes[idx]
        
        # Overlay: grid urbano como contorno
        im = ax.imshow(var_data, cmap='viridis', alpha=0.8)
        ax.contour(urban_grid, levels=[0.5], colors='red', linewidths=1, alpha=0.5)
        
        ax.set_title(f'{var_name}\n(contorno rojo = urbano)', fontweight='bold')
        ax.axis('off')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Ocultar ejes sobrantes
    for idx in range(n_vars, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if output_dir is not None:
        output_path = Path(output_dir) / 'spatial_variables_visualization.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Visualización guardada: {output_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    """Demo rápido con datos sintéticos."""
    
    # Crear grid urbano sintético
    np.random.seed(42)
    grid_size = (200, 200)
    
    # Zona urbana central
    urban_grid = np.zeros(grid_size, dtype=int)
    urban_grid[80:120, 80:120] = 1
    
    # Expansiones aleatorias
    for _ in range(100):
        i, j = np.random.randint(0, grid_size[0], 2)
        if np.any(urban_grid[max(0,i-2):i+3, max(0,j-2):j+3] == 1):
            urban_grid[i, j] = 1
    
    print("Grid urbano sintético creado")
    print(f"  Shape: {urban_grid.shape}")
    print(f"  Píxeles urbanos: {np.sum(urban_grid == 1):,}")
    
    # Calcular variables
    calculator = SpatialVariableCalculator()
    variables = calculator.calculate_all_variables(urban_grid)
    
    print(f"\nVariables calculadas:")
    for var_name, var_data in variables.items():
        print(f"  {var_name}: min={var_data.min():.3f}, max={var_data.max():.3f}, "
              f"mean={var_data.mean():.3f}")
    
    # Visualizar
    visualize_spatial_variables(variables, urban_grid)
