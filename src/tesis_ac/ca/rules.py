"""
Reglas de transición para Autómata Celular basadas en Weight of Evidence (WoE).

Este módulo implementa el núcleo del modelo de crecimiento urbano:
las reglas que determinan las probabilidades de transición de cada celda
usando los pesos de evidencia calculados para variables espaciales.

Fundamentos del modelo:
- Cada celda tiene una probabilidad de urbanizarse basada en evidencia espacial
- WoE cuantifica la contribución de cada factor (carreteras, vecinos, topografía)
- Las transiciones son estocásticas pero dirigidas por evidencia empírica
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from enum import IntEnum
import logging
from dataclasses import dataclass

from ..woe.woe import WoECalculator, WoEResult

logger = logging.getLogger(__name__)


class CellState(IntEnum):
    """Estados discretos de una celda del Autómata Celular (AC).

    En este proyecto una celda puede estar:

    - no‑urbanizada
    - urbanizada
    - restringida (p. ej. agua, áreas protegidas)

    Ejemplo:
        >>> int(CellState.URBAN)
        1
    """
    NO_URBAN = 0
    URBAN = 1
    RESTRICTED = 2  # Zonas protegidas, agua, etc.


@dataclass
class ACParameters:
    """Hiperparámetros del Autómata Celular (AC) para crecimiento urbano.

    Estos parámetros controlan cómo se combina la evidencia WoE y la influencia
    del vecindario (Moore) para obtener probabilidades de **urbanización**.

    Notas:
        En el repositorio se menciona una configuración “best validated” típica
        (por ejemplo ``base_threshold=0.75`` y ponderación WoE por IV). Esos
        valores suelen guardarse en JSON/YAML y luego inyectarse al modelo.

    Ejemplo:
        >>> params = ACParameters(base_threshold=0.75, w_neighbors=0.5)
    """
    # Pesos de influencia para cada factor WoE
    w_neighbors: float = 1.0       # Peso influencia vecinos urbanos
    w_lbp: float = 0.6             # Peso densidad de textura LBP
    w_distance_urban: float = -0.4 # Peso distancia a zonas urbanas existentes
    
    # Parámetros de vecindario
    neighborhood_radius: int = 1   # Radio de vecindario (Moore)
    
    # Factor de decaimiento exponencial por distancia
    decay_rate: float = 0.1        # Tasa de decaimiento (0.05=suave, 0.2=agresivo)
    
    # Umbrales y factores estocásticos
    base_threshold: float = 0.5    # Umbral base de urbanización
    stochastic_factor: float = 0.1 # Factor de aleatoriedad
    
    # Restricciones temporales
    growth_rate_limit: float = 0.05 # Máximo % crecimiento por iteración


class WoECellularAutomaton:
    """
    Autómata Celular para crecimiento urbano basado en Weight of Evidence.
    
    El modelo combina:
    1. Evidencia empírica (WoE) de factores espaciales
    2. Influencia de vecindario (células adyacentes)
    3. Componente estocástico controlado
    4. Restricciones realistas (zonas protegidas, tasas de crecimiento)
    """
    
    def __init__(
        self,
        grid_shape: Tuple[int, int],
        woe_calculator: WoECalculator,
        parameters: ACParameters = None
    ):
        """
        Inicializar Autómata Celular.
        
        Parameters:
        -----------
        grid_shape : Tuple[int, int]
            Dimensiones del grid (height, width)
        woe_calculator : WoECalculator
            Calculadora con WoE precomputados para variables espaciales
        parameters : ACParameters, optional
            Parámetros del modelo. Si None, usa valores por defecto.
        """
        self.grid_shape = grid_shape
        self.woe_calculator = woe_calculator
        self.parameters = parameters or ACParameters()
        
        # Inicializar grid
        self.current_grid = np.zeros(grid_shape, dtype=int)
        self.probability_grid = np.zeros(grid_shape, dtype=float)
        
        # Variables espaciales precalculadas
        self.spatial_variables: Dict[str, np.ndarray] = {}
        self.woe_grids: Dict[str, np.ndarray] = {}
        
        logger.info(f"AC inicializado: grid {grid_shape}, parámetros: {parameters}")
        
    def set_initial_state(self, initial_urban_grid: np.ndarray) -> None:
        """
        Establecer estado inicial del AC.
        
        Parameters:
        -----------
        initial_urban_grid : np.ndarray
            Grid binario con estado urbano inicial (0=no urbano, 1=urbano)
        """
        if initial_urban_grid.shape != self.grid_shape:
            raise ValueError(f"Shape inconsistente: {initial_urban_grid.shape} vs {self.grid_shape}")
            
        self.current_grid = initial_urban_grid.astype(int)
        logger.info(f"Estado inicial: {np.sum(self.current_grid == 1)} celdas urbanas")
        
    def add_spatial_variable(
        self,
        name: str,
        variable_grid: np.ndarray,
        compute_woe: bool = True
    ) -> None:
        """
        Agregar variable espacial al modelo.
        
        Parameters:
        -----------
        name : str
            Nombre de la variable (debe coincidir con WoE precomputado)
        variable_grid : np.ndarray
            Grid con valores de la variable espacial
        compute_woe : bool, default=True
            Si True, calcula grid WoE transformado
        """
        if variable_grid.shape != self.grid_shape:
            raise ValueError(f"Shape inconsistente: {variable_grid.shape} vs {self.grid_shape}")
            
        self.spatial_variables[name] = variable_grid
        
        if compute_woe:
            # Transformar variable usando WoE precomputado
            try:
                var_flat = variable_grid.flatten()
                woe_flat = self.woe_calculator.apply_woe_transform(var_flat, name)
                woe_grid = woe_flat.reshape(self.grid_shape)
                self.woe_grids[name] = woe_grid
                
                logger.info(f"Variable WoE '{name}' agregada: rango [{woe_grid.min():.3f}, {woe_grid.max():.3f}]")
                
            except Exception as e:
                logger.warning(f"No se pudo calcular WoE para '{name}': {e}")
                
    def _calculate_neighborhood_influence(self, row: int, col: int) -> float:
        """Calcula la influencia del vecindario urbano para una celda.

        Usa un vecindario de Moore con un peso decreciente conforme aumenta la
        distancia a la celda central.

        Argumentos:
            row: Coordenada de fila de la celda.
            col: Coordenada de columna de la celda.

        Retorna:
            Influencia del vecindario en el rango $[0, 1]$.
        """
        radius = self.parameters.neighborhood_radius
        
        # Límites del vecindario
        r_min = max(0, row - radius)
        r_max = min(self.grid_shape[0], row + radius + 1)
        c_min = max(0, col - radius)
        c_max = min(self.grid_shape[1], col + radius + 1)
        
        # Extraer vecindario
        neighborhood = self.current_grid[r_min:r_max, c_min:c_max]
        
        # Calcular pesos por distancia (Núcleo de Moore ponderado)
        h, w = neighborhood.shape
        center_r, center_c = (h - 1) // 2, (w - 1) // 2
        
        total_influence = 0.0
        total_weight = 0.0
        
        for i in range(h):
            for j in range(w):
                if i == center_r and j == center_c:
                    continue  # Excluir celda central
                    
                # Peso inversamente proporcional a distancia
                distance = max(abs(i - center_r), abs(j - center_c))
                weight = 1.0 / (distance + 1)
                
                if neighborhood[i, j] == CellState.URBAN:
                    total_influence += weight
                    
                total_weight += weight
                
        # Normalizar influencia
        if total_weight > 0:
            influence = total_influence / total_weight
        else:
            influence = 0.0
            
        return influence
        
    def _calculate_urbanization_probability(self, row: int, col: int) -> float:
        """Calcula la probabilidad de urbanización para una celda.

        Combina la evidencia WoE de múltiples variables espaciales con la
        influencia del vecindario para estimar una probabilidad final.

        Nota:
            Se aplica un factor de decaimiento exponencial basado en distancia
            para penalizar zonas alejadas de áreas urbanas existentes.

        Argumentos:
            row: Coordenada de fila de la celda.
            col: Coordenada de columna de la celda.

        Retorna:
            Probabilidad de urbanización en el rango $[0, 1]$.
        """
        # Si ya es urbana o restringida, probabilidad = 0
        if self.current_grid[row, col] != CellState.NO_URBAN:
            return 0.0
            
        # Componente de vecindario
        neighborhood_influence = self._calculate_neighborhood_influence(row, col)
        neighborhood_score = self.parameters.w_neighbors * neighborhood_influence
        
        # Componentes WoE de variables espaciales
        woe_total = 0.0
        
        # Distancia a centros urbanos con WoE
        if 'distance_urban' in self.woe_grids:
            woe_urban = self.woe_grids['distance_urban'][row, col]
            woe_total += self.parameters.w_distance_urban * woe_urban
        
        # Densidad de vecinos 3x3
        if 'neighbor_density_3x3' in self.woe_grids:
            woe_val = self.woe_grids['neighbor_density_3x3'][row, col]
            woe_total += 0.3 * woe_val  # Peso por defecto
        
        # Densidad de vecinos 5x5
        if 'neighbor_density_5x5' in self.woe_grids:
            woe_val = self.woe_grids['neighbor_density_5x5'][row, col]
            woe_total += 0.2 * woe_val
        
        # Densidad de vecinos 7x7
        if 'neighbor_density_7x7' in self.woe_grids:
            woe_val = self.woe_grids['neighbor_density_7x7'][row, col]
            woe_total += 0.1 * woe_val
        
        # Fragmentación local
        if 'local_fragmentation' in self.woe_grids:
            woe_val = self.woe_grids['local_fragmentation'][row, col]
            woe_total += 0.15 * woe_val
        
        # Tamaño de cluster más cercano
        if 'nearest_cluster_size' in self.woe_grids:
            woe_val = self.woe_grids['nearest_cluster_size'][row, col]
            woe_total += 0.2 * woe_val
        
        # Gradiente de urbanización
        if 'urban_gradient' in self.woe_grids:
            woe_val = self.woe_grids['urban_gradient'][row, col]
            woe_total += 0.15 * woe_val
        
        # NUEVO: Factor de decaimiento exponencial por distancia
        # Zonas muy alejadas tienen penalización exponencial
        if 'distance_urban' in self.spatial_variables:
            distance = self.spatial_variables['distance_urban'][row, col]
            
            # Factor exponencial: e^(-decay_rate * distance)
            # distance=0 -> factor=1.0 (sin penalización)
            # distance=10 -> factor=0.37 (con decay_rate=0.1)
            # distance=20 -> factor=0.14
            # distance=30 -> factor=0.05
            distance_decay = np.exp(-self.parameters.decay_rate * distance)
            
            # Aplicar penalización exponencial al score total
            woe_total *= distance_decay
            neighborhood_score *= distance_decay
            
        # Score total combinado
        combined_score = neighborhood_score + woe_total
        
        # Convertir a probabilidad usando función logística
        probability = 1.0 / (1.0 + np.exp(-combined_score))
        
        # Aplicar umbral base
        if probability < self.parameters.base_threshold:
            probability *= 0.1  # Reducir drásticamente probabilidades bajas
            
        # Componente estocástico
        random_factor = np.random.normal(0, self.parameters.stochastic_factor)
        probability = np.clip(probability + random_factor, 0.0, 1.0)
        
        return probability
        
    def _update_probability_grid(self) -> None:
        """Actualizar grid completo de probabilidades de urbanización."""
        logger.debug("Actualizando grid de probabilidades...")
        
        for row in range(self.grid_shape[0]):
            for col in range(self.grid_shape[1]):
                self.probability_grid[row, col] = self._calculate_urbanization_probability(row, col)
                
    def _apply_growth_constraints(self, candidate_transitions: np.ndarray) -> np.ndarray:
        """Aplica restricciones realistas de crecimiento urbano.

        Actualmente limita la tasa de crecimiento por iteración, seleccionando
        (si es necesario) las celdas candidatas con mayor probabilidad.

        Argumentos:
            candidate_transitions: Grid binario con transiciones candidatas.

        Retorna:
            Grid binario con transiciones después de aplicar restricciones.
        """
        current_urban_count = np.sum(self.current_grid == CellState.URBAN)
        candidate_count = np.sum(candidate_transitions)
        
        # Limitar tasa de crecimiento por iteración
        max_growth = int(current_urban_count * self.parameters.growth_rate_limit)
        
        if candidate_count > max_growth:
            # Seleccionar las celdas con mayor probabilidad
            probabilities_flat = self.probability_grid.flatten()
            transitions_flat = candidate_transitions.flatten()
            
            # Obtener índices de candidatos ordenados por probabilidad
            candidate_indices = np.where(transitions_flat)[0]
            candidate_probs = probabilities_flat[candidate_indices]
            
            # Seleccionar top N candidatos
            top_indices = candidate_indices[np.argsort(candidate_probs)[-max_growth:]]
            
            # Crear nuevo grid de transiciones
            constrained_transitions = np.zeros_like(candidate_transitions)
            constrained_flat = constrained_transitions.flatten()
            constrained_flat[top_indices] = 1
            constrained_transitions = constrained_flat.reshape(self.grid_shape)
            
            logger.info(f"Crecimiento limitado: {candidate_count} → {max_growth} transiciones")
            
            return constrained_transitions
            
        return candidate_transitions
        
    def step(self) -> Tuple[np.ndarray, Dict[str, float]]:
        """Ejecuta un paso de simulación del AC (crecimiento urbano).

        Un *paso*:
        1) actualiza la superficie de probabilidades,
        2) muestrea transiciones estocásticas,
        3) aplica restricciones de crecimiento,
        4) actualiza el estado del grid.

        Retorna:
            Tupla ``(grid, stats)`` donde:

            - ``grid`` es el grid actualizado.
            - ``stats`` contiene métricas diagnósticas (nuevas urbanizaciones,
              probabilidad promedio, tasa de crecimiento).

        Lanza:
            RuntimeError: Si el autómata no ha sido inicializado apropiadamente.
                (Este método asume que se llamó :meth:`set_initial_state`.)

        Ejemplo:
            >>> grid, stats = ac.step()  # doctest: +SKIP
        """
        # Actualizar probabilidades
        self._update_probability_grid()
        
        # Generar transiciones estocásticas
        random_grid = np.random.random(self.grid_shape)
        candidate_transitions = (
            (self.current_grid == CellState.NO_URBAN) & 
            (random_grid < self.probability_grid)
        )
        
        # Aplicar restricciones de crecimiento
        final_transitions = self._apply_growth_constraints(candidate_transitions)
        
        # Actualizar grid
        new_grid = self.current_grid.copy()
        new_grid[final_transitions] = CellState.URBAN
        
        # Calcular estadísticas
        new_urban_cells = np.sum(final_transitions)
        total_urban_cells = np.sum(new_grid == CellState.URBAN)
        avg_probability = np.mean(self.probability_grid[self.current_grid == CellState.NO_URBAN])
        
        stats = {
            'new_urban_cells': int(new_urban_cells),
            'total_urban_cells': int(total_urban_cells),
            'avg_probability': float(avg_probability),
            'growth_rate': float(new_urban_cells / np.sum(self.current_grid == CellState.URBAN)),
            'urban_percentage': float(total_urban_cells / (self.grid_shape[0] * self.grid_shape[1]))
        }
        
        # Actualizar estado
        self.current_grid = new_grid
        
        logger.info(f"Paso AC: +{new_urban_cells} celdas urbanas, total={total_urban_cells}")
        
        return new_grid, stats
        
    def simulate(self, n_steps: int) -> Tuple[List[np.ndarray], List[Dict[str, float]]]:
        """Ejecuta una simulación completa del AC por un número máximo de pasos.

        La simulación puede detenerse antes si se alcanza convergencia (es decir,
        si en un paso no ocurre ninguna nueva urbanización).

        Argumentos:
            n_steps: Número máximo de pasos a ejecutar.

        Retorna:
            Tupla ``(grids, stats_list)`` donde:

            - ``grids`` incluye el grid inicial y un grid por cada paso ejecutado.
            - ``stats_list`` incluye un diccionario de estadísticas por paso.

        Ejemplo:
            >>> grids, stats = ac.simulate(n_steps=20)  # doctest: +SKIP
        """
        logger.info(f"Iniciando simulación AC: {n_steps} pasos")
        
        grids = [self.current_grid.copy()]
        stats_list = []
        
        for step in range(n_steps):
            grid, stats = self.step()
            grids.append(grid.copy())
            stats_list.append(stats)
            
            # Convergencia: si no hay más crecimiento
            if stats['new_urban_cells'] == 0:
                logger.info(f"Convergencia alcanzada en paso {step + 1}")
                break
                
        logger.info(f"Simulación completada: {len(grids) - 1} pasos ejecutados")
        
        return grids, stats_list
        
    def get_transition_rules_summary(self) -> str:
        """Crea un resumen legible de las reglas de transición del AC.

        El resumen incluye la parametrización actual y las variables espaciales
        disponibles ya transformadas a WoE.

        Retorna:
            Cadena multilínea con el resumen.

        Ejemplo:
            >>> print(ac.get_transition_rules_summary())  # doctest: +SKIP
        """
        summary = "\n=== REGLAS DE TRANSICIÓN AC-WoE ===\n\n"
        
        summary += "Parámetros del modelo:\n"
        summary += f"  • Peso vecinos urbanos: {self.parameters.w_neighbors:.3f}\n"
        summary += f"  • Peso textura LBP: {self.parameters.w_lbp:.3f}\n"
        summary += f"  • Peso distancia urbana: {self.parameters.w_distance_urban:.3f}\n"
        summary += f"  • Radio vecindario: {self.parameters.neighborhood_radius}\n"
        summary += f"  • Umbral base: {self.parameters.base_threshold:.3f}\n"
        summary += f"  • Factor estocástico: {self.parameters.stochastic_factor:.3f}\n"
        summary += f"  • Límite crecimiento: {self.parameters.growth_rate_limit:.1%}\n\n"
        
        summary += "Variables WoE disponibles:\n"
        for name in self.woe_grids.keys():
            woe_grid = self.woe_grids[name]
            summary += f"  • {name}: rango [{woe_grid.min():.3f}, {woe_grid.max():.3f}]\n"
            
        summary += f"\nEstado actual: {np.sum(self.current_grid == CellState.URBAN)} celdas urbanas\n"
        
        return summary


def create_default_spatial_variables(
    grid_shape: Tuple[int, int],
    existing_urban: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Calcula variables espaciales base derivadas únicamente del grid urbano.

    Este helper genera el conjunto central de variables espaciales usadas en el
    proyecto para modelar **urbanización** y **transición urbana**:

    - ``distance_urban``
    - ``neighbor_density_3x3`` / ``neighbor_density_5x5`` / ``neighbor_density_7x7``
    - ``local_fragmentation``
    - ``nearest_cluster_size``
    - ``urban_gradient``

    Posteriormente estas variables se pueden transformar a scores WoE con
    :meth:`tesis_ac.woe.woe.WoECalculator.apply_woe_transform`.

    Argumentos:
        grid_shape: Forma del grid como ``(rows, cols)``.
        existing_urban: Grid binario donde 1 indica celdas urbanizadas.

    Retorna:
        Diccionario que mapea nombre de variable a un array 2D con la misma forma
        del grid.

    Lanza:
        ImportError: Si SciPy no está instalado (depende de ``scipy.ndimage``).

    Ejemplo:
        >>> vars_ = create_default_spatial_variables(grid.shape, grid)  # doctest: +SKIP
        >>> vars_["distance_urban"].shape  # doctest: +SKIP
        (500, 500)
    """
    import time
    from scipy.ndimage import distance_transform_edt, uniform_filter, label
    from scipy.ndimage import generic_filter
    
    variables = {}
    total_start = time.time()
    
    # 1. Distancia a zonas urbanas existentes
    logger.info(f"  [1/7] Calculando distance_urban...")
    t0 = time.time()
    urban_distance = distance_transform_edt(existing_urban == 0)
    variables['distance_urban'] = urban_distance
    logger.info(f"        ✓ Completado en {time.time()-t0:.2f}s")
    
    # 2. Densidad de vecinos urbanos en radio 3x3 (Moore)
    # Mide qué tan "rodeada" está cada celda por zonas urbanas
    logger.info(f"  [2/7] Calculando neighbor_density_3x3...")
    t0 = time.time()
    density_3x3 = uniform_filter(existing_urban.astype(float), size=3, mode='constant')
    variables['neighbor_density_3x3'] = density_3x3
    logger.info(f"        ✓ Completado en {time.time()-t0:.2f}s")
    
    # 3. Densidad de vecinos urbanos en radio 5x5
    # Captura influencia de vecindario más amplio
    logger.info(f"  [3/7] Calculando neighbor_density_5x5...")
    t0 = time.time()
    density_5x5 = uniform_filter(existing_urban.astype(float), size=5, mode='constant')
    variables['neighbor_density_5x5'] = density_5x5
    logger.info(f"        ✓ Completado en {time.time()-t0:.2f}s")
    
    # 4. Densidad de vecinos urbanos en radio 7x7
    # Contexto regional
    logger.info(f"  [4/7] Calculando neighbor_density_7x7...")
    t0 = time.time()
    density_7x7 = uniform_filter(existing_urban.astype(float), size=7, mode='constant')
    variables['neighbor_density_7x7'] = density_7x7
    logger.info(f"        ✓ Completado en {time.time()-t0:.2f}s")
    
    # 5. Fragmentación local (varianza en ventana 5x5)
    # Alta varianza = borde/transición, baja varianza = homogéneo
    logger.info(f"  [5/7] Calculando local_fragmentation...")
    t0 = time.time()
    def local_variance(window):
        return np.var(window)
    
    fragmentation = generic_filter(
        existing_urban.astype(float), 
        local_variance, 
        size=5, 
        mode='constant'
    )
    variables['local_fragmentation'] = fragmentation
    logger.info(f"        ✓ Completado en {time.time()-t0:.2f}s")
    
    # 6. Tamaño del cluster urbano más cercano (OPTIMIZADO)
    # Clusters grandes tienen más "inercia" de crecimiento
    logger.info(f"  [6/7] Calculando nearest_cluster_size...")
    t0 = time.time()
    labeled_array, num_features = label(existing_urban)
    logger.info(f"        - Encontrados {num_features} clusters urbanos")
    
    if num_features > 0:
        # Calcular tamaño de cada cluster
        cluster_sizes_dict = {}
        for i in range(1, num_features + 1):
            cluster_sizes_dict[i] = np.sum(labeled_array == i)
        
        # Para cada celda, asignar tamaño de su cluster (si es urbana)
        # o del cluster más cercano (si es no-urbana)
        nearest_cluster_size = np.zeros(grid_shape, dtype=float)
        
        # Celdas urbanas: usar su propio cluster
        for i in range(1, num_features + 1):
            mask = labeled_array == i
            nearest_cluster_size[mask] = cluster_sizes_dict[i]
        
        # Celdas no-urbanas: usar propagación de distancia
        # Para cada cluster, calcular distancia y propagar su tamaño
        logger.info(f"        - Propagando distancias para {num_features} clusters...")
        min_dist = np.full(grid_shape, np.inf)
        
        for i in range(1, num_features + 1):
            if i % 50 == 0:
                logger.info(f"          Cluster {i}/{num_features}...")
            cluster_mask = labeled_array == i
            dist = distance_transform_edt(~cluster_mask)
            
            # Donde esta distancia es menor, actualizar el tamaño
            update_mask = dist < min_dist
            min_dist[update_mask] = dist[update_mask]
            nearest_cluster_size[update_mask] = cluster_sizes_dict[i]
        
        variables['nearest_cluster_size'] = nearest_cluster_size
    else:
        variables['nearest_cluster_size'] = np.zeros(grid_shape)
    
    logger.info(f"        ✓ Completado en {time.time()-t0:.2f}s")
    
    # 7. Gradiente de urbanización (cambio espacial)
    # Captura "frentes" de crecimiento
    logger.info(f"  [7/7] Calculando urban_gradient...")
    t0 = time.time()
    from scipy.ndimage import sobel
    gradient_x = sobel(existing_urban.astype(float), axis=0, mode='constant')
    gradient_y = sobel(existing_urban.astype(float), axis=1, mode='constant')
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    variables['urban_gradient'] = gradient_magnitude
    logger.info(f"        ✓ Completado en {time.time()-t0:.2f}s")
    
    total_time = time.time() - total_start
    logger.info(f"Variables espaciales derivadas: {list(variables.keys())}")
    logger.info(f"  - {len(variables)} variables generadas en {total_time:.2f}s total")
    
    return variables