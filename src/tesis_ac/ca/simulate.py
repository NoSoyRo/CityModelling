"""
Simulación temporal completa del Autómata Celular basado en Weight of Evidence.

Este módulo implementa la simulación espaciotemporal del AC-WoE, integrando
las reglas de transición con capacidades de evaluación y análisis de resultados.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import json
import pickle
from datetime import datetime

from .rules import WoECellularAutomaton, ACParameters, CellState, create_default_spatial_variables
from ..woe.woe import WoECalculator
from ..eval.metrics import calculate_spatial_metrics, evaluate_model_performance

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuración para simulación del AC."""
    n_steps: int = 10
    save_intermediate: bool = True
    validate_each_step: bool = False
    target_grid_path: Optional[str] = None
    output_dir: str = "simulation_results"
    random_seed: int = 42


@dataclass
class SimulationStep:
    """Información de un paso de simulación."""
    step: int
    timestamp: datetime
    grid: np.ndarray
    statistics: Dict[str, float]
    validation_metrics: Optional[Dict[str, float]] = None


class ACSimulator:
    """
    Simulador completo del Autómata Celular con capacidades de análisis.
    
    Integra simulación temporal, validación continua, persistencia de resultados
    y análisis de patrones espaciales emergentes.
    """
    
    def __init__(
        self,
        grid_shape: Tuple[int, int],
        woe_calculator: WoECalculator,
        config: SimulationConfig = None
    ):
        """
        Inicializar simulador.
        
        Parameters:
        -----------
        grid_shape : Tuple[int, int]
            Dimensiones del grid de simulación
        woe_calculator : WoECalculator
            Calculadora WoE con variables precomputadas
        config : SimulationConfig, optional
            Configuración de simulación
        """
        self.grid_shape = grid_shape
        self.woe_calculator = woe_calculator
        self.config = config or SimulationConfig()
        
        # Estado de simulación
        self.ac_model: Optional[WoECellularAutomaton] = None
        self.simulation_history: List[SimulationStep] = []
        self.current_step = 0
        
        # Grids de referencia para validación
        self.target_grid: Optional[np.ndarray] = None
        self.initial_grid: Optional[np.ndarray] = None
        
        # Configurar semilla aleatoria
        np.random.seed(self.config.random_seed)
        
        logger.info(f"Simulador AC inicializado: grid {grid_shape}")
        
    def setup_simulation(
        self,
        initial_urban_grid: np.ndarray,
        spatial_variables: Dict[str, np.ndarray],
        parameters: ACParameters,
        target_grid: Optional[np.ndarray] = None
    ) -> None:
        """
        Configurar simulación con datos iniciales.
        
        Parameters:
        -----------
        initial_urban_grid : np.ndarray
            Estado urbano inicial
        spatial_variables : Dict[str, np.ndarray]
            Variables espaciales para el modelo
        parameters : ACParameters
            Parámetros del AC
        target_grid : np.ndarray, optional
            Grid objetivo para validación
        """
        # Crear modelo AC
        self.ac_model = WoECellularAutomaton(
            self.grid_shape, 
            self.woe_calculator, 
            parameters
        )
        
        # Configurar estado inicial
        self.ac_model.set_initial_state(initial_urban_grid)
        self.initial_grid = initial_urban_grid.copy()
        
        # Agregar variables espaciales
        for name, variable in spatial_variables.items():
            self.ac_model.add_spatial_variable(name, variable, compute_woe=True)
            
        # Configurar grid objetivo si se proporciona
        if target_grid is not None:
            self.target_grid = target_grid.copy()
            
        # Limpiar historial previo
        self.simulation_history = []
        self.current_step = 0
        
        logger.info("Simulación configurada exitosamente")
        
    def run_simulation(self) -> List[SimulationStep]:
        """
        Ejecutar simulación completa.
        
        Returns:
        --------
        List[SimulationStep]
            Historial completo de pasos de simulación
        """
        if self.ac_model is None:
            raise ValueError("Simulación no configurada. Ejecutar setup_simulation() primero.")
            
        logger.info(f"Iniciando simulación: {self.config.n_steps} pasos")
        
        # Paso inicial (estado base)
        initial_stats = self._calculate_statistics(self.ac_model.current_grid)
        initial_validation = self._validate_step(self.ac_model.current_grid) if self.config.validate_each_step else None
        
        initial_step = SimulationStep(
            step=0,
            timestamp=datetime.now(),
            grid=self.ac_model.current_grid.copy(),
            statistics=initial_stats,
            validation_metrics=initial_validation
        )
        
        self.simulation_history = [initial_step]
        
        # Ejecutar pasos de simulación
        for step in range(1, self.config.n_steps + 1):
            logger.info(f"Ejecutando paso {step}/{self.config.n_steps}")
            
            # Ejecutar paso del AC
            new_grid, step_stats = self.ac_model.step()
            
            # Calcular estadísticas adicionales
            extended_stats = self._calculate_statistics(new_grid)
            extended_stats.update(step_stats)
            
            # Validación opcional
            validation_metrics = None
            if self.config.validate_each_step and self.target_grid is not None:
                validation_metrics = self._validate_step(new_grid)
                
            # Crear registro del paso
            step_record = SimulationStep(
                step=step,
                timestamp=datetime.now(),
                grid=new_grid.copy(),
                statistics=extended_stats,
                validation_metrics=validation_metrics
            )
            
            self.simulation_history.append(step_record)
            
            # Criterio de convergencia temprana
            if step_stats.get('new_urban_cells', 1) == 0:
                logger.info(f"Convergencia alcanzada en paso {step}")
                break
                
            # Guardar estados intermedios si se solicita
            if self.config.save_intermediate:
                self._save_intermediate_state(step_record)
                
        logger.info(f"Simulación completada: {len(self.simulation_history)} pasos")
        
        return self.simulation_history
        
    def run_batch_simulation(
        self,
        parameter_sets: List[ACParameters],
        n_replications: int = 3
    ) -> Dict[int, List[List[SimulationStep]]]:
        """
        Ejecutar simulaciones por lotes con múltiples configuraciones.
        
        Parameters:
        -----------
        parameter_sets : List[ACParameters]
            Lista de configuraciones de parámetros
        n_replications : int, default=3
            Número de réplicas por configuración
            
        Returns:
        --------
        Dict[int, List[List[SimulationStep]]]
            Resultados indexados por configuración y réplica
        """
        logger.info(f"Simulación por lotes: {len(parameter_sets)} configuraciones, {n_replications} réplicas")
        
        batch_results = {}
        
        for config_id, parameters in enumerate(parameter_sets):
            logger.info(f"Configuración {config_id + 1}/{len(parameter_sets)}")
            
            config_results = []
            
            for replica in range(n_replications):
                logger.info(f"  Réplica {replica + 1}/{n_replications}")
                
                # Configurar nueva semilla para cada réplica
                np.random.seed(self.config.random_seed + config_id * 1000 + replica)
                
                # Reinicializar AC con nuevos parámetros
                if self.ac_model is not None:
                    # Conservar configuración espacial
                    spatial_vars = self.ac_model.spatial_variables.copy()
                    initial_state = self.initial_grid.copy()
                    
                    # Reconfigurar con nuevos parámetros
                    self.setup_simulation(
                        initial_state,
                        spatial_vars,
                        parameters,
                        self.target_grid
                    )
                    
                    # Ejecutar simulación
                    replica_results = self.run_simulation()
                    config_results.append(replica_results)
                    
            batch_results[config_id] = config_results
            
        logger.info("Simulación por lotes completada")
        
        return batch_results
        
    def _calculate_statistics(self, grid: np.ndarray) -> Dict[str, float]:
        """Calcular estadísticas espaciales del grid."""
        total_cells = grid.size
        urban_cells = np.sum(grid == CellState.URBAN)
        
        stats = {
            'total_cells': float(total_cells),
            'urban_cells': float(urban_cells),
            'urban_percentage': float(urban_cells / total_cells),
            'non_urban_cells': float(np.sum(grid == CellState.NO_URBAN)),
            'restricted_cells': float(np.sum(grid == CellState.RESTRICTED))
        }
        
        # Estadísticas de conectividad (simplificadas)
        if urban_cells > 0:
            # Calcular fragmentación básica
            from scipy.ndimage import label
            labeled_grid, n_components = label(grid == CellState.URBAN)
            stats['urban_patches'] = float(n_components)
            stats['avg_patch_size'] = float(urban_cells / n_components) if n_components > 0 else 0.0
            
        return stats
        
    def _validate_step(self, grid: np.ndarray) -> Dict[str, float]:
        """Validar paso contra grid objetivo."""
        if self.target_grid is None:
            return {}
            
        try:
            metrics = calculate_spatial_metrics(grid, self.target_grid)
            return metrics
        except Exception as e:
            logger.warning(f"Error en validación: {e}")
            return {}
            
    def _save_intermediate_state(self, step: SimulationStep) -> None:
        """Guardar estado intermedio en disco."""
        try:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Guardar grid
            grid_path = output_dir / f"grid_step_{step.step:03d}.npy"
            np.save(grid_path, step.grid)
            
            # Guardar estadísticas
            stats_path = output_dir / f"stats_step_{step.step:03d}.json"
            with open(stats_path, 'w') as f:
                json.dump(step.statistics, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Error guardando estado intermedio: {e}")
            
    def get_simulation_summary(self) -> Dict[str, any]:
        """Generar resumen de la simulación."""
        if not self.simulation_history:
            return {"error": "No hay datos de simulación"}
            
        final_step = self.simulation_history[-1]
        initial_step = self.simulation_history[0]
        
        summary = {
            'simulation_config': asdict(self.config),
            'total_steps': len(self.simulation_history) - 1,  # Excluir paso inicial
            'final_urban_cells': final_step.statistics['urban_cells'],
            'initial_urban_cells': initial_step.statistics['urban_cells'],
            'total_growth': final_step.statistics['urban_cells'] - initial_step.statistics['urban_cells'],
            'final_urban_percentage': final_step.statistics['urban_percentage'],
            'growth_rate': (final_step.statistics['urban_percentage'] - 
                          initial_step.statistics['urban_percentage']),
            'simulation_duration': (final_step.timestamp - initial_step.timestamp).total_seconds(),
        }
        
        # Agregar métricas de validación si están disponibles
        if final_step.validation_metrics:
            summary['final_validation'] = final_step.validation_metrics
            
        return summary
        
    def save_simulation(self, filepath: Union[str, Path]) -> None:
        """Guardar simulación completa en archivo."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        simulation_data = {
            'config': asdict(self.config),
            'grid_shape': self.grid_shape,
            'simulation_history': [
                {
                    'step': s.step,
                    'timestamp': s.timestamp.isoformat(),
                    'grid': s.grid.tolist(),
                    'statistics': s.statistics,
                    'validation_metrics': s.validation_metrics
                }
                for s in self.simulation_history
            ],
            'summary': self.get_simulation_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(simulation_data, f, indent=2)
            
        logger.info(f"Simulación guardada en {filepath}")
        
    def load_simulation(self, filepath: Union[str, Path]) -> None:
        """Cargar simulación desde archivo.""" 
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Restaurar configuración
        self.config = SimulationConfig(**data['config'])
        self.grid_shape = tuple(data['grid_shape'])
        
        # Restaurar historial
        self.simulation_history = []
        for step_data in data['simulation_history']:
            step = SimulationStep(
                step=step_data['step'],
                timestamp=datetime.fromisoformat(step_data['timestamp']),
                grid=np.array(step_data['grid']),
                statistics=step_data['statistics'],
                validation_metrics=step_data['validation_metrics']
            )
            self.simulation_history.append(step)
            
        logger.info(f"Simulación cargada desde {filepath}")
        
    def analyze_temporal_patterns(self) -> Dict[str, any]:
        """Analizar patrones temporales en la simulación."""
        if len(self.simulation_history) < 2:
            return {"error": "Datos insuficientes para análisis temporal"}
            
        analysis = {
            'growth_trajectory': [],
            'growth_rates': [],
            'spatial_metrics_evolution': {},
        }
        
        # Extraer trayectorias temporales
        for step in self.simulation_history:
            analysis['growth_trajectory'].append({
                'step': step.step,
                'urban_percentage': step.statistics['urban_percentage'],
                'urban_cells': step.statistics['urban_cells']
            })
            
        # Calcular tasas de crecimiento
        for i in range(1, len(self.simulation_history)):
            prev_urban = self.simulation_history[i-1].statistics['urban_cells']
            curr_urban = self.simulation_history[i].statistics['urban_cells']
            growth_rate = (curr_urban - prev_urban) / prev_urban if prev_urban > 0 else 0
            analysis['growth_rates'].append({
                'step': i,
                'growth_rate': growth_rate,
                'absolute_growth': curr_urban - prev_urban
            })
            
        # Evolución de métricas espaciales
        if self.simulation_history[0].validation_metrics:
            metric_names = self.simulation_history[0].validation_metrics.keys()
            for metric in metric_names:
                analysis['spatial_metrics_evolution'][metric] = [
                    {
                        'step': step.step,
                        'value': step.validation_metrics.get(metric, np.nan)
                    }
                    for step in self.simulation_history
                    if step.validation_metrics
                ]
                
        return analysis
        
    def export_grids_for_visualization(self, output_dir: Union[str, Path]) -> List[Path]:
        """Exportar grids para visualización externa."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = []
        
        for step in self.simulation_history:
            filename = f"simulation_step_{step.step:03d}.png"
            filepath = output_dir / filename
            
            # Crear visualización simple
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            
            # Colorear por estado de celda
            colored_grid = np.zeros((*step.grid.shape, 3))
            colored_grid[step.grid == CellState.NO_URBAN] = [0.9, 0.9, 0.9]  # Gris claro
            colored_grid[step.grid == CellState.URBAN] = [0.8, 0.2, 0.2]      # Rojo
            colored_grid[step.grid == CellState.RESTRICTED] = [0.2, 0.2, 0.8] # Azul
            
            ax.imshow(colored_grid, origin='lower')
            ax.set_title(f'Simulación AC - Paso {step.step}\n'
                        f'Urbano: {step.statistics["urban_percentage"]:.1%}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            exported_files.append(filepath)
            
        logger.info(f"Exportados {len(exported_files)} archivos de visualización")
        
        return exported_files