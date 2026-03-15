"""
Optimizador de algoritmo genético para calibración automática de parámetros
del modelo WoE-AC.

Este módulo implementa un algoritmo genético distribuido usando DEAP para
encontrar configuraciones óptimas de parámetros que maximicen la precisión
del modelo de autómata celular basado en Weight of Evidence.
"""

import logging
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import json
from pathlib import Path

try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    logging.warning("DEAP no disponible. Instalar con: pip install deap")

from ..ca.rules import ACParameters
from ..ca.simulate import ACSimulator, SimulationConfig
from ..eval.metrics import calculate_spatial_metrics

logger = logging.getLogger(__name__)


@dataclass
class GAConfig:
    """Configuración del algoritmo genético."""
    
    # Parámetros de población
    population_size: int = 50
    n_generations: int = 100
    
    # Parámetros genéticos
    crossover_prob: float = 0.7
    mutation_prob: float = 0.2
    tournament_size: int = 3
    
    # Parámetros de convergencia
    max_stagnant_generations: int = 20
    target_fitness: float = 0.95
    
    # Paralelización
    n_processes: int = 4
    
    # Salida y logging
    output_dir: str = "optimization_results"
    save_frequency: int = 10
    log_frequency: int = 1
    
    # Semilla aleatoria
    random_seed: int = 42


@dataclass 
class ParameterBounds:
    """Límites para los parámetros a optimizar."""
    
    # Parámetros WoE
    distance_weights_min: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    distance_weights_max: List[float] = field(default_factory=lambda: [2.0, 2.0, 2.0])
    
    # Parámetros AC
    growth_probability_min: float = 0.001
    growth_probability_max: float = 0.1
    
    neighborhood_influence_min: float = 0.1
    neighborhood_influence_max: float = 1.0
    
    spontaneous_growth_min: float = 0.0001
    spontaneous_growth_max: float = 0.01
    
    # Restricciones
    protected_influence_min: float = 0.5
    protected_influence_max: float = 1.0
    
    slope_threshold_min: float = 5.0
    slope_threshold_max: float = 45.0


class WoEACOptimizer:
    """
    Optimizador de algoritmo genético para parámetros WoE-AC.
    
    Utiliza DEAP para evolucionar poblaciones de configuraciones de parámetros
    y encontrar la combinación óptima que maximice la precisión del modelo.
    """
    
    def __init__(
        self,
        ga_config: GAConfig,
        parameter_bounds: ParameterBounds,
        fitness_function: Optional[Callable] = None
    ):
        """
        Inicializar optimizador.
        
        Parameters:
        -----------
        ga_config : GAConfig
            Configuración del algoritmo genético
        parameter_bounds : ParameterBounds
            Límites de los parámetros a optimizar
        fitness_function : Callable, optional
            Función de fitness personalizada
        """
        if not DEAP_AVAILABLE:
            raise ImportError("DEAP requerido para optimización. Instalar con: pip install deap")
            
        self.ga_config = ga_config
        self.parameter_bounds = parameter_bounds
        self.fitness_function = fitness_function or self._default_fitness_function
        
        # Configurar salida
        self.output_dir = Path(ga_config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Datos de entrenamiento
        self.training_data: Optional[Dict] = None
        self.validation_data: Optional[Dict] = None
        
        # Estadísticas de optimización
        self.optimization_stats = {
            'generation_fitness': [],
            'best_individual_history': [],
            'convergence_history': [],
            'computation_times': []
        }
        
        # Configurar DEAP
        self._setup_deap()
        
        logger.info("Optimizador WoE-AC inicializado")
        
    def _setup_deap(self) -> None:
        """Configurar estructura DEAP para optimización."""
        
        # Crear tipos base
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
            
        # Configurar toolbox
        self.toolbox = base.Toolbox()
        
        # Generadores de genes
        self.toolbox.register("distance_weight", random.uniform, 
                             min(self.parameter_bounds.distance_weights_min),
                             max(self.parameter_bounds.distance_weights_max))
        
        self.toolbox.register("growth_prob", random.uniform,
                             self.parameter_bounds.growth_probability_min,
                             self.parameter_bounds.growth_probability_max)
        
        self.toolbox.register("neighborhood_inf", random.uniform,
                             self.parameter_bounds.neighborhood_influence_min,
                             self.parameter_bounds.neighborhood_influence_max)
                             
        self.toolbox.register("spontaneous", random.uniform,
                             self.parameter_bounds.spontaneous_growth_min,
                             self.parameter_bounds.spontaneous_growth_max)
                             
        self.toolbox.register("protected_inf", random.uniform,
                             self.parameter_bounds.protected_influence_min,
                             self.parameter_bounds.protected_influence_max)
                             
        self.toolbox.register("slope_thresh", random.uniform,
                             self.parameter_bounds.slope_threshold_min,
                             self.parameter_bounds.slope_threshold_max)
        
        # Constructor de individuos
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                             (self.toolbox.distance_weight, self.toolbox.distance_weight, 
                              self.toolbox.distance_weight, self.toolbox.growth_prob,
                              self.toolbox.neighborhood_inf, self.toolbox.spontaneous,
                              self.toolbox.protected_inf, self.toolbox.slope_thresh), n=1)
                              
        # Constructor de población
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Operadores genéticos
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=self.ga_config.tournament_size)
        self.toolbox.register("evaluate", self._evaluate_individual)
        
        # Configurar paralelización si está disponible
        if self.ga_config.n_processes > 1:
            from multiprocessing import Pool
            pool = Pool(processes=self.ga_config.n_processes)
            self.toolbox.register("map", pool.map)
    
    def set_training_data(
        self,
        initial_grid: np.ndarray,
        target_grid: np.ndarray,
        spatial_variables: Dict[str, np.ndarray],
        validation_split: float = 0.2
    ) -> None:
        """
        Configurar datos de entrenamiento y validación.
        
        Parameters:
        -----------
        initial_grid : np.ndarray
            Grid urbano inicial
        target_grid : np.ndarray  
            Grid urbano objetivo (ground truth)
        spatial_variables : Dict[str, np.ndarray]
            Variables espaciales para WoE
        validation_split : float, default=0.2
            Proporción de datos para validación
        """
        self.training_data = {
            'initial_grid': initial_grid,
            'target_grid': target_grid,
            'spatial_variables': spatial_variables
        }
        
        # Para simplificar, usar mismos datos para validación
        # En implementación real, usar split temporal o espacial
        self.validation_data = self.training_data.copy()
        
        logger.info(f"Datos configurados: grid {initial_grid.shape}, "
                   f"{len(spatial_variables)} variables espaciales")
    
    def _individual_to_parameters(self, individual: List[float]) -> ACParameters:
        """
        Convertir individuo de GA a parámetros AC.
        
        Parameters:
        -----------
        individual : List[float]
            Lista de genes del individuo
            
        Returns:
        --------
        ACParameters
            Parámetros del autómata celular
        """
        return ACParameters(
            distance_weights=[individual[0], individual[1], individual[2]],
            growth_probability=individual[3],
            neighborhood_influence=individual[4],
            spontaneous_growth_prob=individual[5],
            protected_influence=individual[6],
            slope_threshold=individual[7]
        )
    
    def _evaluate_individual(self, individual: List[float]) -> Tuple[float,]:
        """
        Evaluar fitness de un individuo.
        
        Parameters:
        -----------
        individual : List[float]
            Genes del individuo a evaluar
            
        Returns:
        --------
        Tuple[float,]
            Fitness del individuo (tupla requerida por DEAP)
        """
        try:
            # Convertir a parámetros
            parameters = self._individual_to_parameters(individual)
            
            # Evaluar usando función de fitness
            fitness = self.fitness_function(parameters, self.training_data)
            
            return (fitness,)
            
        except Exception as e:
            logger.warning(f"Error evaluando individuo: {e}")
            return (0.0,)  # Fitness penalizado
    
    def _default_fitness_function(
        self, 
        parameters: ACParameters, 
        data: Dict
    ) -> float:
        """
        Función de fitness por defecto basada en precisión espacial.
        
        Parameters:
        -----------
        parameters : ACParameters
            Parámetros a evaluar
        data : Dict
            Datos de entrenamiento
            
        Returns:
        --------
        float
            Valor de fitness (0-1)
        """
        try:
            # Configurar simulación
            simulator = ACSimulator(
                grid_shape=data['initial_grid'].shape,
                config=SimulationConfig(n_steps=1, save_all_steps=False)
            )
            
            simulator.setup_simulation(
                initial_urban_grid=data['initial_grid'],
                spatial_variables=data['spatial_variables'],
                parameters=parameters,
                target_grid=data['target_grid']
            )
            
            # Ejecutar simulación
            results = simulator.run_simulation()
            
            if not results:
                return 0.0
                
            # Calcular métricas
            final_grid = results[-1].grid
            metrics = calculate_spatial_metrics(final_grid, data['target_grid'])
            
            # Fitness compuesto (IoU + Kappa)
            iou = metrics.get('iou', 0.0)
            kappa = metrics.get('kappa', 0.0)
            
            # Combinar métricas con pesos
            fitness = 0.6 * iou + 0.4 * max(0.0, kappa)
            
            return float(fitness)
            
        except Exception as e:
            logger.warning(f"Error en función de fitness: {e}")
            return 0.0
    
    def optimize(self) -> Dict[str, Any]:
        """
        Ejecutar optimización completa.
        
        Returns:
        --------
        Dict[str, Any]
            Resultados de optimización
        """
        if self.training_data is None:
            raise ValueError("Datos de entrenamiento no configurados")
            
        logger.info(f"Iniciando optimización: {self.ga_config.n_generations} generaciones, "
                   f"población {self.ga_config.population_size}")
        
        # Configurar semillas
        random.seed(self.ga_config.random_seed)
        np.random.seed(self.ga_config.random_seed)
        
        # Crear población inicial
        population = self.toolbox.population(n=self.ga_config.population_size)
        
        # Estadísticas
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Hall of Fame
        hof = tools.HallOfFame(1)
        
        # Algoritmo evolutivo
        population, logbook = algorithms.eaSimple(
            population, self.toolbox,
            cxpb=self.ga_config.crossover_prob,
            mutpb=self.ga_config.mutation_prob,
            ngen=self.ga_config.n_generations,
            stats=stats,
            halloffame=hof,
            verbose=True
        )
        
        # Mejor individuo
        best_individual = hof[0]
        best_parameters = self._individual_to_parameters(best_individual)
        best_fitness = best_individual.fitness.values[0]
        
        # Resultados
        results = {
            'best_parameters': best_parameters,
            'best_fitness': best_fitness,
            'optimization_history': logbook,
            'final_population': population,
            'convergence_data': self.optimization_stats
        }
        
        # Guardar resultados
        self._save_results(results)
        
        logger.info(f"Optimización completada. Mejor fitness: {best_fitness:.4f}")
        
        return results
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Guardar resultados de optimización."""
        
        # Guardar parámetros óptimos
        best_params_file = self.output_dir / "best_parameters.json"
        with open(best_params_file, 'w') as f:
            # Convertir parámetros a dict serializable
            params_dict = {
                'distance_weights': results['best_parameters'].distance_weights,
                'growth_probability': results['best_parameters'].growth_probability,
                'neighborhood_influence': results['best_parameters'].neighborhood_influence,
                'spontaneous_growth_prob': results['best_parameters'].spontaneous_growth_prob,
                'protected_influence': results['best_parameters'].protected_influence,
                'slope_threshold': results['best_parameters'].slope_threshold,
                'best_fitness': results['best_fitness']
            }
            json.dump(params_dict, f, indent=2)
            
        # Guardar datos completos
        full_results_file = self.output_dir / "optimization_results.pkl"
        with open(full_results_file, 'wb') as f:
            pickle.dump(results, f)
            
        logger.info(f"Resultados guardados en {self.output_dir}")
    
    def load_best_parameters(self, filepath: Optional[str] = None) -> ACParameters:
        """
        Cargar mejores parámetros desde archivo.
        
        Parameters:
        -----------
        filepath : str, optional
            Ruta al archivo de parámetros
            
        Returns:
        --------
        ACParameters
            Parámetros optimizados
        """
        if filepath is None:
            filepath = self.output_dir / "best_parameters.json"
            
        with open(filepath, 'r') as f:
            params_dict = json.load(f)
            
        return ACParameters(
            distance_weights=params_dict['distance_weights'],
            growth_probability=params_dict['growth_probability'],
            neighborhood_influence=params_dict['neighborhood_influence'],
            spontaneous_growth_prob=params_dict['spontaneous_growth_prob'],
            protected_influence=params_dict['protected_influence'],
            slope_threshold=params_dict['slope_threshold']
        )


def create_optimization_experiment(
    config_path: str,
    data_path: str,
    output_dir: str = "optimization_results"
) -> WoEACOptimizer:
    """
    Crear experimento de optimización desde archivos de configuración.
    
    Parameters:
    -----------
    config_path : str
        Ruta al archivo de configuración
    data_path : str
        Ruta a los datos de entrenamiento
    output_dir : str, default="optimization_results"
        Directorio de salida
        
    Returns:
    --------
    WoEACOptimizer
        Optimizador configurado
    """
    # Cargar configuración
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
        
    ga_config = GAConfig(**config_dict.get('ga_config', {}))
    parameter_bounds = ParameterBounds(**config_dict.get('parameter_bounds', {}))
    
    # Crear optimizador
    optimizer = WoEACOptimizer(
        ga_config=ga_config,
        parameter_bounds=parameter_bounds
    )
    
    # Cargar datos (implementar según formato)
    # optimizer.set_training_data(...)
    
    return optimizer


def run_parameter_sensitivity_analysis(
    optimizer: WoEACOptimizer,
    parameter_ranges: Dict[str, Tuple[float, float]],
    n_samples: int = 100
) -> Dict[str, List[float]]:
    """
    Ejecutar análisis de sensibilidad de parámetros.
    
    Parameters:
    -----------
    optimizer : WoEACOptimizer
        Optimizador configurado
    parameter_ranges : Dict[str, Tuple[float, float]]
        Rangos de parámetros a analizar
    n_samples : int, default=100
        Número de muestras por parámetro
        
    Returns:
    --------
    Dict[str, List[float]]
        Resultados de sensibilidad
    """
    logger.info("Iniciando análisis de sensibilidad de parámetros")
    
    sensitivity_results = {}
    
    for param_name, (min_val, max_val) in parameter_ranges.items():
        param_fitness = []
        param_values = np.linspace(min_val, max_val, n_samples)
        
        for value in param_values:
            # Crear parámetros base
            base_params = ACParameters()
            
            # Modificar parámetro específico
            setattr(base_params, param_name, value)
            
            # Evaluar fitness
            fitness = optimizer.fitness_function(base_params, optimizer.training_data)
            param_fitness.append(fitness)
            
        sensitivity_results[param_name] = {
            'values': param_values.tolist(),
            'fitness': param_fitness
        }
        
    return sensitivity_results