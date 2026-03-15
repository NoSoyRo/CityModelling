#!/usr/bin/env python3
"""
Script principal para optimización de parámetros usando algoritmo genético.

Este script ejecuta la calibración automática de parámetros del modelo
WoE-AC usando algoritmos genéticos, permitiendo encontrar la configuración
óptima para modelado de crecimiento urbano.
"""

import sys
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse

# Agregar src al path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from tesis_ac.config import load_config
from tesis_ac.utils.io import load_spatial_data
from tesis_ac.woe.woe import WoECalculator
from tesis_ac.ca.rules import ACParameters
from tesis_ac.ga.optimizer import WoEACOptimizer, GAConfig, ParameterBounds
from tesis_ac.ca.simulate import ACSimulator, SimulationConfig

logger = logging.getLogger(__name__)


def setup_optimization_experiment(
    data_dir: Path,
    output_dir: Path,
    config: dict
) -> WoEACOptimizer:
    """
    Configurar experimento de optimización con datos reales.
    
    Parameters:
    -----------
    data_dir : Path
        Directorio con datos de entrada
    output_dir : Path
        Directorio de salida
    config : dict
        Configuración del experimento
        
    Returns:
    --------
    WoEACOptimizer
        Optimizador configurado
    """
    logger.info("Configurando experimento de optimización")
    
    # Configuración del algoritmo genético
    ga_config = GAConfig(
        population_size=config.get('population_size', 30),
        n_generations=config.get('n_generations', 50),
        crossover_prob=config.get('crossover_prob', 0.7),
        mutation_prob=config.get('mutation_prob', 0.2),
        tournament_size=config.get('tournament_size', 3),
        n_processes=config.get('n_processes', 2),
        output_dir=str(output_dir),
        random_seed=config.get('random_seed', 42)
    )
    
    # Límites de parámetros
    parameter_bounds = ParameterBounds(
        distance_weights_min=[0.1, 0.1, 0.1],
        distance_weights_max=[3.0, 2.0, 1.5],
        growth_probability_min=0.001,
        growth_probability_max=0.05,
        neighborhood_influence_min=0.2,
        neighborhood_influence_max=1.0,
        spontaneous_growth_min=0.0001,
        spontaneous_growth_max=0.005,
        protected_influence_min=0.7,
        protected_influence_max=1.0,
        slope_threshold_min=10.0,
        slope_threshold_max=35.0
    )
    
    # Crear optimizador
    optimizer = WoEACOptimizer(
        ga_config=ga_config,
        parameter_bounds=parameter_bounds
    )
    
    return optimizer


def load_training_data(data_dir: Path) -> tuple:
    """
    Cargar datos de entrenamiento para optimización.
    
    Parameters:
    -----------
    data_dir : Path
        Directorio con datos
        
    Returns:
    --------
    tuple
        (initial_grid, target_grid, spatial_variables)
    """
    logger.info(f"Cargando datos de entrenamiento desde {data_dir}")
    
    try:
        # Cargar grids urbanos (ejemplo con archivos numpy)
        initial_grid_file = data_dir / "urban_1990.npy"
        target_grid_file = data_dir / "urban_2000.npy"
        
        if initial_grid_file.exists() and target_grid_file.exists():
            initial_grid = np.load(initial_grid_file)
            target_grid = np.load(target_grid_file)
        else:
            # Crear datos sintéticos para prueba
            logger.warning("Archivos de datos no encontrados. Generando datos sintéticos.")
            grid_size = (100, 100)
            initial_grid = np.random.choice([0, 1], size=grid_size, p=[0.8, 0.2])
            target_grid = np.random.choice([0, 1], size=grid_size, p=[0.7, 0.3])
        
        # Cargar variables espaciales
        spatial_variables = {}
        
        # Variables básicas (crear sintéticas si no existen)
        var_files = {
            'distance_to_roads': 'distance_roads.npy',
            'distance_to_centers': 'distance_centers.npy', 
            'slope': 'slope.npy',
            'protected_areas': 'protected.npy'
        }
        
        for var_name, filename in var_files.items():
            var_file = data_dir / filename
            if var_file.exists():
                spatial_variables[var_name] = np.load(var_file)
            else:
                # Generar variable sintética
                logger.warning(f"Archivo {filename} no encontrado. Generando datos sintéticos.")
                if var_name == 'protected_areas':
                    spatial_variables[var_name] = np.random.choice([0, 1], size=initial_grid.shape, p=[0.9, 0.1])
                else:
                    spatial_variables[var_name] = np.random.uniform(0, 100, size=initial_grid.shape)
        
        logger.info(f"Datos cargados: grid {initial_grid.shape}, {len(spatial_variables)} variables")
        
        return initial_grid, target_grid, spatial_variables
        
    except Exception as e:
        logger.error(f"Error cargando datos: {e}")
        raise


def run_optimization_experiment(
    data_dir: str = "data/processed",
    output_dir: str = "optimization_results",
    config_file: str = None
) -> dict:
    """
    Ejecutar experimento completo de optimización.
    
    Parameters:
    -----------
    data_dir : str
        Directorio con datos de entrada
    output_dir : str
        Directorio de salida
    config_file : str, optional
        Archivo de configuración personalizada
        
    Returns:
    --------
    dict
        Resultados de optimización
    """
    # Configurar directorios
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Configurar logging específico para el experimento
    log_file = output_path / f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)
    
    logger.info("="*50)
    logger.info("INICIANDO EXPERIMENTO DE OPTIMIZACIÓN WoE-AC")
    logger.info("="*50)
    
    # Cargar configuración
    if config_file:
        config = load_config(config_file)
    else:
        config = {
            'population_size': 20,
            'n_generations': 30,
            'crossover_prob': 0.7,
            'mutation_prob': 0.2,
            'n_processes': 2,
            'random_seed': 42
        }
    
    logger.info(f"Configuración: {config}")
    
    try:
        # 1. Configurar experimento
        optimizer = setup_optimization_experiment(data_path, output_path, config)
        
        # 2. Cargar datos
        initial_grid, target_grid, spatial_variables = load_training_data(data_path)
        
        # 3. Configurar datos de entrenamiento
        optimizer.set_training_data(
            initial_grid=initial_grid,
            target_grid=target_grid,
            spatial_variables=spatial_variables,
            validation_split=0.2
        )
        
        # 4. Ejecutar optimización
        logger.info("Iniciando proceso de optimización...")
        results = optimizer.optimize()
        
        # 5. Validar resultados
        logger.info("Validando mejores parámetros...")
        best_params = results['best_parameters']
        validation_fitness = validate_best_parameters(
            best_params, initial_grid, target_grid, spatial_variables
        )
        
        results['validation_fitness'] = validation_fitness
        
        # 6. Generar reporte
        generate_optimization_report(results, output_path)
        
        logger.info("="*50)
        logger.info("OPTIMIZACIÓN COMPLETADA EXITOSAMENTE")
        logger.info(f"Mejor fitness: {results['best_fitness']:.4f}")
        logger.info(f"Fitness de validación: {validation_fitness:.4f}")
        logger.info(f"Resultados guardados en: {output_path}")
        logger.info("="*50)
        
        return results
        
    except Exception as e:
        logger.error(f"Error en experimento de optimización: {e}")
        raise


def validate_best_parameters(
    parameters: ACParameters,
    initial_grid: np.ndarray,
    target_grid: np.ndarray,
    spatial_variables: dict
) -> float:
    """
    Validar los mejores parámetros con simulación independiente.
    
    Parameters:
    -----------
    parameters : ACParameters
        Parámetros optimizados
    initial_grid : np.ndarray
        Grid inicial
    target_grid : np.ndarray
        Grid objetivo
    spatial_variables : dict
        Variables espaciales
        
    Returns:
    --------
    float
        Fitness de validación
    """
    try:
        # Configurar simulador
        simulator = ACSimulator(
            grid_shape=initial_grid.shape,
            config=SimulationConfig(
                n_steps=1,
                save_all_steps=False,
                random_seed=123  # Semilla diferente para validación
            )
        )
        
        # Ejecutar simulación
        simulator.setup_simulation(
            initial_urban_grid=initial_grid,
            spatial_variables=spatial_variables,
            parameters=parameters,
            target_grid=target_grid
        )
        
        results = simulator.run_simulation()
        
        if not results:
            return 0.0
            
        # Calcular métricas de validación
        from tesis_ac.eval.metrics import calculate_spatial_metrics
        
        final_grid = results[-1].grid
        metrics = calculate_spatial_metrics(final_grid, target_grid)
        
        # Fitness de validación
        iou = metrics.get('iou', 0.0)
        kappa = metrics.get('kappa', 0.0)
        validation_fitness = 0.6 * iou + 0.4 * max(0.0, kappa)
        
        return float(validation_fitness)
        
    except Exception as e:
        logger.warning(f"Error en validación: {e}")
        return 0.0


def generate_optimization_report(results: dict, output_dir: Path) -> None:
    """
    Generar reporte de optimización.
    
    Parameters:
    -----------
    results : dict
        Resultados de optimización
    output_dir : Path
        Directorio de salida
    """
    report_file = output_dir / "optimization_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("REPORTE DE OPTIMIZACIÓN WoE-AC\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("MEJORES PARÁMETROS ENCONTRADOS:\n")
        f.write("-" * 30 + "\n")
        best_params = results['best_parameters']
        f.write(f"Distance Weights: {best_params.distance_weights}\n")
        f.write(f"Growth Probability: {best_params.growth_probability:.6f}\n")
        f.write(f"Neighborhood Influence: {best_params.neighborhood_influence:.4f}\n")
        f.write(f"Spontaneous Growth: {best_params.spontaneous_growth_prob:.6f}\n")
        f.write(f"Protected Influence: {best_params.protected_influence:.4f}\n")
        f.write(f"Slope Threshold: {best_params.slope_threshold:.2f}\n\n")
        
        f.write("RENDIMIENTO:\n")
        f.write("-" * 15 + "\n")
        f.write(f"Mejor Fitness (Entrenamiento): {results['best_fitness']:.4f}\n")
        if 'validation_fitness' in results:
            f.write(f"Fitness de Validación: {results['validation_fitness']:.4f}\n")
        f.write("\n")
        
        if 'optimization_history' in results:
            logbook = results['optimization_history']
            f.write("EVOLUCIÓN DE FITNESS:\n")
            f.write("-" * 20 + "\n")
            for i, record in enumerate(logbook):
                if i % 5 == 0 or i == len(logbook) - 1:  # Cada 5 generaciones
                    f.write(f"Generación {record['gen']:3d}: "
                           f"Max={record['max']:.4f}, "
                           f"Avg={record['avg']:.4f}, "
                           f"Std={record['std']:.4f}\n")
    
    logger.info(f"Reporte generado: {report_file}")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Optimización de parámetros WoE-AC')
    parser.add_argument('--data-dir', default='data/processed',
                       help='Directorio con datos de entrada')
    parser.add_argument('--output-dir', default='optimization_results',
                       help='Directorio de salida')
    parser.add_argument('--config', help='Archivo de configuración')
    parser.add_argument('--verbose', action='store_true',
                       help='Salida detallada')
    
    args = parser.parse_args()
    
    # Configurar logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Ejecutar optimización
        results = run_optimization_experiment(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            config_file=args.config
        )
        
        print(f"\n✅ Optimización completada exitosamente!")
        print(f"Mejor fitness: {results['best_fitness']:.4f}")
        print(f"Resultados en: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Error en optimización: {e}")
        print(f"\n❌ Error en optimización: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()