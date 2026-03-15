#!/usr/bin/env python3
"""
Script principal para clasificación y generación de grids urbanos.

Este script ejecuta el pipeline completo de modelado WoE-AC-GA:
1. Carga y preprocesamiento de datos espaciales
2. Cálculo de Weight of Evidence
3. Simulación con Autómata Celular  
4. Optimización con Algoritmo Genético
5. Generación de resultados y visualizaciones
"""

import sys
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import json

# Agregar src al path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from tesis_ac.config import load_config
from tesis_ac.utils.io import load_spatial_data
from tesis_ac.woe.woe import WoECalculator
from tesis_ac.ca.rules import ACParameters, WoECellularAutomaton
from tesis_ac.ca.simulate import ACSimulator, SimulationConfig
from tesis_ac.ga.optimizer import WoEACOptimizer, GAConfig, ParameterBounds
from tesis_ac.eval.metrics import calculate_spatial_metrics, evaluate_model_performance

logger = logging.getLogger(__name__)


class WoEACPipeline:
    """
    Pipeline principal para modelado urbano WoE-AC-GA.
    
    Integra todos los componentes del modelo para ejecutar experimentos
    completos de simulación y optimización de crecimiento urbano.
    """
    
    def __init__(self, config: dict, output_dir: Path):
        """
        Inicializar pipeline.
        
        Parameters:
        -----------
        config : dict
            Configuración del experimento
        output_dir : Path
            Directorio de salida
        """
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Componentes del modelo
        self.woe_calculator: WoECalculator = None
        self.ac_simulator: ACSimulator = None
        self.optimizer: WoEACOptimizer = None
        
        # Datos del experimento
        self.spatial_data = {}
        self.urban_grids = {}
        self.woe_results = {}
        
        # Resultados
        self.experiment_results = {}
        
        logger.info(f"Pipeline WoE-AC inicializado: {output_dir}")
    
    def load_spatial_data(self, data_dir: Path) -> None:
        """
        Cargar datos espaciales del experimento.
        
        Parameters:
        -----------
        data_dir : Path
            Directorio con datos de entrada
        """
        logger.info(f"Cargando datos espaciales desde {data_dir}")
        
        try:
            # Cargar grids urbanos para diferentes años
            urban_years = [1990, 2000, 2010, 2020]
            for year in urban_years:
                urban_file = data_dir / f"urban_{year}.npy"
                if urban_file.exists():
                    self.urban_grids[year] = np.load(urban_file)
                    logger.info(f"Grid urbano {year} cargado: {self.urban_grids[year].shape}")
                else:
                    logger.warning(f"Archivo {urban_file} no encontrado")
            
            # Si no hay datos reales, generar sintéticos
            if not self.urban_grids:
                logger.warning("Generando datos urbanos sintéticos")
                self._generate_synthetic_urban_data()
            
            # Cargar variables espaciales
            spatial_vars = {
                'distance_to_roads': 'distance_roads.npy',
                'distance_to_centers': 'distance_centers.npy',
                'distance_to_water': 'distance_water.npy',
                'slope': 'slope.npy',
                'elevation': 'elevation.npy',
                'protected_areas': 'protected.npy'
            }
            
            grid_shape = list(self.urban_grids.values())[0].shape
            
            for var_name, filename in spatial_vars.items():
                var_file = data_dir / filename
                if var_file.exists():
                    self.spatial_data[var_name] = np.load(var_file)
                else:
                    logger.warning(f"Generando variable sintética: {var_name}")
                    if var_name == 'protected_areas':
                        self.spatial_data[var_name] = np.random.choice([0, 1], size=grid_shape, p=[0.85, 0.15])
                    else:
                        self.spatial_data[var_name] = np.random.uniform(0, 100, size=grid_shape)
                        
            logger.info(f"Variables espaciales cargadas: {len(self.spatial_data)}")
            
        except Exception as e:
            logger.error(f"Error cargando datos espaciales: {e}")
            raise
    
    def _generate_synthetic_urban_data(self) -> None:
        """Generar datos urbanos sintéticos para pruebas."""
        grid_size = (150, 150)
        
        # Simular crecimiento urbano gradual
        np.random.seed(42)
        
        # 1990: núcleo urbano inicial
        center = (grid_size[0]//2, grid_size[1]//2)
        urban_1990 = np.zeros(grid_size, dtype=int)
        
        # Crear núcleo urbano compacto
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                if dist < 15:
                    urban_1990[i, j] = 1
                elif dist < 25 and np.random.random() < 0.3:
                    urban_1990[i, j] = 1
        
        # 2000: expansión moderada
        urban_2000 = urban_1990.copy()
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                if urban_1990[i, j] == 0:
                    # Probabilidad basada en proximidad urbana
                    neighbors = self._count_urban_neighbors(urban_1990, i, j, radius=3)
                    prob = neighbors * 0.1
                    if np.random.random() < prob:
                        urban_2000[i, j] = 1
        
        # 2010: expansión acelerada
        urban_2010 = urban_2000.copy()
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                if urban_2000[i, j] == 0:
                    neighbors = self._count_urban_neighbors(urban_2000, i, j, radius=5)
                    prob = neighbors * 0.15
                    if np.random.random() < prob:
                        urban_2010[i, j] = 1
        
        # 2020: saturación
        urban_2020 = urban_2010.copy()
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                if urban_2010[i, j] == 0:
                    neighbors = self._count_urban_neighbors(urban_2010, i, j, radius=2)
                    prob = neighbors * 0.08
                    if np.random.random() < prob:
                        urban_2020[i, j] = 1
        
        self.urban_grids = {
            1990: urban_1990,
            2000: urban_2000, 
            2010: urban_2010,
            2020: urban_2020
        }
        
        logger.info("Datos urbanos sintéticos generados")
    
    def _count_urban_neighbors(self, grid: np.ndarray, i: int, j: int, radius: int = 1) -> int:
        """Contar vecinos urbanos en un radio dado."""
        count = 0
        rows, cols = grid.shape
        
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols and grid[ni, nj] == 1:
                    count += 1
                    
        return count
    
    def calculate_woe(self) -> None:
        """Calcular Weight of Evidence para variables espaciales."""
        logger.info("Calculando Weight of Evidence")
        
        try:
            # Usar períodos de entrenamiento (1990-2000)
            initial_grid = self.urban_grids[1990]
            final_grid = self.urban_grids[2000]
            
            # Identificar cambio urbano
            change_grid = (final_grid == 1) & (initial_grid == 0)
            
            # Inicializar calculadora WoE
            self.woe_calculator = WoECalculator()
            
            # Calcular WoE para cada variable
            for var_name, var_data in self.spatial_data.items():
                if var_name == 'protected_areas':
                    # Variable categórica
                    woe_results = self.woe_calculator.calculate_woe(
                        spatial_variable=var_data,
                        urban_change=change_grid,
                        method='categorical'
                    )
                else:
                    # Variable continua
                    woe_results = self.woe_calculator.calculate_woe(
                        spatial_variable=var_data,
                        urban_change=change_grid,
                        method='equal_interval',
                        n_bins=10
                    )
                
                self.woe_results[var_name] = woe_results
                
                logger.info(f"WoE calculado para {var_name}: IV = {woe_results['information_value']:.4f}")
            
            # Guardar resultados WoE
            self._save_woe_results()
            
        except Exception as e:
            logger.error(f"Error calculando WoE: {e}")
            raise
    
    def _save_woe_results(self) -> None:
        """Guardar resultados de Weight of Evidence."""
        woe_dir = self.output_dir / "woe_results"
        woe_dir.mkdir(exist_ok=True)
        
        # Guardar resumen WoE
        woe_summary = {}
        for var_name, results in self.woe_results.items():
            woe_summary[var_name] = {
                'information_value': float(results['information_value']),
                'predictive_power': results['predictive_power'],
                'n_bins': len(results['woe_table'])
            }
        
        with open(woe_dir / "woe_summary.json", 'w') as f:
            json.dump(woe_summary, f, indent=2)
        
        # Guardar tablas WoE detalladas
        for var_name, results in self.woe_results.items():
            table_file = woe_dir / f"{var_name}_woe_table.json"
            with open(table_file, 'w') as f:
                # Convertir tabla a formato serializable
                serializable_table = []
                for row in results['woe_table']:
                    serializable_table.append({
                        'bin_range': [float(row['bin_min']), float(row['bin_max'])],
                        'count_positive': int(row['count_positive']),
                        'count_negative': int(row['count_negative']),
                        'woe_value': float(row['woe']),
                        'information_value': float(row['iv'])
                    })
                json.dump(serializable_table, f, indent=2)
        
        logger.info(f"Resultados WoE guardados en {woe_dir}")
    
    def setup_simulation(self) -> None:
        """Configurar simulador AC con parámetros optimizados."""
        logger.info("Configurando simulador de Autómata Celular")
        
        try:
            # Configuración base del simulador
            grid_shape = list(self.urban_grids.values())[0].shape
            sim_config = SimulationConfig(
                n_steps=self.config.get('simulation_steps', 10),
                save_all_steps=True,
                random_seed=self.config.get('random_seed', 42)
            )
            
            self.ac_simulator = ACSimulator(grid_shape, sim_config)
            
            logger.info(f"Simulador configurado: {grid_shape}, {sim_config.n_steps} pasos")
            
        except Exception as e:
            logger.error(f"Error configurando simulador: {e}")
            raise
    
    def run_optimization(self) -> None:
        """Ejecutar optimización de parámetros con GA."""
        logger.info("Ejecutando optimización con Algoritmo Genético")
        
        try:
            # Configurar optimizador
            ga_config = GAConfig(
                population_size=self.config.get('population_size', 20),
                n_generations=self.config.get('n_generations', 25),
                crossover_prob=0.7,
                mutation_prob=0.2,
                n_processes=self.config.get('n_processes', 2),
                output_dir=str(self.output_dir / "optimization"),
                random_seed=self.config.get('random_seed', 42)
            )
            
            parameter_bounds = ParameterBounds()
            
            self.optimizer = WoEACOptimizer(ga_config, parameter_bounds)
            
            # Configurar datos de entrenamiento (1990 -> 2000)  
            self.optimizer.set_training_data(
                initial_grid=self.urban_grids[1990],
                target_grid=self.urban_grids[2000],
                spatial_variables=self.spatial_data
            )
            
            # Ejecutar optimización
            optimization_results = self.optimizer.optimize()
            
            self.experiment_results['optimization'] = optimization_results
            
            logger.info(f"Optimización completada. Mejor fitness: {optimization_results['best_fitness']:.4f}")
            
        except Exception as e:
            logger.error(f"Error en optimización: {e}")
            raise
    
    def run_validation_experiments(self) -> None:
        """Ejecutar experimentos de validación temporal."""
        logger.info("Ejecutando experimentos de validación")
        
        try:
            if 'optimization' not in self.experiment_results:
                logger.warning("No hay parámetros optimizados. Usando parámetros por defecto.")
                best_params = ACParameters()
            else:
                best_params = self.experiment_results['optimization']['best_parameters']
            
            validation_results = {}
            
            # Experimento 1: 1990 -> 2000 (entrenamiento)
            val_1990_2000 = self._run_single_validation(
                initial_year=1990, target_year=2000, 
                parameters=best_params, experiment_name="train_1990_2000"
            )
            validation_results['train_1990_2000'] = val_1990_2000
            
            # Experimento 2: 2000 -> 2010 (validación)
            if 2010 in self.urban_grids:
                val_2000_2010 = self._run_single_validation(
                    initial_year=2000, target_year=2010,
                    parameters=best_params, experiment_name="validation_2000_2010"
                )
                validation_results['validation_2000_2010'] = val_2000_2010
            
            # Experimento 3: 2010 -> 2020 (prueba)
            if 2020 in self.urban_grids:
                val_2010_2020 = self._run_single_validation(
                    initial_year=2010, target_year=2020,
                    parameters=best_params, experiment_name="test_2010_2020"
                )
                validation_results['test_2010_2020'] = val_2010_2020
            
            self.experiment_results['validation'] = validation_results
            
            logger.info("Experimentos de validación completados")
            
        except Exception as e:
            logger.error(f"Error en validación: {e}")
            raise
    
    def _run_single_validation(
        self, 
        initial_year: int, 
        target_year: int,
        parameters: ACParameters,
        experiment_name: str
    ) -> dict:
        """Ejecutar un experimento de validación individual."""
        
        logger.info(f"Validación {experiment_name}: {initial_year} -> {target_year}")
        
        # Configurar simulación
        self.ac_simulator.setup_simulation(
            initial_urban_grid=self.urban_grids[initial_year],
            spatial_variables=self.spatial_data,
            parameters=parameters,
            target_grid=self.urban_grids[target_year]
        )
        
        # Ejecutar simulación
        simulation_results = self.ac_simulator.run_simulation()
        
        if not simulation_results:
            return {'error': 'Simulación falló'}
        
        # Evaluar resultados
        final_grid = simulation_results[-1].grid
        metrics = calculate_spatial_metrics(final_grid, self.urban_grids[target_year])
        
        # Guardar resultados
        experiment_dir = self.output_dir / "validation" / experiment_name
        experiment_dir.mkdir(exist_ok=True, parents=True)
        
        # Guardar grid final
        np.save(experiment_dir / "predicted_grid.npy", final_grid)
        np.save(experiment_dir / "target_grid.npy", self.urban_grids[target_year])
        
        # Guardar métricas
        with open(experiment_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return {
            'metrics': metrics,
            'simulation_steps': len(simulation_results),
            'final_urban_cells': int(np.sum(final_grid)),
            'target_urban_cells': int(np.sum(self.urban_grids[target_year]))
        }
    
    def generate_final_report(self) -> None:
        """Generar reporte final del experimento."""
        logger.info("Generando reporte final")
        
        report_file = self.output_dir / "experiment_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("REPORTE EXPERIMENTAL WoE-AC-GA\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuración: {json.dumps(self.config, indent=2)}\n\n")
            
            # Resumen de datos
            f.write("DATOS EXPERIMENTALES:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Años urbanos: {list(self.urban_grids.keys())}\n")
            f.write(f"Variables espaciales: {list(self.spatial_data.keys())}\n")
            if self.urban_grids:
                grid_shape = list(self.urban_grids.values())[0].shape
                f.write(f"Dimensiones grid: {grid_shape}\n")
            f.write("\n")
            
            # Resultados WoE
            if self.woe_results:
                f.write("WEIGHT OF EVIDENCE:\n")
                f.write("-" * 18 + "\n")
                for var_name, results in self.woe_results.items():
                    iv = results['information_value']
                    power = results['predictive_power']
                    f.write(f"{var_name}: IV={iv:.4f} ({power})\n")
                f.write("\n")
            
            # Resultados optimización
            if 'optimization' in self.experiment_results:
                opt_results = self.experiment_results['optimization']
                f.write("OPTIMIZACIÓN:\n")
                f.write("-" * 13 + "\n")
                f.write(f"Mejor fitness: {opt_results['best_fitness']:.4f}\n")
                
                best_params = opt_results['best_parameters']
                f.write("Mejores parámetros:\n")
                f.write(f"  - Distance weights: {best_params.distance_weights}\n")
                f.write(f"  - Growth probability: {best_params.growth_probability:.6f}\n")
                f.write(f"  - Neighborhood influence: {best_params.neighborhood_influence:.4f}\n")
                f.write("\n")
            
            # Resultados validación
            if 'validation' in self.experiment_results:
                f.write("VALIDACIÓN:\n")
                f.write("-" * 11 + "\n")
                val_results = self.experiment_results['validation']
                
                for exp_name, exp_data in val_results.items():
                    if 'metrics' in exp_data:
                        metrics = exp_data['metrics']
                        f.write(f"{exp_name}:\n")
                        f.write(f"  - IoU: {metrics.get('iou', 'N/A'):.4f}\n")
                        f.write(f"  - Kappa: {metrics.get('kappa', 'N/A'):.4f}\n")
                        f.write(f"  - Accuracy: {metrics.get('accuracy', 'N/A'):.4f}\n")
        
        logger.info(f"Reporte generado: {report_file}")
    
    def run_complete_experiment(self, data_dir: Path) -> dict:
        """Ejecutar experimento completo WoE-AC-GA."""
        logger.info("="*60)
        logger.info("INICIANDO EXPERIMENTO COMPLETO WoE-AC-GA")
        logger.info("="*60)
        
        try:
            # 1. Cargar datos
            self.load_spatial_data(data_dir)
            
            # 2. Calcular WoE
            self.calculate_woe()
            
            # 3. Configurar simulación
            self.setup_simulation()
            
            # 4. Ejecutar optimización
            if self.config.get('run_optimization', True):
                self.run_optimization()
            
            # 5. Experimentos de validación
            self.run_validation_experiments()
            
            # 6. Generar reporte
            self.generate_final_report()
            
            logger.info("="*60)
            logger.info("EXPERIMENTO COMPLETADO EXITOSAMENTE")
            logger.info("="*60)
            
            return self.experiment_results
            
        except Exception as e:
            logger.error(f"Error en experimento: {e}")
            raise


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Pipeline completo WoE-AC-GA')
    parser.add_argument('--config', help='Archivo de configuración YAML')
    parser.add_argument('--data-dir', default='data/processed',
                       help='Directorio con datos de entrada')
    parser.add_argument('--output-dir', default='experiment_results',
                       help='Directorio de salida')
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
        # Cargar configuración
        if args.config:
            config = load_config(args.config)
        else:
            config = {
                'simulation_steps': 5,
                'population_size': 15,
                'n_generations': 20,
                'n_processes': 2,
                'random_seed': 42,
                'run_optimization': True
            }
        
        # Crear pipeline
        output_dir = Path(args.output_dir)
        pipeline = WoEACPipeline(config, output_dir)
        
        # Ejecutar experimento completo
        results = pipeline.run_complete_experiment(Path(args.data_dir))
        
        print(f"\n✅ Experimento completado exitosamente!")
        print(f"Resultados en: {output_dir}")
        
        # Mostrar resumen de resultados
        if 'optimization' in results:
            opt_fitness = results['optimization']['best_fitness']
            print(f"Mejor fitness optimización: {opt_fitness:.4f}")
        
        if 'validation' in results:
            val_count = len(results['validation'])
            print(f"Experimentos de validación: {val_count}")
        
    except Exception as e:
        logger.error(f"Error en pipeline: {e}")
        print(f"\n❌ Error en experimento: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()