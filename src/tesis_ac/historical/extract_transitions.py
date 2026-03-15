"""
Extracción de transiciones urbanas del histórico de mapas clasificados.

Este módulo analiza secuencias temporales de mapas binarios urbano/no-urbano
para identificar dónde y cuándo ocurrieron transiciones reales.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging
import pickle
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TransitionPeriod:
    """Información de un período de transición entre dos años."""
    year_from: int
    year_to: int
    time_span: int  # Años entre mediciones
    grid_t0: np.ndarray
    grid_t1: np.ndarray
    transitions: np.ndarray  # Máscara binaria: True donde hubo transición
    n_transitions: int
    n_urban_t0: int
    n_urban_t1: int
    growth_rate: float  # Porcentaje de crecimiento
    
    def __post_init__(self):
        """Calcular métricas derivadas."""
        if self.n_transitions == 0 and self.transitions is not None:
            self.n_transitions = int(np.sum(self.transitions))
        if self.n_urban_t0 == 0 and self.grid_t0 is not None:
            self.n_urban_t0 = int(np.sum(self.grid_t0 == 1))
        if self.n_urban_t1 == 0 and self.grid_t1 is not None:
            self.n_urban_t1 = int(np.sum(self.grid_t1 == 1))
        if self.growth_rate == 0 and self.n_urban_t0 > 0:
            self.growth_rate = (self.n_urban_t1 - self.n_urban_t0) / self.n_urban_t0 * 100


@dataclass
class HistoricalTransitionsDataset:
    """Dataset completo de transiciones históricas."""
    transitions: List[TransitionPeriod]
    years_available: List[int]
    grid_shape: Tuple[int, int]
    total_transitions: int = 0
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Calcular estadísticas agregadas."""
        if self.total_transitions == 0:
            self.total_transitions = sum(t.n_transitions for t in self.transitions)
        
        self.metadata.update({
            'n_periods': len(self.transitions),
            'year_range': (min(self.years_available), max(self.years_available)),
            'total_years': max(self.years_available) - min(self.years_available),
            'extraction_date': datetime.now().isoformat()
        })
    
    def get_summary_dataframe(self) -> pd.DataFrame:
        """Crear DataFrame resumen de transiciones."""
        records = []
        for t in self.transitions:
            records.append({
                'year_from': t.year_from,
                'year_to': t.year_to,
                'time_span': t.time_span,
                'n_transitions': t.n_transitions,
                'urban_t0': t.n_urban_t0,
                'urban_t1': t.n_urban_t1,
                'growth_rate_%': t.growth_rate,
                'transition_rate_%': (t.n_transitions / t.n_urban_t0 * 100) if t.n_urban_t0 > 0 else 0
            })
        return pd.DataFrame(records)
    
    def save(self, filepath: Path) -> None:
        """Guardar dataset a disco."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        logger.info(f"Dataset guardado: {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'HistoricalTransitionsDataset':
        """Cargar dataset desde disco."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class HistoricalTransitionExtractor:
    """
    Extractor de transiciones urbanas del histórico.
    
    Analiza secuencias de mapas binarios para identificar:
    - Dónde ocurrieron transiciones (0→1)
    - Cuándo ocurrieron
    - Tasa de crecimiento por período
    - Patrones espaciales de expansión
    """
    
    def __init__(self, validate_consistency: bool = True):
        """Inicializa el extractor.

        Argumentos:
            validate_consistency: Si ``True``, valida que las transiciones sean
                unidireccionales (0→1 solamente) y reporta regresiones (1→0).
        """
        self.validate_consistency = validate_consistency
        self.transitions_dataset: Optional[HistoricalTransitionsDataset] = None
    
    def extract_transitions(
        self,
        years_data: Dict[int, np.ndarray],
        allow_gaps: bool = True
    ) -> HistoricalTransitionsDataset:
        """Extrae transiciones a partir de un diccionario de mapas históricos.

        Argumentos:
            years_data: Diccionario ``{año: grid_urbano}`` (binario 0/1).
            allow_gaps: Si ``True``, permite saltos entre años (p. ej. 2014→2016
                sin 2015). En la práctica, el extractor siempre toma pares
                consecutivos del ordenado; el parámetro se conserva para
                compatibilidad/claridad.

        Retorna:
            Dataset con todas las transiciones identificadas.
        """
        sorted_years = sorted(years_data.keys())
        
        logger.info(f"Extrayendo transiciones de {len(sorted_years)} años: {sorted_years[0]}-{sorted_years[-1]}")
        
        # Validar consistencia de shapes
        grid_shapes = {year: grid.shape for year, grid in years_data.items()}
        unique_shapes = set(grid_shapes.values())
        
        if len(unique_shapes) > 1:
            logger.warning(f"Múltiples shapes detectados: {unique_shapes}")
            # Tomar el más común
            from collections import Counter
            grid_shape = Counter(grid_shapes.values()).most_common(1)[0][0]
        else:
            grid_shape = list(unique_shapes)[0]
        
        transitions = []
        
        for i in range(len(sorted_years) - 1):
            year_t0 = sorted_years[i]
            year_t1 = sorted_years[i + 1]
            
            grid_t0 = years_data[year_t0]
            grid_t1 = years_data[year_t1]
            
            # Validar shapes
            if grid_t0.shape != grid_t1.shape:
                logger.warning(f"Shape mismatch {year_t0}→{year_t1}: {grid_t0.shape} vs {grid_t1.shape}")
                continue
            
            # Extraer transición
            transition_period = self._extract_single_transition(
                year_t0, year_t1, grid_t0, grid_t1
            )
            
            transitions.append(transition_period)
            
            logger.info(
                f"  {year_t0}→{year_t1}: {transition_period.n_transitions:,} transiciones "
                f"({transition_period.growth_rate:.2f}% crecimiento)"
            )
        
        # Crear dataset
        self.transitions_dataset = HistoricalTransitionsDataset(
            transitions=transitions,
            years_available=sorted_years,
            grid_shape=grid_shape
        )
        
        logger.info(f"\n✅ Extracción completa: {self.transitions_dataset.total_transitions:,} transiciones totales")
        
        return self.transitions_dataset
    
    def _extract_single_transition(
        self,
        year_t0: int,
        year_t1: int,
        grid_t0: np.ndarray,
        grid_t1: np.ndarray
    ) -> TransitionPeriod:
        """Extrae la transición (0→1) entre dos grids consecutivos.

        Argumentos:
            year_t0: Año del grid inicial.
            year_t1: Año del grid final.
            grid_t0: Grid binario en $t_0$ (0=no urbano, 1=urbano).
            grid_t1: Grid binario en $t_1$ (0=no urbano, 1=urbano).

        Retorna:
            Objeto :class:`TransitionPeriod` con máscara de transiciones y
            métricas agregadas.
        """
        # Identificar transiciones: 0 en t0 → 1 en t1
        transitions = (grid_t0 == 0) & (grid_t1 == 1)
        
        # Validar consistencia si está habilitado
        if self.validate_consistency:
            # Detectar regresiones (1→0): urbanización que desaparece
            regressions = (grid_t0 == 1) & (grid_t1 == 0)
            n_regressions = np.sum(regressions)
            
            if n_regressions > 0:
                logger.warning(
                    f"  ⚠️ {n_regressions:,} regresiones detectadas ({year_t0}→{year_t1}). "
                    f"Esto puede indicar error en clasificación."
                )
        
        n_urban_t0 = int(np.sum(grid_t0 == 1))
        n_urban_t1 = int(np.sum(grid_t1 == 1))
        n_transitions = int(np.sum(transitions))
        
        return TransitionPeriod(
            year_from=year_t0,
            year_to=year_t1,
            time_span=year_t1 - year_t0,
            grid_t0=grid_t0,
            grid_t1=grid_t1,
            transitions=transitions,
            n_transitions=n_transitions,
            n_urban_t0=n_urban_t0,
            n_urban_t1=n_urban_t1,
            growth_rate=0  # Se calculará en __post_init__
        )
    
    def get_aggregated_transitions(
        self,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None
    ) -> np.ndarray:
        """
        Obtener máscara agregada de todas las transiciones en rango de años.
        
        Parameters:
        -----------
        min_year, max_year : int, optional
            Rango de años a incluir
            
        Returns:
        --------
        np.ndarray
            Máscara binaria con todas las transiciones agregadas
        """
        if self.transitions_dataset is None:
            raise ValueError("Primero debe ejecutar extract_transitions()")
        
        aggregated = np.zeros(self.transitions_dataset.grid_shape, dtype=bool)
        
        for t in self.transitions_dataset.transitions:
            if min_year is not None and t.year_from < min_year:
                continue
            if max_year is not None and t.year_to > max_year:
                continue
            
            aggregated |= t.transitions
        
        return aggregated
    
    def analyze_spatial_patterns(self) -> Dict:
        """
        Analizar patrones espaciales de las transiciones.
        
        Returns:
        --------
        Dict
            Estadísticas de patrones espaciales:
            - Fragmentación
            - Compacidad
            - Dirección de expansión
        """
        if self.transitions_dataset is None:
            raise ValueError("Primero debe ejecutar extract_transitions()")
        
        from scipy.ndimage import label
        
        patterns = {}
        
        # Analizar cada período
        for t in self.transitions_dataset.transitions:
            # Componentes conectados de transiciones
            labeled, n_components = label(t.transitions)
            
            # Tamaño promedio de cluster de transición
            if n_components > 0:
                component_sizes = [
                    np.sum(labeled == i) for i in range(1, n_components + 1)
                ]
                avg_size = np.mean(component_sizes)
                max_size = np.max(component_sizes)
            else:
                avg_size = 0
                max_size = 0
            
            patterns[f"{t.year_from}_{t.year_to}"] = {
                'n_clusters': n_components,
                'avg_cluster_size': avg_size,
                'max_cluster_size': max_size,
                'fragmentation': n_components / max(t.n_transitions, 1)
            }
        
        return patterns


def load_historical_maps(
    data_dir: Path,
    file_pattern: str = "svm_2_classes.npy",
    year_pattern: str = "year_"
) -> Dict[int, np.ndarray]:
    """
    Cargar mapas históricos desde directorio de procesamiento batch.
    
    Parameters:
    -----------
    data_dir : Path
        Directorio raíz (ej: batch_processing_20251102_160628)
    file_pattern : str
        Patrón de nombre de archivo a buscar
    year_pattern : str
        Patrón para identificar subdirectorios de años
        
    Returns:
    --------
    Dict[int, np.ndarray]
        Diccionario {año: grid_urbano}
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Directorio no encontrado: {data_dir}")
    
    years_data = {}
    
    # Buscar subdirectorios de años
    year_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and year_pattern in d.name])
    
    logger.info(f"Buscando mapas en {data_dir}")
    logger.info(f"  Encontrados {len(year_dirs)} directorios de años")
    
    for year_dir in year_dirs:
        # Extraer año del nombre del directorio
        year_str = year_dir.name.replace(year_pattern, '')
        
        try:
            year = int(year_str)
        except ValueError:
            logger.warning(f"  No se pudo extraer año de: {year_dir.name}")
            continue
        
        # Buscar archivo de predicción
        prediction_files = list(year_dir.rglob(file_pattern))
        
        if len(prediction_files) == 0:
            logger.warning(f"  {year}: No se encontró {file_pattern}")
            continue
        
        if len(prediction_files) > 1:
            logger.warning(f"  {year}: Múltiples archivos encontrados, usando primero")
        
        # Cargar grid
        grid_path = prediction_files[0]
        try:
            grid = np.load(grid_path)
            years_data[year] = grid
            logger.info(f"  ✅ {year}: {grid.shape} - {np.sum(grid==1):,} píxeles urbanos")
        except Exception as e:
            logger.error(f"  ❌ {year}: Error cargando {grid_path}: {e}")
    
    logger.info(f"\n✅ Cargados {len(years_data)} mapas históricos")
    
    return years_data


def extract_transitions_from_directory(
    data_dir: Path,
    output_path: Optional[Path] = None,
    file_pattern: str = "svm_2_classes.npy"
) -> HistoricalTransitionsDataset:
    """
    Pipeline completo: cargar mapas y extraer transiciones.
    
    Parameters:
    -----------
    data_dir : Path
        Directorio con procesamiento batch
    output_path : Path, optional
        Donde guardar el dataset extraído
    file_pattern : str
        Patrón de archivo a buscar
        
    Returns:
    --------
    HistoricalTransitionsDataset
        Dataset con transiciones extraídas
    """
    # Cargar mapas
    years_data = load_historical_maps(data_dir, file_pattern=file_pattern)
    
    if len(years_data) < 2:
        raise ValueError(f"Se necesitan al menos 2 años, encontrados {len(years_data)}")
    
    # Extraer transiciones
    extractor = HistoricalTransitionExtractor(validate_consistency=True)
    dataset = extractor.extract_transitions(years_data)
    
    # Guardar si se especifica output
    if output_path is not None:
        dataset.save(output_path)
        
        # Guardar también CSV resumen
        summary_df = dataset.get_summary_dataframe()
        csv_path = output_path.with_suffix('.csv')
        summary_df.to_csv(csv_path, index=False)
        logger.info(f"Resumen guardado: {csv_path}")
    
    return dataset


if __name__ == "__main__":
    """Script CLI para extracción rápida."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extraer transiciones urbanas del histórico de mapas clasificados"
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        required=True,
        help='Directorio con procesamiento batch (ej: batch_processing_20251102_160628)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('analysis/historical_transitions.pkl'),
        help='Archivo de salida para dataset'
    )
    parser.add_argument(
        '--file-pattern',
        type=str,
        default='svm_2_classes.npy',
        help='Patrón de archivo a buscar'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Modo verbose'
    )
    
    args = parser.parse_args()
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Ejecutar extracción
    dataset = extract_transitions_from_directory(
        data_dir=args.data_dir,
        output_path=args.output,
        file_pattern=args.file_pattern
    )
    
    # Mostrar resumen
    print("\n" + "="*60)
    print("RESUMEN DE TRANSICIONES HISTÓRICAS")
    print("="*60)
    print(dataset.get_summary_dataframe().to_string())
    print("\n" + "="*60)
    print(f"Total transiciones: {dataset.total_transitions:,}")
    print(f"Años analizados: {len(dataset.years_available)}")
    print(f"Rango: {min(dataset.years_available)}-{max(dataset.years_available)}")
    print("="*60)
