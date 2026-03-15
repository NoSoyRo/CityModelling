"""Cálculo de Weight of Evidence (WoE) para transiciones del Autómata Celular.

Este módulo implementa el cálculo de **Weight of Evidence (WoE)** para cuantificar
la evidencia espacial asociada a la **transición urbana** (celdas que cambian de
no‑urbanizadas a urbanizadas) en el sistema de simulación de crecimiento urbano
de Querétaro (1984–2020).

En el pipeline del proyecto, WoE pertenece a la etapa:

1) **Construcción de variables espaciales** (p. ej. ``distance_urban``,
   ``neighbor_density_3x3/5x5/7x7``, ``local_fragmentation``, ``nearest_cluster_size``,
   ``urban_gradient``).
2) **Entrenamiento WoE**: discretización por bins + cálculo por bin.
3) **Aplicación WoE**: transformar capas/raster nuevos a “scores WoE” para alimentar
   reglas del Autómata Celular.

Fundamento teórico (por bin $b$):

.. math::

   WoE(b) = \ln\left(\frac{P(X \in b\mid \text{transición})}{P(X \in b\mid \text{no transición})}\right)

Además se calcula el **Information Value (IV)** como medida agregada de capacidad
predictiva de la variable.

Entradas típicas:
    - Arreglos 1D con valores muestreados de variables espaciales.
    - Arreglo 1D binario (0/1) que indica transición urbana.

Salidas típicas:
    - :class:`WoEResult` por variable (bins, WoE por bin, IV total, tabla de stats).
    - Transformaciones WoE aplicables a grillas/matrices N‑D.

Referencias:
    - Bonham-Carter, G.F. (1994). *Geographic Information Systems for Geoscientists*.
    - Agterberg, F.P. (2014). *Geomathematics: Theoretical Foundations...*.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import warnings
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class WoEResult:
    """Resultado del cálculo de Weight of Evidence (WoE) para una variable.

    En el contexto del modelo de crecimiento urbano, un :class:`WoEResult`
    encapsula la evidencia empírica que una variable espacial (p. ej.
    ``distance_urban`` o ``neighbor_density_3x3``) aporta para explicar la
    **transición urbana**.

    Este objeto es el artefacto principal producido por
    :meth:`WoECalculator.calculate_woe`. Posteriormente se usa para transformar
    nuevas capas/grillas a **scores WoE** que alimentan las reglas del Autómata
    Celular.

    Atributos:
        variable_name: Nombre de la variable.
        bins: Bordes de bins usados en la discretización.
        woe_values: Valor WoE por bin.
        iv_total: Information Value (IV) total, una medida de fuerza predictiva.
        n_positive_samples: Número de muestras positivas (transición/urbanización).
        n_negative_samples: Número de muestras negativas (no transición).
        bin_stats: Tabla por bin con conteos, tasas y WoE.

    Ejemplo:
        >>> resultado.iv_total  # doctest: +SKIP
        0.213
    """
    variable_name: str
    bins: np.ndarray
    woe_values: np.ndarray
    iv_total: float  # Information Value total
    n_positive_samples: int
    n_negative_samples: int
    bin_stats: pd.DataFrame


class WoECalculator:
    """Calculadora de Weight of Evidence (WoE) para variables espaciales.

    Esta clase estima WoE por bins para una variable explicativa y un objetivo
    binario de **transición urbana**.

    Fórmula operativa (por bin $i$):

    .. math::

       WoE_i = \ln\left(\frac{P(Y=1\mid X\in i)}{P(Y=0\mid X\in i)}\right)

    donde los conteos se aproximan con frecuencias muestrales:

    - $N_{1i}$: casos positivos (urbanización/transición) en el bin $i$.
    - $N_{0i}$: casos negativos (no transición) en el bin $i$.
    - $N_1, N_0$: totales de positivos y negativos.

    Notas:
        - Este módulo discretiza variables continuas; si la variable ya es
          categórica, conviene discretizar externamente o adaptar el binning.
        - Se usa suavizado mínimo (clipping a 1) para evitar divisiones por cero.
        - Se calcula IV para evaluar fuerza predictiva.
    """
    
    def __init__(self, min_bin_size: int = 50, max_bins: int = 10):
        """Inicializa la calculadora WoE.

        La calculadora discretiza cada variable espacial en bins y calcula WoE
        por bin, además del Information Value (IV) total.

        Argumentos:
            min_bin_size: Mínimo de muestras por bin para reducir varianza y
                evitar estimaciones WoE inestables.
            max_bins: Máximo número de bins permitidos.

        Ejemplo:
            >>> calc = WoECalculator(min_bin_size=200, max_bins=12)  # doctest: +SKIP
        """
        self.min_bin_size = min_bin_size
        self.max_bins = max_bins
        self.woe_results: Dict[str, WoEResult] = {}
        
    def calculate_woe(
        self,
        variable: np.ndarray,
        target: np.ndarray,
        variable_name: str,
        binning_method: str = 'quantile'
    ) -> WoEResult:
        """Calcula Weight of Evidence (WoE) para una variable explicativa.

        En este proyecto, el objetivo ``target`` representa la **transición
        urbana**:

        - ``1`` si la celda cambia de no‑urbanizada a urbanizada en un periodo.
        - ``0`` en caso contrario.

        El WoE se calcula por bin $b$:

        .. math::

           WoE(b) = \ln\left(\frac{P(X \in b\mid \text{transición})}{P(X \in b\mid \text{no transición})}\right)

        Argumentos:
            variable: Arreglo 1D con valores de una variable espacial para
                celdas muestreadas (p. ej. ``distance_urban`` o ``urban_gradient``).
            target: Arreglo 1D binario (0/1) alineado con ``variable``.
            variable_name: Nombre con el que se almacena el resultado.
            binning_method: Método de discretización. Valores soportados:
                ``"quantile"``, ``"equal_width"``, ``"kmeans"``.

        Retorna:
            Un :class:`WoEResult` con bins, WoE por bin, estadísticas por bin y
            el Information Value (IV) total.

        Lanza:
            ValueError: Si ``variable`` y ``target`` difieren en longitud, si
                ``target`` no es binario, si hay datos insuficientes tras remover
                NaN, o si falla la discretización.

        Ejemplo:
            >>> calc = WoECalculator()  # doctest: +SKIP
            >>> res = calc.calculate_woe(var, y, "distance_urban")  # doctest: +SKIP
            >>> res.iv_total  # doctest: +SKIP
            0.12
        """
        # Validación de entrada
        if len(variable) != len(target):
            raise ValueError(f"Longitudes inconsistentes: variable={len(variable)}, target={len(target)}")
            
        if not np.all(np.isin(target, [0, 1])):
            raise ValueError("Target debe ser binario (0,1)")
            
        # Remover valores faltantes
        mask = ~(np.isnan(variable) | np.isnan(target))
        variable_clean = variable[mask]
        target_clean = target[mask]
        
        if len(variable_clean) < self.min_bin_size * 2:
            raise ValueError(f"Datos insuficientes: {len(variable_clean)} < {self.min_bin_size * 2}")
            
        logger.info(f"Calculando WoE para {variable_name}: {len(variable_clean)} observaciones")
        
        # Crear bins
        bins = self._create_bins(variable_clean, binning_method)
        
        # Asignar observaciones a bins
        bin_indices = np.digitize(variable_clean, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(bins) - 2)  # Evitar índices fuera de rango
        
        # Calcular estadísticas por bin
        bin_stats = []
        woe_values = []
        
        n_positive_total = np.sum(target_clean == 1)
        n_negative_total = np.sum(target_clean == 0)
        
        for i in range(len(bins) - 1):
            mask_bin = bin_indices == i
            
            if np.sum(mask_bin) < self.min_bin_size:
                warnings.warn(f"Bin {i} tiene pocos datos: {np.sum(mask_bin)}")
                
            n_positive_bin = np.sum(target_clean[mask_bin] == 1)
            n_negative_bin = np.sum(target_clean[mask_bin] == 0)
            
            # Suavizado para evitar división por cero
            n_positive_bin = max(n_positive_bin, 1)
            n_negative_bin = max(n_negative_bin, 1)
            
            # Cálculo WoE
            pos_rate = n_positive_bin / n_positive_total
            neg_rate = n_negative_bin / n_negative_total
            
            woe = np.log(pos_rate / neg_rate)
            woe_values.append(woe)
            
            # Estadísticas del bin
            bin_stats.append({
                'bin_id': i,
                'bin_min': bins[i],
                'bin_max': bins[i + 1],
                'n_total': np.sum(mask_bin),
                'n_positive': n_positive_bin,
                'n_negative': n_negative_bin,
                'positive_rate': n_positive_bin / np.sum(mask_bin),
                'woe': woe
            })
            
        bin_stats_df = pd.DataFrame(bin_stats)
        woe_array = np.array(woe_values)
        
        # Calcular Information Value (IV) - métrica de calidad
        iv_total = self._calculate_information_value(bin_stats_df)
        
        # Crear resultado
        result = WoEResult(
            variable_name=variable_name,
            bins=bins,
            woe_values=woe_array,
            iv_total=iv_total,
            n_positive_samples=n_positive_total,
            n_negative_samples=n_negative_total,
            bin_stats=bin_stats_df
        )
        
        # Almacenar resultado
        self.woe_results[variable_name] = result
        
        logger.info(f"WoE calculado: IV={iv_total:.4f}, bins={len(bins)-1}")
        
        return result
        
    def _create_bins(self, variable: np.ndarray, method: str) -> np.ndarray:
        """Crea bins para discretizar una variable espacial continua.

        Helper interno usado por :meth:`calculate_woe`.

        Argumentos:
            variable: Arreglo 1D con valores (sin NaN).
            method: Estrategia de discretización. Valores soportados:
                - ``"quantile"``: bins por cuantiles.
                - ``"equal_width"``: bins de ancho uniforme entre min/max.
                - ``"kmeans"``: límites derivados de centros de KMeans.

        Retorna:
            Arreglo con bordes de bins, longitud ``n_bins + 1``.

        Lanza:
            ValueError: Si el método es desconocido o los bins son inválidos.

        Ejemplo:
            >>> calc = WoECalculator()  # doctest: +SKIP
            >>> bins = calc._create_bins(np.array([0.0, 1.0, 2.0]), "equal_width")  # doctest: +SKIP
        """
        if method == 'quantile':
            # Bins basados en quantiles para distribución uniforme
            n_bins = min(self.max_bins, len(variable) // self.min_bin_size)
            quantiles = np.linspace(0, 1, n_bins + 1)
            bins = np.quantile(variable, quantiles)
            
        elif method == 'equal_width':
            # Bins de ancho igual
            n_bins = min(self.max_bins, len(variable) // self.min_bin_size)
            bins = np.linspace(variable.min(), variable.max(), n_bins + 1)
            
        elif method == 'kmeans':
            # Bins basados en K-means clustering
            from sklearn.cluster import KMeans
            n_bins = min(self.max_bins, len(variable) // self.min_bin_size)
            kmeans = KMeans(n_clusters=n_bins, random_state=42, n_init=10)
            kmeans.fit(variable.reshape(-1, 1))
            centers = sorted(kmeans.cluster_centers_.flatten())
            
            # Crear bins desde centros
            bins = [variable.min()]
            for i in range(len(centers) - 1):
                bins.append((centers[i] + centers[i + 1]) / 2)
            bins.append(variable.max())
            bins = np.array(bins)
            
        else:
            raise ValueError(f"Método de binning desconocido: {method}")
            
        # Garantizar bins únicos y ordenados
        bins = np.unique(bins)
        
        if len(bins) < 3:
            raise ValueError(f"Muy pocos bins: {len(bins)}. Se requieren al menos 2 bins.")
            
        return bins
        
    def _calculate_information_value(self, bin_stats: pd.DataFrame) -> float:
        """Calcula el Information Value (IV) total de una variable.

        IV mide la fuerza predictiva de una variable (en el sentido WoE). La
        regla práctica de interpretación suele ser:

        - IV < 0.02: sin valor predictivo
        - 0.02 ≤ IV < 0.10: débil
        - 0.10 ≤ IV < 0.30: medio
        - 0.30 ≤ IV < 0.50: fuerte
        - IV ≥ 0.50: muy fuerte (posible sobreajuste)

        Argumentos:
            bin_stats: DataFrame con columnas ``n_positive``, ``n_negative`` y
                ``woe`` (mínimo).

        Retorna:
            Valor IV total de la variable.

        Lanza:
            KeyError: Si faltan columnas requeridas.

        Ejemplo:
            >>> # iv = calc._calculate_information_value(df)  # doctest: +SKIP
        """
        total_pos = bin_stats['n_positive'].sum()
        total_neg = bin_stats['n_negative'].sum()
        
        iv = 0.0
        for _, row in bin_stats.iterrows():
            pos_rate = row['n_positive'] / total_pos
            neg_rate = row['n_negative'] / total_neg
            
            if pos_rate > 0 and neg_rate > 0:
                iv += (pos_rate - neg_rate) * row['woe']
                
        return iv
        
    def apply_woe_transform(
        self,
        variable: np.ndarray,
        variable_name: str
    ) -> np.ndarray:
        """Aplica la transformación WoE a una variable usando bins precomputados.

        Esta función es crucial para el Autómata Celular: convierte variables
        originales (distancias, densidades, fragmentación, etc.) a **scores WoE**
        utilizados por las reglas de transición.

        Argumentos:
            variable: Arreglo de valores crudos a transformar. Puede ser 1D o N‑D.
                Internamente usa :func:`numpy.digitize` para asignar bins.
            variable_name: Nombre de la variable para la cual ya existe WoE.

        Retorna:
            Arreglo con la misma forma que ``variable`` que contiene scores WoE.

        Lanza:
            ValueError: Si no se ha calculado WoE para ``variable_name``.

        Ejemplo:
            >>> calc = WoECalculator()  # doctest: +SKIP
            >>> w = calc.apply_woe_transform(np.array([10.0, 20.0]), "distance_urban")  # doctest: +SKIP
        """
        if variable_name not in self.woe_results:
            raise ValueError(f"WoE no calculado para {variable_name}. Ejecutar calculate_woe() primero.")
            
        result = self.woe_results[variable_name]
        bins = result.bins
        woe_values = result.woe_values
        
        # Asignar bins
        bin_indices = np.digitize(variable, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(woe_values) - 1)
        
        # Aplicar transformación WoE
        woe_transformed = woe_values[bin_indices]
        
        return woe_transformed
        
    def get_urban_probability(
        self,
        **woe_scores: float
    ) -> float:
        """Calcula probabilidad de urbanización combinando múltiples evidencias WoE.

        En su forma más simple, este método suma los scores WoE y aplica una
        función logística para obtener una probabilidad en $[0,1]$.

        Nota:
            En modelos WoE-CA más completos se suele incluir un término base
            (prior), ponderaciones por IV y/o interacción con vecindarios.

        Argumentos:
            **woe_scores: Scores WoE para una celda/ubicación. Las llaves son
                nombres de variables y los valores son números (WoE).

        Retorna:
            Probabilidad de urbanización en el rango ``[0, 1]``.

        Ejemplo:
            >>> calc = WoECalculator()  # doctest: +SKIP
            >>> p = calc.get_urban_probability(distance_urban=-0.3, neighbor_density_3x3=0.8)  # doctest: +SKIP
        """
        # Suma ponderada de evidencias
        total_woe = sum(woe_scores.values())
        
        # Convertir a probabilidad usando función logística
        probability = 1 / (1 + np.exp(-total_woe))
        
        return probability
        
    def summary_report(self) -> str:
        """Genera un reporte legible con el resumen de variables WoE calculadas.

        El reporte está pensado para inspección rápida durante corridas de
        investigación. Resume número de bins, IV y tamaños muestrales.

        Retorna:
            Cadena multilínea con el resumen.

        Ejemplo:
            >>> calc = WoECalculator()  # doctest: +SKIP
            >>> print(calc.summary_report())  # doctest: +SKIP
        """
        if not self.woe_results:
            return "No se han calculado WoE para ninguna variable."

        report = "\n=== REPORTE RESUMEN WEIGHT OF EVIDENCE (WoE) ===\n\n"
        
        for name, result in self.woe_results.items():
            iv_strength = self._classify_iv_strength(result.iv_total)
            
            report += f"Variable: {name}\n"
            report += f"  Information Value (IV): {result.iv_total:.4f} ({iv_strength})\n"
            report += f"  Bins: {len(result.bins) - 1}\n"
            report += f"  Rango WoE: [{result.woe_values.min():.3f}, {result.woe_values.max():.3f}]\n"
            report += f"  Muestras: {result.n_positive_samples} positivas, {result.n_negative_samples} negativas\n\n"
            
        return report
        
    def _classify_iv_strength(self, iv: float) -> str:
        """Clasifica la fuerza predictiva según el Information Value (IV).

        Argumentos:
            iv: Information Value total.

        Retorna:
            Etiqueta cualitativa ("Débil", "Medio", etc.).

        Ejemplo:
            >>> calc = WoECalculator()  # doctest: +SKIP
            >>> calc._classify_iv_strength(0.12)  # doctest: +SKIP
            'Medio'
        """
        if iv < 0.02:
            return "Sin valor predictivo"
        elif iv < 0.1:
            return "Débil"
        elif iv < 0.3:
            return "Medio"
        elif iv < 0.5:
            return "Fuerte"
        else:
            return "Muy fuerte"
            
    def save_results(self, filepath: Union[str, Path]) -> None:
        """Guarda resultados WoE (``self.woe_results``) en un archivo pickle.

        Importante:
            Este método serializa un diccionario ``Dict[str, WoEResult]``.
            Para que el pickle sea portable, conviene cargarlo en un entorno
            donde exista el paquete ``tesis_ac``.

        Argumentos:
            filepath: Ruta de salida (str o :class:`pathlib.Path`).

        Lanza:
            OSError: Si no se puede crear el directorio o escribir el archivo.

        Ejemplo:
            >>> calc.save_results('reports/woe_results.pkl')  # doctest: +SKIP
        """
        import pickle
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.woe_results, f)
            
        logger.info(f"Resultados WoE guardados en {filepath}")
        
    def load_results(self, filepath: Union[str, Path]) -> None:
        """Carga resultados WoE previamente guardados (pickle) a ``self.woe_results``.

        Argumentos:
            filepath: Ruta del archivo pickle.

        Lanza:
            FileNotFoundError: Si el archivo no existe.
            pickle.UnpicklingError: Si el archivo no corresponde a un pickle válido.

        Ejemplo:
            >>> calc.load_results('reports/woe_results.pkl')  # doctest: +SKIP
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            self.woe_results = pickle.load(f)
            
        logger.info(f"Resultados WoE cargados desde {filepath}")


def calculate_spatial_woe_variables(
    grid: np.ndarray,
    urban_history: np.ndarray,
    features: Dict[str, np.ndarray],
    calculator: Optional[WoECalculator] = None
) -> Dict[str, WoEResult]:
    """Calcula WoE para múltiples variables espaciales del Autómata Celular.

    Esta función actúa como un *helper* de alto nivel: toma grillas/capas
    espaciales y un objetivo binario de transición urbana, y regresa un
    :class:`WoEResult` por variable.

    Argumentos:
        grid: Grid clasificado actual (urbano/no-urbano). Se mantiene por
            compatibilidad con el pipeline, aunque aquí no se usa directamente.
        urban_history: Grid/array binario objetivo (1=transición urbana).
        features: Diccionario de variables explicativas (cada valor puede ser
            un array 2D/3D). Ejemplos comunes del proyecto:

            - ``distance_urban``
            - ``neighbor_density_3x3`` / ``neighbor_density_5x5`` / ``neighbor_density_7x7``
            - ``local_fragmentation``
            - ``nearest_cluster_size``
            - ``urban_gradient``

        calculator: Instancia reutilizable de :class:`WoECalculator`. Si es None,
            se crea una con parámetros moderados.

    Retorna:
        Diccionario ``Dict[str, WoEResult]`` con resultados WoE por variable.

    Lanza:
        ValueError: Puede propagarse desde :meth:`WoECalculator.calculate_woe`.

    Ejemplo:
        >>> calc = WoECalculator()  # doctest: +SKIP
        >>> res = calculate_spatial_woe_variables(grid, y, feats, calc)  # doctest: +SKIP
        >>> list(res.keys())[:2]  # doctest: +SKIP
        ['distance_urban', 'neighbor_density_3x3']
    """
    if calculator is None:
        calculator = WoECalculator(min_bin_size=100, max_bins=8)
        
    results = {}
    
    logger.info(f"Calculando WoE para {len(features)} variables espaciales")
    
    for var_name, var_data in features.items():
        try:
            # Aplanar datos espaciales para análisis
            var_flat = var_data.flatten()
            target_flat = urban_history.flatten()
            
            # Filtrar valores válidos
            mask = ~(np.isnan(var_flat) | np.isnan(target_flat))
            var_clean = var_flat[mask]
            target_clean = target_flat[mask]
            
            if len(var_clean) < 200:
                warnings.warn(f"Pocos datos para {var_name}: {len(var_clean)}")
                continue
                
            # Calcular WoE
            result = calculator.calculate_woe(
                variable=var_clean,
                target=target_clean,
                variable_name=var_name,
                binning_method='quantile'
            )
            
            results[var_name] = result
            
            logger.info(f"WoE {var_name}: IV={result.iv_total:.4f}")
            
        except Exception as e:
            logger.error(f"Error calculando WoE para {var_name}: {e}")
            continue
            
    logger.info(f"WoE calculado exitosamente para {len(results)} variables")
    
    return results