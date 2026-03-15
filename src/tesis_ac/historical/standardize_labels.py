"""
Estandarización de labels en mapas de clasificación binaria.

Este script detecta y corrige mapas donde las etiquetas están invertidas,
asegurando que TODOS los mapas sigan la convención: 1=urbano, 0=no-urbano.

Estrategia:
1. Para cada mapa, calcular qué label (0 o 1) es minoritario
2. ASUMIR que el centro de la imagen tiene más urbanización
3. Verificar si el centro tiene mayoría de 1s o 0s
4. Si el centro tiene mayoría de 0s, invertir el mapa
"""

import numpy as np
from pathlib import Path
import argparse
import logging
from typing import Dict, Tuple
import shutil

logger = logging.getLogger(__name__)


def analyze_map_center(grid: np.ndarray, center_ratio: float = 0.3) -> Dict:
    """Analiza la región central de un mapa binario.

    Argumentos:
        grid: Mapa binario.
        center_ratio: Proporción del lado (aprox.) que se toma como región
            central (0.3 = 30% central).

    Retorna:
        Diccionario con estadísticas del centro (conteos, proporciones y label
        dominante).
    """
    h, w = grid.shape
    
    # Definir región central
    center_h_start = int(h * (0.5 - center_ratio/2))
    center_h_end = int(h * (0.5 + center_ratio/2))
    center_w_start = int(w * (0.5 - center_ratio/2))
    center_w_end = int(w * (0.5 + center_ratio/2))
    
    center = grid[center_h_start:center_h_end, center_w_start:center_w_end]
    
    n_ones_center = np.sum(center == 1)
    n_zeros_center = np.sum(center == 0)
    total_center = center.size
    
    return {
        'n_ones': n_ones_center,
        'n_zeros': n_zeros_center,
        'total': total_center,
        'ones_ratio': n_ones_center / total_center,
        'zeros_ratio': n_zeros_center / total_center,
        'dominant_label': 1 if n_ones_center > n_zeros_center else 0
    }


def should_invert_map(grid: np.ndarray, method: str = 'center') -> Tuple[bool, str]:
    """Determina si un mapa binario debe invertirse (0↔1).

    Métodos disponibles:
        - ``'center'``: asume que el centro de la imagen debería ser urbano (1).
        - ``'minority'``: asume que urbano es minoritario globalmente.

    Argumentos:
        grid: Mapa binario.
        method: Método de detección.

    Retorna:
        Tupla ``(debe_invertir, razon)``.
    """
    if method == 'center':
        center_stats = analyze_map_center(grid, center_ratio=0.3)
        
        # Si el centro tiene mayoría de 0s, probablemente está invertido
        if center_stats['dominant_label'] == 0:
            return True, f"Centro tiene {center_stats['zeros_ratio']:.1%} de 0s (esperamos 1s urbanos)"
        else:
            return False, f"Centro tiene {center_stats['ones_ratio']:.1%} de 1s (correcto)"
    
    elif method == 'minority':
        n_ones = np.sum(grid == 1)
        n_zeros = np.sum(grid == 0)
        total = grid.size
        
        ones_ratio = n_ones / total
        
        # Si 1s son mayoría global (>50%), probablemente está invertido
        # Porque urbano debería ser minoritario
        if ones_ratio > 0.50:
            return True, f"1s son mayoría global ({ones_ratio:.1%}), esperamos que urbano sea minoritario"
        else:
            return False, f"1s son minoría global ({ones_ratio:.1%}), correcto"
    
    return False, "Método desconocido"


def standardize_all_maps(
    data_dir: Path,
    file_pattern: str = "svm_2_classes.npy",
    method: str = 'center',
    apply_corrections: bool = False,
    backup: bool = True
) -> Dict:
    """Estandariza etiquetas de mapas binarios en un directorio.

    En modo *dry-run* (``apply_corrections=False``) solo reporta qué mapas
    parecen estar invertidos. En modo aplicación, opcionalmente crea respaldo
    y sobrescribe el archivo.

    Argumentos:
        data_dir: Directorio con mapas.
        file_pattern: Patrón de archivo a buscar dentro de cada año.
        method: Método de detección (``'center'`` o ``'minority'``).
        apply_corrections: Si ``True``, aplica correcciones sobre los archivos.
        backup: Si ``True``, crea un ``.backup`` antes de modificar.

    Retorna:
        Diccionario con resultados agregados y listas de mapas corregidos/omitidos.
    """
    data_dir = Path(data_dir)
    
    # Buscar todos los mapas
    year_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and 'year_' in d.name])
    
    results = {
        'total_maps': 0,
        'inverted_maps': 0,
        'correct_maps': 0,
        'corrected': [],
        'skipped': []
    }
    
    logger.info(f"{'='*80}")
    logger.info(f"ESTANDARIZACIÓN DE MAPAS - Método: {method}")
    logger.info(f"{'='*80}")
    logger.info(f"Directorio: {data_dir}")
    logger.info(f"Modo: {'APLICAR CORRECCIONES' if apply_corrections else 'DRY-RUN (solo análisis)'}")
    logger.info(f"")
    
    for year_dir in year_dirs:
        year_str = year_dir.name.replace('year_', '')
        
        try:
            year = int(year_str)
        except ValueError:
            continue
        
        # Buscar archivo
        map_files = list(year_dir.rglob(file_pattern))
        
        if len(map_files) == 0:
            logger.warning(f"  ⚠️ {year}: No se encontró {file_pattern}")
            continue
        
        map_path = map_files[0]
        
        # Cargar mapa
        grid = np.load(map_path)
        results['total_maps'] += 1
        
        # Analizar
        should_invert, reason = should_invert_map(grid, method=method)
        
        if should_invert:
            results['inverted_maps'] += 1
            
            if apply_corrections:
                # Hacer backup
                if backup:
                    backup_path = map_path.with_suffix('.npy.backup')
                    if not backup_path.exists():
                        shutil.copy2(map_path, backup_path)
                
                # Invertir
                grid_corrected = 1 - grid
                
                # Guardar
                np.save(map_path, grid_corrected)
                
                results['corrected'].append({
                    'year': year,
                    'path': str(map_path),
                    'reason': reason
                })
                
                logger.info(f"  ✅ {year}: INVERTIDO - {reason}")
            else:
                logger.info(f"  🔍 {year}: NECESITA INVERSIÓN - {reason}")
        else:
            results['correct_maps'] += 1
            results['skipped'].append({
                'year': year,
                'reason': reason
            })
            logger.info(f"  ✓ {year}: OK - {reason}")
    
    # Resumen
    logger.info(f"\n{'='*80}")
    logger.info(f"RESUMEN")
    logger.info(f"{'='*80}")
    logger.info(f"Total mapas analizados: {results['total_maps']}")
    logger.info(f"Mapas correctos: {results['correct_maps']}")
    logger.info(f"Mapas invertidos detectados: {results['inverted_maps']}")
    
    if apply_corrections:
        logger.info(f"Mapas corregidos: {len(results['corrected'])}")
        logger.info(f"\n✅ CORRECCIONES APLICADAS")
    else:
        logger.info(f"\n⚠️ MODO DRY-RUN - No se aplicaron cambios")
        logger.info(f"   Ejecuta con --apply para aplicar correcciones")
    
    logger.info(f"{'='*80}")
    
    return results


def main():
    """CLI para estandarización de mapas."""
    parser = argparse.ArgumentParser(
        description="Estandarizar labels en mapas de clasificación binaria"
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        required=True,
        help='Directorio con procesamiento batch'
    )
    parser.add_argument(
        '--file-pattern',
        type=str,
        default='svm_2_classes.npy',
        help='Patrón de archivo a buscar'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['center', 'minority'],
        default='center',
        help='Método de detección (center=analizar región central, minority=urbano minoritario)'
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Aplicar correcciones (por defecto solo analiza)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='No hacer backup de archivos originales'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Modo verbose'
    )
    
    args = parser.parse_args()
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.INFO,
        format='%(message)s'
    )
    
    # Ejecutar estandarización
    results = standardize_all_maps(
        data_dir=args.data_dir,
        file_pattern=args.file_pattern,
        method=args.method,
        apply_corrections=args.apply,
        backup=not args.no_backup
    )
    
    # Guardar reporte si se aplicaron correcciones
    if args.apply and results['corrected']:
        report_path = Path('analysis/label_standardization_report.txt')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("REPORTE DE ESTANDARIZACIÓN DE LABELS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Fecha: {import_datetime_now()}\n")
            f.write(f"Método: {args.method}\n")
            f.write(f"Total mapas: {results['total_maps']}\n")
            f.write(f"Mapas corregidos: {len(results['corrected'])}\n\n")
            
            f.write("MAPAS INVERTIDOS:\n")
            f.write("-"*80 + "\n")
            for item in results['corrected']:
                f.write(f"  Año {item['year']}: {item['reason']}\n")
        
        logger.info(f"\n📄 Reporte guardado: {report_path}")


def import_datetime_now():
    """Helper para importar datetime solo cuando se necesita."""
    from datetime import datetime
    return datetime.now().isoformat()


if __name__ == "__main__":
    main()
