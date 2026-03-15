"""
Métricas comprehensivas para validación de modelos de expansión urbana.

Incluye:
- Métricas de coincidencia espacial (Precision, Recall, F1)
- Figure of Merit (FoM) - métrica crítica para cambio urbano
- Análisis detallado de transiciones
"""
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_confusion_matrix(real: np.ndarray, predicted: np.ndarray) -> Dict[str, int]:
    """Calcula la matriz de confusión binaria (TP, TN, FP, FN).

    Argumentos:
        real: Grid real (0=no urbano, 1=urbano).
        predicted: Grid predicho (0=no urbano, 1=urbano).

    Retorna:
        Diccionario con TP, TN, FP y FN.
    """
    # Asegurar que sean binarios
    real_binary = (real > 0).astype(int)
    pred_binary = (predicted > 0).astype(int)
    
    # Calcular componentes
    tp = np.sum((real_binary == 1) & (pred_binary == 1))
    tn = np.sum((real_binary == 0) & (pred_binary == 0))
    fp = np.sum((real_binary == 0) & (pred_binary == 1))
    fn = np.sum((real_binary == 1) & (pred_binary == 0))
    
    return {
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn)
    }


def calculate_spatial_metrics(confusion: Dict[str, int]) -> Dict[str, float]:
    """Calcula métricas espaciales globales a partir de una matriz de confusión.

    Argumentos:
        confusion: Matriz de confusión (TP, TN, FP, FN).

    Retorna:
        Diccionario con accuracy, precision, recall, f1_score y kappa.
    """
    tp = confusion['TP']
    tn = confusion['TN']
    fp = confusion['FP']
    fn = confusion['FN']
    
    total = tp + tn + fp + fn
    
    # Accuracy (Aproximación)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    
    # Precision (confiabilidad de predicciones urbanas)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Recall/Sensibilidad (detección de crecimiento real)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1-Score (balance entre precision y recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Kappa de Cohen (ajuste por azar)
    p0 = accuracy  # Acuerdo observado
    
    # Acuerdo esperado por azar
    p_urbano_real = (tp + fn) / total
    p_urbano_pred = (tp + fp) / total
    p_no_urbano_real = (tn + fp) / total
    p_no_urbano_pred = (tn + fn) / total
    
    pe = (p_urbano_real * p_urbano_pred) + (p_no_urbano_real * p_no_urbano_pred)
    
    kappa = (p0 - pe) / (1 - pe) if (1 - pe) > 0 else 0.0
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'kappa': float(kappa)
    }


def calculate_figure_of_merit(t0_real: np.ndarray, 
                              t1_real: np.ndarray, 
                              t1_predicted: np.ndarray) -> Dict[str, float]:
    """
    Calcula Figure of Merit (FoM) - métrica crítica para expansión urbana.
    
    FoM evalúa solo el CAMBIO (0→1), ignorando áreas estables.
    Es la métrica más importante para modelos de crecimiento urbano.
    
    Argumentos:
        t0_real: Grid inicial real (tiempo t0).
        t1_real: Grid final real (tiempo t1).
        t1_predicted: Grid final predicho (tiempo t1).

    Retorna:
        Diccionario con:
        - fom: Figure of Merit [0-1] (típico 0.25-0.40, >0.50 excelente)
        - A_correct_change: cambio real correctamente predicho
        - B_false_change: cambio simulado incorrecto (FP de cambio)
        - C_missed_change: cambio real no detectado (FN de cambio)
        - change_rate_real: tasa de cambio real
        - change_rate_predicted: tasa de cambio predicha
    """
    # Binarizar
    t0_binary = (t0_real > 0).astype(int)
    t1_real_binary = (t1_real > 0).astype(int)
    t1_pred_binary = (t1_predicted > 0).astype(int)
    
    # Detectar cambios (transiciones 0→1)
    change_real = ((t0_binary == 0) & (t1_real_binary == 1)).astype(int)
    change_predicted = ((t0_binary == 0) & (t1_pred_binary == 1)).astype(int)
    
    # Componentes del FoM
    # A: cambio real correctamente predicho (R ∩ S)
    A = np.sum((change_real == 1) & (change_predicted == 1))
    
    # B: cambio simulado pero no real (S - R) - falso positivo de cambio
    B = np.sum((change_real == 0) & (change_predicted == 1))
    
    # C: cambio real no detectado (R - S) - falso negativo de cambio
    C = np.sum((change_real == 1) & (change_predicted == 0))
    
    # FoM = |R ∩ S| / |R ∪ S|
    # |R ∪ S| = A + B + C (todos los píxeles involucrados en cambio)
    denominator = A + B + C
    fom = A / denominator if denominator > 0 else 0.0
    
    # Tasas de cambio
    total_pixels = t0_binary.size
    change_rate_real = np.sum(change_real) / total_pixels
    change_rate_predicted = np.sum(change_predicted) / total_pixels
    
    logger.info(f"FoM Components: A={A:,} (correcto), B={B:,} (FP cambio), C={C:,} (FN cambio)")
    logger.info(f"FoM = {fom:.4f} (típico: 0.25-0.40, excelente: >0.50)")
    
    return {
        'fom': float(fom),
        'A_correct_change': int(A),
        'B_false_change': int(B),
        'C_missed_change': int(C),
        'change_rate_real': float(change_rate_real),
        'change_rate_predicted': float(change_rate_predicted)
    }


def calculate_change_detection_metrics(t0_real: np.ndarray,
                                       t1_real: np.ndarray,
                                       t1_predicted: np.ndarray) -> Dict[str, float]:
    """
    Métricas específicas para detección de cambio urbano.
    
    Similar a FoM pero con métricas adicionales de cambio.
    """
    # Binarizar
    t0_binary = (t0_real > 0).astype(int)
    t1_real_binary = (t1_real > 0).astype(int)
    t1_pred_binary = (t1_predicted > 0).astype(int)
    
    # Detectar cambios
    change_real = ((t0_binary == 0) & (t1_real_binary == 1)).astype(int)
    change_predicted = ((t0_binary == 0) & (t1_pred_binary == 1)).astype(int)
    
    # Métricas solo sobre cambio
    tp_change = np.sum((change_real == 1) & (change_predicted == 1))
    fp_change = np.sum((change_real == 0) & (change_predicted == 1))
    fn_change = np.sum((change_real == 1) & (change_predicted == 0))
    
    # Precision del cambio
    precision_change = tp_change / (tp_change + fp_change) if (tp_change + fp_change) > 0 else 0.0
    
    # Recall del cambio
    recall_change = tp_change / (tp_change + fn_change) if (tp_change + fn_change) > 0 else 0.0
    
    # F1 del cambio
    f1_change = 2 * (precision_change * recall_change) / (precision_change + recall_change) \
                if (precision_change + recall_change) > 0 else 0.0
    
    return {
        'change_precision': float(precision_change),
        'change_recall': float(recall_change),
        'change_f1': float(f1_change)
    }


def calculate_all_metrics(t0_real: np.ndarray,
                         t1_real: np.ndarray,
                         t1_predicted: np.ndarray) -> Dict[str, any]:
    """
    Calcula TODAS las métricas comprehensivas.
    
    Argumentos:
        t0_real: Grid inicial real.
        t1_real: Grid final real.
        t1_predicted: Grid final predicho.

    Retorna:
        Diccionario completo con todas las métricas.
    """
    logger.info("Calculando métricas comprehensivas...")
    
    # 1. Matriz de confusión (estado final)
    confusion = calculate_confusion_matrix(t1_real, t1_predicted)
    
    # 2. Métricas espaciales globales
    spatial = calculate_spatial_metrics(confusion)
    
    # 3. Figure of Merit (cambio)
    fom = calculate_figure_of_merit(t0_real, t1_real, t1_predicted)
    
    # 4. Métricas de detección de cambio
    change = calculate_change_detection_metrics(t0_real, t1_real, t1_predicted)
    
    # Consolidar todo
    metrics = {
        'confusion_matrix': confusion,
        'spatial_metrics': spatial,
        'figure_of_merit': fom,
        'change_detection': change
    }
    
    logger.info("✓ Métricas calculadas exitosamente")
    
    return metrics


def print_metrics_report(metrics: Dict[str, any], period: str = ""):
    """Imprime un reporte formateado de métricas.

    Argumentos:
        metrics: Diccionario de métricas calculadas (salida de
            :func:`calculate_all_metrics`).
        period: Nombre del periodo (ej. ``"2016→2017"``).
    """
    print("\n" + "="*70)
    if period:
        print(f"REPORTE DE MÉTRICAS: {period}")
    else:
        print("REPORTE DE MÉTRICAS")
    print("="*70)
    
    # Matriz de confusión
    cm = metrics['confusion_matrix']
    print("\n📊 MATRIZ DE CONFUSIÓN (Estado Final)")
    print(f"  TP (urbano correcto):    {cm['TP']:>12,}")
    print(f"  TN (no urbano correcto): {cm['TN']:>12,}")
    print(f"  FP (sobre-estimación):   {cm['FP']:>12,}")
    print(f"  FN (sub-estimación):     {cm['FN']:>12,}")
    
    # Métricas espaciales
    sm = metrics['spatial_metrics']
    print("\n📍 MÉTRICAS ESPACIALES GLOBALES")
    print(f"  Accuracy (Aproximación): {sm['accuracy']:>8.2%}")
    print(f"  Precision:               {sm['precision']:>8.2%}")
    print(f"  Recall (Sensibilidad):   {sm['recall']:>8.2%}")
    print(f"  F1-Score:                {sm['f1_score']:>8.4f}")
    print(f"  Kappa de Cohen:          {sm['kappa']:>8.4f}")
    
    # Figure of Merit
    fom_data = metrics['figure_of_merit']
    print("\n🎯 FIGURE OF MERIT (FoM) - Cambio Urbano")
    print(f"  FoM:                     {fom_data['fom']:>8.4f}")
    print(f"  A (cambio correcto):     {fom_data['A_correct_change']:>12,}")
    print(f"  B (falso cambio):        {fom_data['B_false_change']:>12,}")
    print(f"  C (cambio perdido):      {fom_data['C_missed_change']:>12,}")
    print(f"  Tasa cambio real:        {fom_data['change_rate_real']:>8.2%}")
    print(f"  Tasa cambio predicha:    {fom_data['change_rate_predicted']:>8.2%}")
    
    # Interpretación FoM
    fom_val = fom_data['fom']
    if fom_val > 0.50:
        status = "🟢 EXCELENTE"
    elif fom_val > 0.40:
        status = "🟡 MUY BUENO"
    elif fom_val > 0.25:
        status = "🟠 ACEPTABLE"
    else:
        status = "🔴 MEJORABLE"
    print(f"  Interpretación:          {status}")
    
    # Métricas de cambio
    cd = metrics['change_detection']
    print("\n🔄 DETECCIÓN DE CAMBIO")
    print(f"  Precision (cambio):      {cd['change_precision']:>8.2%}")
    print(f"  Recall (cambio):         {cd['change_recall']:>8.2%}")
    print(f"  F1 (cambio):             {cd['change_f1']:>8.4f}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # Test con datos sintéticos
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("Test de métricas comprehensivas\n")
    
    # Crear grids sintéticos
    size = 100
    t0 = np.zeros((size, size))
    t0[40:60, 40:60] = 1  # Área urbana inicial
    
    t1_real = t0.copy()
    t1_real[35:65, 35:65] = 1  # Expansión real
    
    t1_pred = t0.copy()
    t1_pred[38:62, 38:62] = 1  # Expansión predicha (similar pero no exacta)
    
    # Calcular métricas
    metrics = calculate_all_metrics(t0, t1_real, t1_pred)
    
    # Mostrar reporte
    print_metrics_report(metrics, period="Test Sintético")
