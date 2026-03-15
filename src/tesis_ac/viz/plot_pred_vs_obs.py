import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from tesis_ac.historical.extract_transitions import load_historical_maps
from tesis_ac.ca.rules import WoECellularAutomaton, ACParameters, create_default_spatial_variables
from tesis_ac.woe.woe import WoECalculator, calculate_spatial_woe_variables

BASE = "/Users/rod/Projects/MSC/Tesis/CityModelling"
DATA_DIR = Path(os.path.join(BASE, "data", "processed", "standardized_maps"))
VAL_DIR = Path(os.path.join(BASE, "data", "processed", "validation_2015_2020"))
OUT_DIR = Path(os.path.join(BASE, "figures"))

def discover_periods() -> list[tuple[int, int]]:
    """Descubre periodos disponibles basados en predicciones precomputadas en VAL_DIR.
    Busca archivos predicted_YYYY_*.npy y asume t0=(YYYY-1), t1=YYYY si existen grids en DATA_DIR.
    """
    periods: list[tuple[int,int]] = []
    if not VAL_DIR.exists():
        return periods
    for p in VAL_DIR.iterdir():
        name = p.name
        if name.startswith("predicted_") and name.endswith(".npy"):
            try:
                year = int(name.split("_")[1])
            except Exception:
                continue
            t0, t1 = year - 1, year
            # Validar que existan los grids estándar
            std_t0 = DATA_DIR / f"year_{t0}" / "svm_2_classes.npy"
            std_t1 = DATA_DIR / f"year_{t1}" / "svm_2_classes.npy"
            if std_t0.exists() and std_t1.exists():
                periods.append((t0, t1))
    return sorted(periods)


def load_predicted_for_period(target_year: int) -> np.ndarray | None:
    """Intentar cargar un mapa predicho existente para el año objetivo.
    Busca archivos como predicted_YYYY_full.npy en data/processed/validation_2015_2020.
    """
    if not VAL_DIR.exists():
        return None
    candidates = []
    for name in [f"predicted_{target_year}_full.npy", f"predicted_{target_year}_simple.npy"]:
        p = VAL_DIR / name
        if p.exists():
            candidates.append(p)
    if not candidates:
        return None
    # Prioriza 'full' si existe
    path = candidates[0]
    try:
        return np.load(path)
    except Exception:
        return None


def simulate_period(data_dir: Path, initial_year: int, target_year: int, n_steps: int = 3, seed: int = 123):
    years_data = load_historical_maps(data_dir, file_pattern="svm_2_classes.npy")
    if initial_year not in years_data or target_year not in years_data:
        raise ValueError(f"Años no disponibles: {initial_year},{target_year}")

    grid_t0 = years_data[initial_year]
    grid_t1 = years_data[target_year]
    grid_shape = grid_t0.shape
    assert grid_t1.shape == grid_shape, "Shapes distintos entre t0 y t1"

    spatial_vars = create_default_spatial_variables(grid_shape, existing_urban=grid_t0)

    urban_history = (grid_t0 == 0) & (grid_t1 == 1)
    woe_calc = WoECalculator(min_bin_size=100, max_bins=8)
    _ = calculate_spatial_woe_variables(
        grid=grid_t0,
        urban_history=urban_history.astype(int),
        features=spatial_vars,
        calculator=woe_calc,
    )

    grid_pred = load_predicted_for_period(target_year)
    if grid_pred is None:
        raise RuntimeError(f"No hay predicción precomputada para {target_year} en {VAL_DIR}")

    return grid_t0, grid_t1, grid_pred


def make_confusion_change(grid_t0: np.ndarray, grid_t1: np.ndarray, grid_pred: np.ndarray) -> np.ndarray:
    """
    Categorización:
      0: TN (no cambio correcto)
      1: TP cambio (predice cambio y ocurrió cambio)
      2: FP cambio (predice cambio pero no ocurrió)
      3: FN cambio (no predice cambio pero ocurrió)
    """
    pred_change = (grid_pred == 1) & (grid_t0 == 0)
    obs_change = (grid_t1 == 1) & (grid_t0 == 0)
    tn = (~pred_change) & (~obs_change)
    tp = pred_change & obs_change
    fp = pred_change & (~obs_change)
    fn = (~pred_change) & obs_change
    conf = np.zeros_like(grid_t0, dtype=np.uint8)
    conf[tp] = 1
    conf[fp] = 2
    conf[fn] = 3
    return conf

def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = (a == 1)
    b = (b == 1)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def orient_to_match(ref: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, bool]:
    """Devuelve grid (posible inversión) para maximizar IoU vs ref, y flag si se invirtió."""
    i_same = iou(grid, ref)
    i_inv = iou(1 - grid, ref)
    if i_inv > i_same:
        return (1 - grid), True
    return grid, False


def plot_period(initial_year: int, target_year: int, grid_t1: np.ndarray, grid_pred: np.ndarray, conf: np.ndarray, out_path: Path):
    plt.figure(figsize=(12, 7))
    ax1 = plt.subplot(1, 3, 1)
    # Fijar paleta: urbano (1) en blanco, no urbano (0) en negro
    ax1.imshow(grid_pred, cmap='binary', vmin=0, vmax=1)
    ax1.set_title(f"Predicción t1 ({target_year})")
    ax1.axis('off')

    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(grid_t1, cmap='binary', vmin=0, vmax=1)
    ax2.set_title(f"Observado t1 ({target_year})")
    ax2.axis('off')

    ax3 = plt.subplot(1, 3, 3)
    # Colormap discreto: 0=TN gris claro, 1=TP verde, 2=FP rojo, 3=FN naranja
    import matplotlib.colors as mcolors
    cmap = mcolors.ListedColormap(['#d9d9d9', '#4daf4a', '#e41a1c', '#ff7f00'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    ax3.imshow(conf, cmap=cmap, norm=norm)
    ax3.set_title("Error de cambio: TN/TP/FP/FN")
    ax3.axis('off')

    plt.suptitle(f"Pred vs Obs (WoE-AC) {initial_year}→{target_year}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    periods = discover_periods()
    if not periods:
        print("No se encontraron periodos con predicciones precomputadas en validation_2015_2020.")
        return
    for (t0, t1) in periods:
        try:
            grid_t0, grid_t1, grid_pred = simulate_period(DATA_DIR, t0, t1, n_steps=3, seed=123)
        except Exception as e:
            print(f"Skip {t0}→{t1}: {e}")
            continue
        # Alinear orientaciones: t0 y pred respecto a t1
        grid_t0_oriented, inv_t0 = orient_to_match(grid_t1, grid_t0)
        grid_pred_oriented, inv_pred = orient_to_match(grid_t1, grid_pred)
        if inv_t0 or inv_pred:
            print(f"[{t0}→{t1}] Inversion aplicada: t0={inv_t0}, pred={inv_pred}")
        conf = make_confusion_change(grid_t0_oriented, grid_t1, grid_pred_oriented)
        out_path = OUT_DIR / f"pred_vs_obs_{t0}_{t1}.png"
        plot_period(t0, t1, grid_t1, grid_pred_oriented, conf, out_path)


if __name__ == "__main__":
    main()
