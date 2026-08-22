"""Verifica el encuadre recuperado renderizando la referencia con sus parametros.

Toma el centro, el giro y la escala estimados en georreferencia_capturas.json,
renderiza la referencia Sentinel-2 con esa misma geometria y numero de pixeles,
y la pone junto a la captura. Si los parametros son correctos las dos imagenes
deben mostrar la misma escena con los mismos rasgos en las mismas posiciones.
"""

import argparse
import concurrent.futures as futures
import json
import math
import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, "tools")
from registrar_s2 import a_pixel, bajar  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
TILE = 256
W, H = 3024, 1792


def mosaico_centrado(lat_c, lon_c, semi_km, zoom, anio_s2):
    m_px = 156543.03392 / 2 ** zoom * math.cos(math.radians(lat_c))
    semi = semi_km * 1000 / m_px
    cx, cy = a_pixel(lat_c, lon_c, zoom)
    tx0, ty0 = int((cx - semi) // TILE), int((cy - semi) // TILE)
    tx1, ty1 = int((cx + semi) // TILE), int((cy + semi) // TILE)
    os.makedirs("/tmp/teselas_s2", exist_ok=True)
    tareas = [(anio_s2, zoom, tx, ty)
              for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]
    with futures.ThreadPoolExecutor(max_workers=24) as ex:
        list(ex.map(bajar, tareas))
    lienzo = Image.new("RGB", ((tx1 - tx0 + 1) * TILE, (ty1 - ty0 + 1) * TILE))
    for t in tareas:
        p = f"/tmp/teselas_s2/{t[0]}_{t[1]}_{t[2]}_{t[3]}.jpg"
        if os.path.exists(p):
            with Image.open(p) as im:
                lienzo.paste(im.convert("RGB"),
                             ((t[2] - tx0) * TILE, (t[3] - ty0) * TILE))
    return lienzo, tx0 * TILE, ty0 * TILE, m_px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anio", default="2020")
    ap.add_argument("--json", default="data/processed/georreferencia_capturas.json")
    ap.add_argument("--zoom", type=int, default=13)
    ap.add_argument("--anio-s2", type=int, default=2020)
    ap.add_argument("--salida", default="/tmp/verificacion_encuadre.png")
    a = ap.parse_args()

    d = json.load(open(a.json))
    r = d["resumen"]
    lat_c, lon_c = r["lat_centro_mediana"], r["lon_centro_mediana"]
    m_px, giro = r["m_por_pixel_mediana"], r["giro_deg_mediana"]
    print(f"parametros recuperados: centro ({lat_c}, {lon_c}) | "
          f"{m_px} m/px | giro {giro} deg")
    print(f"ventana {r['km_x']} x {r['km_y']} km = {r['km2']} km2")

    diag = math.hypot(W, H) * m_px / 1000 / 2 * 1.1
    mos, ox, oy, m_px_mos = mosaico_centrado(lat_c, lon_c, diag, a.zoom,
                                             a.anio_s2)
    f = m_px_mos / m_px
    mos_e = mos.resize((int(mos.size[0] * f), int(mos.size[1] * f)),
                       Image.LANCZOS)
    cx, cy = a_pixel(lat_c, lon_c, a.zoom)
    cxe, cye = (cx - ox) * f, (cy - oy) * f
    # El giro estimado lleva la captura al marco norte arriba; para ir en
    # sentido contrario se aplica el mismo angulo con signo opuesto.
    rot = mos_e.rotate(giro, resample=Image.BICUBIC, center=(cxe, cye))
    ref = rot.crop((int(cxe - W / 2), int(cye - H / 2),
                    int(cxe + W / 2), int(cye + H / 2)))

    cap = Image.open(f"data/raw/imagen_{a.anio}.png").convert("RGB")
    vw = 1250
    vh = int(vw * H / W)
    cap_v = cap.resize((vw, vh), Image.LANCZOS)
    ref_v = ref.resize((vw, vh), Image.LANCZOS)

    barra = 30
    lienzo = Image.new("RGB", (vw, (vh + barra) * 3 + 6), (16, 16, 16))
    dr = ImageDraw.Draw(lienzo)
    lienzo.paste(cap_v, (0, barra))
    dr.text((6, 8), f"CAPTURA {a.anio} (Google Earth, tal cual)",
            fill=(255, 255, 0))
    lienzo.paste(ref_v, (0, vh + barra * 2 + 2))
    dr.text((6, vh + barra + 8),
            f"REFERENCIA Sentinel-2 con los parametros recuperados: "
            f"{m_px} m/px, giro {giro} deg, centro ({lat_c}, {lon_c})",
            fill=(0, 255, 255))
    # Tablero para juzgar la continuidad de los rasgos.
    import numpy as np
    A, B = np.array(cap_v), np.array(ref_v)
    tab = B.copy()
    paso = vw // 14
    for i in range(0, vh, paso):
        for j in range(0, vw, paso):
            if ((i // paso) + (j // paso)) % 2 == 0:
                tab[i:i + paso, j:j + paso] = A[i:i + paso, j:j + paso]
    lienzo.paste(Image.fromarray(tab), (0, (vh + barra) * 2 + 4))
    dr.text((6, (vh + barra) * 2 - 22),
            "TABLERO: casillas alternas de captura y referencia",
            fill=(255, 255, 255))
    lienzo.save(a.salida)
    print(f"escrito {a.salida} {lienzo.size}")


if __name__ == "__main__":
    main()
