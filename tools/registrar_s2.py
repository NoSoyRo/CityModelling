"""Georreferencia las capturas de Google Earth contra Sentinel-2 cloudless.

Las capturas de data/raw son screenshots del visor de Google Earth y no llevan
georreferencia ni barra de escala, asi que la ventana que cubren y su resolucion
efectiva no se pueden leer del archivo. Los intentos de registro contra imagen
aerea de alta resolucion fallan porque la atribucion de las propias capturas
("Image Landsat / Copernicus") indica que a ese nivel de zoom el visor no
muestra imagen aerea sino el mosaico Sentinel-2. Aqui se usa como referencia el
mosaico s2cloudless de EOX, que es de esa misma familia, y por eso si
correlaciona.

Se estima una transformacion de similitud con SIFT y RANSAC, y se dejan salidas
visuales (tablero de ajedrez y lado a lado) para verificar el ajuste, porque una
solucion con pocos inliers puede ser espuria aunque el numero parezca razonable.
"""

import argparse
import concurrent.futures as futures
import json
import math
import os

import cv2
import numpy as np
import requests
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

TILE = 256
CACHE = "/tmp/teselas_s2"
URL = ("https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-{anio}_3857"
       "/default/g/{z}/{y}/{x}.jpg")
ANIOS_S2 = (2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024)

LAT_C, LON_C = 20.59, -100.39


def a_pixel(lat, lon, z):
    n = TILE * 2 ** z
    s = math.sin(math.radians(lat))
    return ((lon + 180.0) / 360.0 * n,
            (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * n)


def a_lonlat(x, y, z):
    n = TILE * 2 ** z
    return (x / n * 360.0 - 180.0,
            math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n)))))


def bajar(args):
    anio, z, tx, ty = args
    ruta = f"{CACHE}/{anio}_{z}_{tx}_{ty}.jpg"
    if os.path.exists(ruta) and os.path.getsize(ruta) > 500:
        return
    for _ in range(3):
        try:
            r = requests.get(URL.format(anio=anio, z=z, x=tx, y=ty),
                             headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            if r.status_code == 200 and len(r.content) > 500:
                with open(ruta, "wb") as fh:
                    fh.write(r.content)
                return
        except requests.RequestException:
            pass


def mosaico(anio_s2, zoom, margen_lat, margen_lon):
    destino = f"/tmp/mos_s2_{anio_s2}_z{zoom}_{margen_lon}.png"
    x0, y0 = a_pixel(LAT_C + margen_lat, LON_C - margen_lon, zoom)
    x1, y1 = a_pixel(LAT_C - margen_lat, LON_C + margen_lon, zoom)
    tx0, ty0 = int(x0 // TILE), int(y0 // TILE)
    tx1, ty1 = int(x1 // TILE), int(y1 // TILE)
    if os.path.exists(destino):
        return Image.open(destino), tx0 * TILE, ty0 * TILE

    os.makedirs(CACHE, exist_ok=True)
    tareas = [(anio_s2, zoom, tx, ty)
              for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]
    print(f"descargando {len(tareas)} teselas s2cloudless-{anio_s2} z={zoom}",
          flush=True)
    with futures.ThreadPoolExecutor(max_workers=24) as ex:
        list(ex.map(bajar, tareas))
    lienzo = Image.new("RGB", ((tx1 - tx0 + 1) * TILE, (ty1 - ty0 + 1) * TILE))
    faltan = 0
    for t in tareas:
        p = f"{CACHE}/{t[0]}_{t[1]}_{t[2]}_{t[3]}.jpg"
        if os.path.exists(p):
            with Image.open(p) as im:
                lienzo.paste(im.convert("RGB"),
                             ((t[2] - tx0) * TILE, (t[3] - ty0) * TILE))
        else:
            faltan += 1
    lienzo.save(destino)
    print(f"mosaico {lienzo.size} | faltantes {faltan}")
    return lienzo, tx0 * TILE, ty0 * TILE


def prep(arr):
    g = cv2.cvtColor(np.asarray(arr, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    return cv2.createCLAHE(3.0, (8, 8)).apply(g)


def registrar(cap_img, ref_g, ref_rgb, ox, oy, zoom, red_cap, m_px_ref,
              etiqueta, ref_kd=None):
    W, H = cap_img.size
    cap = cap_img.crop((0, 0, W, int(H * 0.94)))
    cw, ch = int(cap.size[0] / red_cap), int(cap.size[1] / red_cap)
    cap_g = prep(np.array(cap.resize((cw, ch), Image.LANCZOS)))

    sift = cv2.SIFT_create(nfeatures=60000, contrastThreshold=0.015)
    k1, d1 = sift.detectAndCompute(cap_g, None)
    # Los rasgos de la referencia son los mismos para todos los anios.
    k2, d2 = ref_kd if ref_kd else sift.detectAndCompute(ref_g, None)
    flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 8}, {"checks": 200})
    buenos = [m for m, n in flann.knnMatch(d1, d2, k=2)
              if m.distance < 0.8 * n.distance]
    if len(buenos) < 15:
        print(f"{etiqueta}: solo {len(buenos)} candidatos")
        return None
    src = np.float32([k1[m.queryIdx].pt for m in buenos]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in buenos]).reshape(-1, 1, 2)
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                         ransacReprojThreshold=3.0,
                                         maxIters=80000, confidence=0.9995)
    if M is None:
        return None
    inl = inl.ravel().astype(bool)
    esc = math.hypot(M[0, 0], M[1, 0])
    giro = math.degrees(math.atan2(M[1, 0], M[0, 0]))
    m_px = esc * m_px_ref / red_cap
    centro = cv2.transform(np.float32([[[cw / 2, ch / 2 / 0.94]]]), M)[0][0]
    lon_c, lat_c = a_lonlat(ox + centro[0], oy + centro[1], zoom)
    res = {
        "rasgos_captura": int(len(k1)), "candidatos": int(len(buenos)),
        "inliers": int(inl.sum()),
        "pct_inliers": round(100 * float(inl.sum()) / len(buenos), 1),
        "giro_deg": round(float(giro), 2),
        "m_por_pixel": round(float(m_px), 2),
        "km_x": round(float(W * m_px / 1000), 2),
        "km_y": round(float(H * m_px / 1000), 2),
        "km2": int(round(W * H * m_px ** 2 / 1e6)),
        "lat_centro": round(float(lat_c), 4),
        "lon_centro": round(float(lon_c), 4),
    }
    print(f"{etiqueta}: inliers {res['inliers']}/{res['candidatos']} "
          f"({res['pct_inliers']}%) | giro {giro:+.2f} | "
          f"pixel {m_px:.2f} m | ventana {res['km_x']} x {res['km_y']} km "
          f"= {res['km2']} km2 | centro ({lat_c:.4f}, {lon_c:.4f})", flush=True)

    # Verificacion visual.
    alin = cv2.warpAffine(np.array(cap.resize((cw, ch), Image.LANCZOS)), M,
                          (ref_g.shape[1], ref_g.shape[0]))
    esq = cv2.transform(np.float32([[[0, 0]], [[cw, 0]], [[cw, ch]],
                                    [[0, ch]]]), M).reshape(-1, 2)
    xi, yi = max(0, int(esq[:, 0].min())), max(0, int(esq[:, 1].min()))
    xf = min(ref_g.shape[1], int(esq[:, 0].max()))
    yf = min(ref_g.shape[0], int(esq[:, 1].max()))
    if xf - xi > 60 and yf - yi > 60:
        A, B = alin[yi:yf, xi:xf], ref_rgb[yi:yf, xi:xf]
        os.makedirs("/tmp/reg_s2", exist_ok=True)
        tam = (1150, int(1150 * A.shape[0] / A.shape[1]))
        tab = B.copy()
        paso = max(24, A.shape[1] // 14)
        for i in range(0, A.shape[0], paso):
            for j in range(0, A.shape[1], paso):
                if ((i // paso) + (j // paso)) % 2 == 0:
                    tab[i:i + paso, j:j + paso] = A[i:i + paso, j:j + paso]
        Image.fromarray(tab).resize(tam, Image.LANCZOS).save(
            f"/tmp/reg_s2/tablero_{etiqueta}.png")
        Image.fromarray(A).resize(tam, Image.LANCZOS).save(
            f"/tmp/reg_s2/alineada_{etiqueta}.png")
        Image.fromarray(B).resize(tam, Image.LANCZOS).save(
            f"/tmp/reg_s2/referencia_{etiqueta}.png")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("anios", nargs="*", default=["2020"])
    ap.add_argument("--anio-s2", type=int, default=2020)
    ap.add_argument("--zoom", type=int, default=13)
    ap.add_argument("--margen-lat", type=float, default=0.55)
    ap.add_argument("--margen-lon", type=float, default=0.75)
    ap.add_argument("--red-cap", type=float, default=2.0)
    ap.add_argument("--salida", default=None)
    a = ap.parse_args()

    ref_img, ox, oy = mosaico(a.anio_s2, a.zoom, a.margen_lat, a.margen_lon)
    ref_rgb = np.array(ref_img)
    ref_g = prep(ref_rgb)
    m_px_ref = 156543.03392 / 2 ** a.zoom * math.cos(math.radians(LAT_C))
    print(f"referencia a {m_px_ref:.2f} m/px", flush=True)
    ref_kd = cv2.SIFT_create(nfeatures=60000,
                             contrastThreshold=0.015).detectAndCompute(ref_g,
                                                                       None)
    print(f"rasgos de referencia: {len(ref_kd[0])}\n", flush=True)

    out = []
    for anio in a.anios:
        cap = Image.open(f"data/raw/imagen_{anio}.png").convert("RGB")
        r = registrar(cap, ref_g, ref_rgb, ox, oy, a.zoom, a.red_cap,
                      m_px_ref, anio, ref_kd)
        if r:
            r["anio"] = int(anio)
            out.append(r)

    # Un ajuste con muy pocos inliers no es una solucion, es ruido: se descarta
    # antes de resumir para que no contamine las estadisticas.
    buenas = [o for o in out if o["inliers"] >= 6 and o["m_por_pixel"] > 1]
    descartadas = [o["anio"] for o in out if o not in buenas]
    resumen = None
    if len(buenas) > 1:
        v = np.array([o["m_por_pixel"] for o in buenas])
        g = np.array([o["giro_deg"] for o in buenas])
        la = np.array([o["lat_centro"] for o in buenas])
        lo = np.array([o["lon_centro"] for o in buenas])
        mp = float(np.median(v))
        print(f"\n=== {len(buenas)} capturas con ajuste solido "
              f"(descartadas por pocos inliers: {descartadas}) ===")
        print(f"pixel  : mediana {np.median(v):.2f} m | rango "
              f"{v.min():.2f}-{v.max():.2f} | desv {v.std():.2f}")
        print(f"giro   : mediana {np.median(g):+.2f} deg | rango "
              f"{g.min():+.2f} a {g.max():+.2f} | desv {g.std():.2f}")
        print(f"centro : ({np.median(la):.4f}, {np.median(lo):.4f}) | "
              f"disp lat {la.std() * 110540:.0f} m, "
              f"lon {lo.std() * 111320 * math.cos(math.radians(LAT_C)):.0f} m")
        print(f"ventana: {3024 * mp / 1000:.1f} x {1792 * mp / 1000:.1f} km = "
              f"{3024 * 1792 * mp ** 2 / 1e6:.0f} km2 | "
              f"{mp ** 2:.0f} m2 por pixel")
        resumen = {
            "n_capturas_solidas": len(buenas),
            "anios_descartados": descartadas,
            "m_por_pixel_mediana": round(mp, 2),
            "m_por_pixel_rango": [round(float(v.min()), 2),
                                  round(float(v.max()), 2)],
            "giro_deg_mediana": round(float(np.median(g)), 2),
            "giro_deg_desv": round(float(g.std()), 2),
            "lat_centro_mediana": round(float(np.median(la)), 4),
            "lon_centro_mediana": round(float(np.median(lo)), 4),
            "km_x": round(3024 * mp / 1000, 1),
            "km_y": round(1792 * mp / 1000, 1),
            "km2": int(round(3024 * 1792 * mp ** 2 / 1e6)),
            "m2_por_pixel": int(round(mp ** 2)),
        }
    if a.salida and out:
        with open(a.salida, "w") as fh:
            json.dump({"resumen": resumen, "por_captura": out}, fh, indent=2)
        print(f"escrito {a.salida}")


if __name__ == "__main__":
    main()
