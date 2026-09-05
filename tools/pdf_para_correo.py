#!/usr/bin/env python3
"""Genera una copia ligera de la tesis, apta para enviar por correo.

Ghostscript remuestrea las imágenes a 300 ppp y baja el PDF de 92 MB a unos
3 MB, pero al reescribir las fuentes mutila los mapas ToUnicode: el texto
extraído pierde las ligaduras («clasi<ESC>cación») y las letras matemáticas,
así que las búsquedas fallan. Aquí se copian los mapas del PDF original a la
copia ligera, código por código, para que se pueda buscar y copiar igual que
el original.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pikepdf

# Las ligaduras no están en el ToUnicode ni del original: el extractor las
# resuelve por el nombre del glifo, que Ghostscript pierde. Van explícitas.
LIGADURAS = {0x1B: "FB01", 0x1C: "FB03", 0x1D: "FB00", 0x1E: "FB02"}
FUENTES_DE_TEXTO = "LinLibertine"


def comprimir(entrada: Path, salida: Path, ppp: int) -> None:
    subprocess.run(
        [
            "gs",
            "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.7",
            "-dNOPAUSE",
            "-dQUIET",
            "-dBATCH",
            "-dDetectDuplicateImages=true",
            "-dCompressFonts=true",
            "-dDownsampleColorImages=true",
            "-dDownsampleGrayImages=true",
            "-dDownsampleMonoImages=true",
            f"-dColorImageResolution={ppp}",
            f"-dGrayImageResolution={ppp}",
            f"-dMonoImageResolution={ppp}",
            "-dColorImageDownsampleType=/Bicubic",
            "-dGrayImageDownsampleType=/Bicubic",
            f"-sOutputFile={salida}",
            str(entrada),
        ],
        check=True,
    )


def leer_cmap(texto: str) -> dict[int, str]:
    """Devuelve código -> destino Unicode en hexadecimal, tal como lo declara el CMap."""
    mapa: dict[int, str] = {}
    for bloque in re.findall(r"beginbfrange(.*?)endbfrange", texto, re.S):
        patron = r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>"
        for inicio, fin, destino in re.findall(patron, bloque):
            base, ancho = int(destino, 16), len(destino)
            for desplazamiento, codigo in enumerate(range(int(inicio, 16), int(fin, 16) + 1)):
                mapa[codigo] = format(base + desplazamiento, f"0{ancho}X")
    for bloque in re.findall(r"beginbfchar(.*?)endbfchar", texto, re.S):
        for codigo, destino in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", bloque):
            mapa[int(codigo, 16)] = destino.upper()
    return mapa


def mapas_del_original(pdf_path: Path) -> dict[str, dict[int, str]]:
    """Junta, por familia tipográfica, todos los mapas ToUnicode del PDF de origen."""
    referencia: dict[str, dict[int, str]] = {}
    with pikepdf.open(pdf_path) as pdf:
        for pagina in pdf.pages:
            for _, fuente in (pagina.get("/Resources", {}).get("/Font", {}) or {}).items():
                tounicode = fuente.get("/ToUnicode")
                if tounicode is None:
                    continue
                familia = str(fuente.get("/BaseFont", "")).split("+")[-1]
                cmap = bytes(tounicode.read_bytes()).decode("latin-1")
                referencia.setdefault(familia, {}).update(leer_cmap(cmap))
    return referencia


def reparar_unicode(pdf_path: Path, original: Path) -> int:
    referencia = mapas_del_original(original)
    pdf = pikepdf.open(pdf_path, allow_overwriting_input=True)
    reparadas = 0
    vistos: set[tuple[int, int]] = set()
    for pagina in pdf.pages:
        for _, fuente in (pagina.get("/Resources", {}).get("/Font", {}) or {}).items():
            tounicode = fuente.get("/ToUnicode")
            if tounicode is None:
                continue
            clave = tounicode.objgen
            if clave in vistos:
                continue
            vistos.add(clave)

            familia = str(fuente.get("/BaseFont", "")).split("+")[-1]
            esperado = dict(referencia.get(familia, {}))
            if FUENTES_DE_TEXTO in familia:
                esperado.update(LIGADURAS)
            if not esperado:
                continue

            cmap = bytes(tounicode.read_bytes()).decode("latin-1")
            presentes = leer_cmap(cmap)
            faltantes = {c: u for c, u in esperado.items() if c not in presentes}
            if not faltantes:
                continue

            # El formato CMap admite como mucho 100 pares por bloque.
            entradas = sorted(faltantes.items())
            bloque: list[str] = []
            for inicio in range(0, len(entradas), 100):
                tramo = entradas[inicio : inicio + 100]
                bloque.append(f"{len(tramo)} beginbfchar")
                bloque += [f"<{c:02X}> <{u}>" for c, u in tramo]
                bloque.append("endbfchar")
            nuevo = cmap.replace("endcmap", "\n".join(bloque) + "\nendcmap", 1)
            tounicode.write(nuevo.encode("latin-1"))
            reparadas += 1
    pdf.save()
    return reparadas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entrada", type=Path)
    parser.add_argument("salida", type=Path)
    parser.add_argument("--ppp", type=int, default=300)
    args = parser.parse_args()

    comprimir(args.entrada, args.salida, args.ppp)
    reparadas = reparar_unicode(args.salida, args.entrada)
    origen_mb = args.entrada.stat().st_size / 1e6
    destino_mb = args.salida.stat().st_size / 1e6
    print(f"{origen_mb:.1f} MB -> {destino_mb:.1f} MB, {reparadas} fuentes reparadas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
