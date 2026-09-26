"""Arma un paquete autocontenido de la tesis listo para subir a Overleaf.

Genera entregas/overleaf/ y su zip. No toca las fuentes del repositorio: todas
las transformaciones se aplican sobre la copia.

Tres cosas se resuelven aqui porque Overleaf no las tolera tal como estan:

1. main.tex no declara \\documentclass (lo hereda de standalone_preamble.tex).
   Overleaf elige el documento principal buscando ese comando, asi que elegiria
   el preambulo. En la copia el preambulo queda fusionado dentro de main.tex.
2. \\addbibresource apunta a ../back/referencias.bib, fuera de caps_larraga.
   En la copia el .bib viaja dentro del proyecto.
3. Las figuras usadas pesan 84 MB y el zip de Overleaf admite 50 MB. Las
   fotograficas pasan a JPEG; las de color plano se copian intactas porque
   reconvertirlas las engorda.

Uso:
    python tools/preparar_overleaf.py
"""

from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
FUENTE = RAIZ / "report/tesis/tesis_indice_nuevo"
CAPS = FUENTE / "caps_larraga"
DESTINO = RAIZ / "entregas/overleaf"
ZIP = RAIZ / "entregas/Tesis_Overleaf.zip"

# Por encima de este tamano conviene pasar a JPEG; por debajo el PNG ya es
# mas compacto que cualquier reconversion.
UMBRAL_JPEG = 400 * 1024
ANCHO_MAXIMO = 2200
CALIDAD = 90

DIRECTORIOS_TEX = [
    "front",
    "cap01_introduccion",
    "cap02_marco_teorico",
    "cap03_estado_del_arte",
    "cap05_modelo_crecimiento_urbano",
    "cap06_resultados_analisis",
    "cap07_conclusiones",
    "apendice_arquitectura",
]


def figuras_referenciadas() -> tuple[set[Path], list[str]]:
    """Resuelve cada \\includegraphics a un archivo real bajo caps_larraga."""
    referencias = set()
    for tex in CAPS.rglob("*.tex"):
        if tex.name.endswith("_standalone.tex"):
            continue
        texto = tex.read_text(errors="ignore")
        patron = r"includegraphics(?:\[[^\]]*\])?\{([^}]+)\}"
        referencias.update(m.group(1) for m in re.finditer(patron, texto))

    usadas: set[Path] = set()
    sin_resolver: list[str] = []
    for ref in sorted(referencias):
        encontrado = None
        for prefijo in ("", "figures/"):
            for ext in ("", ".png", ".pdf", ".jpg", ".jpeg"):
                candidato = CAPS / (prefijo + ref + ext)
                if candidato.is_file():
                    encontrado = candidato
                    break
            if encontrado:
                break
        if encontrado:
            usadas.add(encontrado)
        else:
            sin_resolver.append(ref)
    return usadas, sin_resolver


def copiar_figura(origen: Path, carpeta: Path) -> tuple[Path, int, int]:
    """Copia la figura, pasandola a JPEG si es fotografica y pesada."""
    antes = origen.stat().st_size
    if origen.suffix.lower() != ".png" or antes <= UMBRAL_JPEG:
        destino = carpeta / origen.name
        shutil.copy2(origen, destino)
        return destino, antes, antes

    imagen = Image.open(origen)
    if imagen.mode in ("RGBA", "LA", "P"):
        imagen = imagen.convert("RGBA")
        fondo = Image.new("RGBA", imagen.size, (255, 255, 255, 255))
        imagen = Image.alpha_composite(fondo, imagen)
    imagen = imagen.convert("RGB")

    if imagen.width > ANCHO_MAXIMO:
        alto = round(imagen.height * ANCHO_MAXIMO / imagen.width)
        imagen = imagen.resize((ANCHO_MAXIMO, alto), Image.LANCZOS)

    destino = carpeta / (origen.stem + ".jpg")
    imagen.save(destino, "JPEG", quality=CALIDAD, optimize=True, progressive=False)
    return destino, antes, destino.stat().st_size


def construir_main() -> str:
    """Fusiona el preambulo dentro de main.tex y corrige la ruta del .bib."""
    preambulo = (CAPS / "standalone_preamble.tex").read_text()
    preambulo = preambulo.replace(
        "\\addbibresource{../back/referencias.bib}",
        "\\addbibresource{back/referencias.bib}",
    )
    cuerpo = (CAPS / "main.tex").read_text()
    marca = "\\input{standalone_preamble.tex}"
    if marca not in cuerpo:
        raise SystemExit("main.tex ya no incluye el preambulo con la marca esperada")
    encabezado = (
        "% Version para Overleaf generada por tools/preparar_overleaf.py\n"
        "% El preambulo esta fusionado aqui para que Overleaf reconozca este\n"
        "% archivo como documento principal.\n"
    )
    return encabezado + cuerpo.replace(marca, preambulo)


def main() -> None:
    if DESTINO.exists():
        shutil.rmtree(DESTINO)
    DESTINO.mkdir(parents=True)

    (DESTINO / "main.tex").write_text(construir_main())
    shutil.copy2(CAPS / "thesis-commands.sty", DESTINO / "thesis-commands.sty")
    (DESTINO / "back").mkdir()
    shutil.copy2(FUENTE / "back/referencias.bib", DESTINO / "back/referencias.bib")

    for nombre in DIRECTORIOS_TEX:
        origen = CAPS / nombre
        carpeta = DESTINO / nombre
        carpeta.mkdir()
        for tex in sorted(origen.glob("*.tex")):
            if tex.name.endswith("_standalone.tex"):
                continue
            shutil.copy2(tex, carpeta / tex.name)

    usadas, sin_resolver = figuras_referenciadas()
    destino_figuras = DESTINO / "figures"
    destino_figuras.mkdir()

    renombres: dict[str, str] = {}
    total_antes = total_despues = 0
    convertidas = 0
    for figura in sorted(usadas):
        nuevo, antes, despues = copiar_figura(figura, destino_figuras)
        total_antes += antes
        total_despues += despues
        if nuevo.name != figura.name:
            renombres[figura.name] = nuevo.name
            convertidas += 1

    # Solo las referencias con extension explicita necesitan reescritura; las
    # que la omiten las resuelve graphicx contra el .jpg.
    tocados = 0
    for tex in sorted(DESTINO.rglob("*.tex")):
        texto = original = tex.read_text()
        for viejo, nuevo in renombres.items():
            texto = texto.replace(viejo, nuevo)
        if texto != original:
            tex.write_text(texto)
            tocados += 1

    (DESTINO / "README_OVERLEAF.md").write_text(LEEME)

    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for archivo in sorted(DESTINO.rglob("*")):
            if archivo.is_file():
                z.write(archivo, archivo.relative_to(DESTINO))

    mb = 1024 * 1024
    print(f"figuras usadas:    {len(usadas)}")
    print(f"convertidas a JPEG:{convertidas}")
    print(f"peso figuras:      {total_antes/mb:.1f} MB -> {total_despues/mb:.1f} MB")
    print(f".tex reescritos:   {tocados}")
    print(f"carpeta:           {DESTINO.relative_to(RAIZ)}")
    print(f"zip:               {ZIP.relative_to(RAIZ)} ({ZIP.stat().st_size/mb:.1f} MB)")
    if sin_resolver:
        print(f"referencias opcionales ausentes: {', '.join(sin_resolver)}")


LEEME = """# Tesis en Overleaf

Paquete generado con `tools/preparar_overleaf.py`. Se sube tal cual.

## Como subirlo

1. En el proyecto de Overleaf, menu de la izquierda, boton de subir archivos.
2. Arrastra `Tesis_Overleaf.zip`. Overleaf lo descomprime y conserva las
   carpetas.
3. Si el proyecto ya tenia un `main.tex` de ejemplo, borralo antes para que no
   choque con el de la tesis.
4. Menu, seccion Settings, campo *Main document*: debe decir `main.tex`.
5. Compilador: pdfLaTeX. Bibliografia: Biber. Overleaf lo detecta solo.

## Diferencias respecto al repositorio

El contenido es el mismo, con tres ajustes que solo existen en esta copia:

- El preambulo (`standalone_preamble.tex`) esta fusionado dentro de `main.tex`.
  Overleaf identifica el documento principal buscando `\\documentclass`, y en el
  repositorio ese comando vive en el preambulo, no en `main.tex`.
- `referencias.bib` viaja en `back/` dentro del proyecto. En el repositorio esta
  un nivel arriba de la carpeta de capitulos.
- Las figuras fotograficas estan en JPEG a 2200 px de ancho. Las de color plano
  siguen en PNG porque convertirlas las hacia mas pesadas. Se excluyeron las 37
  figuras que ningun capitulo referencia.

Los archivos `*_standalone.tex`, que sirven para compilar capitulos por separado
en local, no se incluyen: dependen del preambulo como archivo aparte.

## Si la compilacion se pasa de tiempo

El documento tarda unos 55 segundos en cuatro pasadas en una maquina local. En
Overleaf el limite de los planes de pago es de 4 minutos, que deberia alcanzar.
Si no alcanza, en Settings se puede desactivar `microtype`, que es de lo mas
costoso, o compilar por capitulos cambiando el *Main document*.
"""


if __name__ == "__main__":
    main()
