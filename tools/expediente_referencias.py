"""Arma un expediente por referencia para poder validarla contra su fuente.

Para cada clave del .bib reune dos cosas:
  1. La entrada bibliografica completa, tal como esta escrita.
  2. Cada oracion de la tesis que la cita, con archivo y linea.

Asi la validacion se hace contra el texto literal del documento y no contra un
resumen, que es donde se colarian invenciones.

Uso: python tools/expediente_referencias.py [--lotes N] [--solo cap03 cap05 ...]
     [--prefijo nombre]

Con --solo se restringe el barrido a los capitulos indicados, para revalidar una
parte del documento sin rehacer el resto. Con --prefijo se cambia el nombre de
los lotes, de modo que una revalidacion no sobreescriba el expediente anterior.
"""
import argparse
import json
import pathlib
import re

RAIZ = pathlib.Path(__file__).resolve().parent.parent
BIB = RAIZ / "report/tesis/tesis_indice_nuevo/back/referencias.bib"
CAPS = RAIZ / "report/tesis/tesis_indice_nuevo/caps_larraga"
SALIDA = RAIZ / "revisiones/validacion_referencias"

RE_ENTRADA = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,")
RE_CITA = re.compile(r"\\[a-zA-Z]*cite[a-zA-Z]*\*?(?:\[[^]]*\])*\{([^}]+)\}")


def entradas_bib(texto: str) -> dict:
    """Devuelve {clave: texto completo de la entrada} con llaves balanceadas."""
    resultado = {}
    for m in RE_ENTRADA.finditer(texto):
        tipo, clave = m.group(1), m.group(2)
        i = texto.index("{", m.start())
        nivel, j = 0, i
        while j < len(texto):
            if texto[j] == "{":
                nivel += 1
            elif texto[j] == "}":
                nivel -= 1
                if nivel == 0:
                    break
            j += 1
        resultado[clave] = {"tipo": tipo, "bibtex": texto[m.start():j + 1]}
    return resultado


def oraciones(parrafo: str) -> list:
    """Parte un parrafo en oraciones sin romper abreviaturas ni decimales."""
    protegido = parrafo.replace("et al.", "et al<PUNTO>")
    protegido = re.sub(r"(\d)\.(\d)", r"\1<PUNTO>\2", protegido)
    piezas = re.split(r"(?<=[.;:])\s+(?=[A-ZÁÉÍÓÚÑ\\])", protegido)
    return [p.replace("<PUNTO>", ".").strip() for p in piezas if p.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lotes", type=int, default=8)
    ap.add_argument("--solo", nargs="*", default=None,
                    help="prefijos de archivo a incluir, p. ej. cap03 cap05")
    ap.add_argument("--prefijo", default="lote")
    args = ap.parse_args()

    bib = entradas_bib(BIB.read_text(encoding="utf-8"))

    usos = {clave: [] for clave in bib}
    archivos = [t for t in sorted(CAPS.rglob("*.tex")) if "standalone" not in t.name]
    if args.solo:
        archivos = [t for t in archivos if any(t.name.startswith(p) for p in args.solo)]

    for tex in archivos:
        lineas = tex.read_text(encoding="utf-8", errors="ignore").splitlines()
        # Varios capitulos vienen con las lineas cortadas a mano, asi que una
        # oracion puede abarcar varias lineas. Se reconstruye el parrafo (lo que
        # va entre lineas en blanco) antes de partir en oraciones, y se guarda la
        # linea donde arranca para poder ubicarlo en el archivo.
        parrafos, actual, inicio = [], [], 1
        for numero, linea in enumerate(lineas, 1):
            if linea.strip() and not linea.lstrip().startswith("%"):
                if not actual:
                    inicio = numero
                actual.append(linea.strip())
            elif actual:
                parrafos.append((inicio, " ".join(actual)))
                actual = []
        if actual:
            parrafos.append((inicio, " ".join(actual)))

        for numero, parrafo in parrafos:
            if not RE_CITA.search(parrafo):
                continue
            for oracion in oraciones(parrafo):
                claves = set()
                for grupo in RE_CITA.findall(oracion):
                    claves.update(c.strip() for c in grupo.split(","))
                for clave in claves:
                    if clave in usos:
                        usos[clave].append({
                            "archivo": str(tex.relative_to(CAPS)),
                            "linea": numero,
                            "afirmacion": re.sub(r"\s+", " ", oracion),
                        })

    expediente = []
    for clave in sorted(bib, key=str.lower):
        # Con --solo, una clave puede no aparecer en el tramo revisado. No se
        # incluye, porque no hay nada que validar de ella en este barrido.
        if args.solo and not usos[clave]:
            continue
        expediente.append({
            "clave": clave,
            "tipo": bib[clave]["tipo"],
            "bibtex": bib[clave]["bibtex"],
            "afirmaciones": usos[clave],
            "veces_citada": len(usos[clave]),
        })

    SALIDA.mkdir(parents=True, exist_ok=True)
    (SALIDA / f"expediente_{args.prefijo}.json").write_text(
        json.dumps(expediente, ensure_ascii=False, indent=2), encoding="utf-8")

    # Reparte en lotes equilibrados por numero de afirmaciones, no por numero de
    # entradas, para que ningun agente reciba una carga desproporcionada.
    lotes = [[] for _ in range(args.lotes)]
    carga = [0] * args.lotes
    for ref in sorted(expediente, key=lambda r: -r["veces_citada"]):
        i = carga.index(min(carga))
        lotes[i].append(ref)
        carga[i] += max(ref["veces_citada"], 1)

    for i, lote in enumerate(lotes, 1):
        if not lote:
            continue
        ruta = SALIDA / f"{args.prefijo}_{i:02d}.md"
        with ruta.open("w", encoding="utf-8") as f:
            f.write(f"# {args.prefijo} {i} de {args.lotes}: {len(lote)} referencias\n\n")
            for ref in sorted(lote, key=lambda r: r["clave"].lower()):
                f.write(f"## {ref['clave']}\n\n")
                f.write("### Entrada bibliografica tal como esta en referencias.bib\n\n")
                f.write("```bibtex\n" + ref["bibtex"] + "\n```\n\n")
                if ref["afirmaciones"]:
                    f.write(f"### Afirmaciones que la tesis le atribuye ({len(ref['afirmaciones'])})\n\n")
                    for k, a in enumerate(ref["afirmaciones"], 1):
                        f.write(f"{k}. `{a['archivo']}:{a['linea']}`\n")
                        f.write(f"   > {a['afirmacion']}\n\n")
                else:
                    f.write("### Afirmaciones\n\nNinguna: no se cita en el cuerpo.\n\n")
                f.write("---\n\n")

    total_af = sum(r["veces_citada"] for r in expediente)
    print(f"referencias: {len(expediente)}")
    print(f"afirmaciones atribuidas: {total_af}")
    print(f"lotes escritos en: {SALIDA}")
    for i, lote in enumerate(lotes, 1):
        print(f"  {args.prefijo}_{i:02d}.md  {len(lote):2d} refs, {carga[i-1]:3d} afirmaciones")


if __name__ == "__main__":
    main()
