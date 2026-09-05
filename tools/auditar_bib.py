"""Compara las entradas de referencias.bib contra las que realmente cita la tesis.

Uso: python tools/auditar_bib.py
"""
import pathlib
import re

RAIZ = pathlib.Path(__file__).resolve().parent.parent
BIB = RAIZ / "report/tesis/tesis_indice_nuevo/back/referencias.bib"
BBL = RAIZ / "report/tesis/tesis_indice_nuevo/caps_larraga/main.bbl"
CAPS = RAIZ / "report/tesis/tesis_indice_nuevo/caps_larraga"


def main() -> None:
    bib = BIB.read_text(encoding="utf-8")
    en_bib = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib))

    bbl = BBL.read_text(encoding="utf-8")
    impresas = set(re.findall(r"\\entry\{([^}]+)\}", bbl))

    citadas_tex = set()
    for tex in CAPS.rglob("*.tex"):
        if "standalone" in tex.name:
            continue
        texto = tex.read_text(encoding="utf-8", errors="ignore")
        for grupo in re.findall(r"\\[a-zA-Z]*cite[a-zA-Z]*\*?(?:\[[^]]*\])*\{([^}]+)\}", texto):
            citadas_tex.update(c.strip() for c in grupo.split(","))

    print(f"entradas en referencias.bib:        {len(en_bib)}")
    print(f"entradas en la bibliografia final:  {len(impresas)}")
    print(f"claves citadas en los .tex:         {len(citadas_tex)}")
    print()

    huerfanas = sorted(en_bib - impresas)
    print(f"HUERFANAS (en el .bib y nunca citadas): {len(huerfanas)}")
    for clave in huerfanas:
        print(f"   - {clave}")

    fantasma = sorted(citadas_tex - en_bib)
    print(f"\nFANTASMA (citadas sin entrada en el .bib): {len(fantasma)}")
    for clave in fantasma:
        print(f"   - {clave}")

    duplicadas = [c for c in set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib))
                  if len(re.findall(r"@\w+\s*\{\s*" + re.escape(c) + r"\s*,", bib)) > 1]
    print(f"\nCLAVES DUPLICADAS en el .bib: {len(duplicadas)}")
    for clave in sorted(duplicadas):
        print(f"   - {clave}")


if __name__ == "__main__":
    main()
