"""Revisión gramatical del texto de la tesis.

Complementa a revisar_ortografia.py, que solo cubre dedazos, tildes y nueve
patrones. Aquí se buscan los errores frecuentes de la prosa académica en
español: dequeísmo, queísmo, régimen preposicional, concordancia, gerundio de
posterioridad y puntuación.

Uso:
    python tools/revisar_gramatica.py
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

CAPS = Path(__file__).resolve().parents[1] / "report/tesis/tesis_indice_nuevo/caps_larraga"

# Entornos y comandos que no son prosa y que ensucian la búsqueda.
LIMPIEZA = [
    (r"(?m)^\s*%.*$", " "),
    (r"\\begin\{(equation|align|figure|table|tabular|tabularx|lstlisting|verbatim|algorithm|algorithmic)\*?\}.*?\\end\{\1\*?\}", " "),
    (r"\$[^$]*\$", " NUM "),
    (r"\\(paren|text|foot|auto)cite\{[^}]*\}", " CITA "),
    (r"\\(ref|label|eqref|cref)\{[^}]*\}", " REF "),
    (r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " "),
    (r"[{}]", " "),
]

# Verbos que NO rigen «de que» (dequeísmo) y fórmulas que sí lo exigen (queísmo).
REGLAS = [
    # --- dequeísmo ---
    (r"\b(pienso|piensa|pensamos|creo|cree|creemos|considera|consideran|consideramos|"
     r"resulta|resultan|opina|opinan|sugiere|sugieren|indica|indican|muestra|muestran|"
     r"demuestra|demuestran|señala|señalan|afirma|afirman|supone|suponen|permite|permiten|"
     r"hace|hacen|es|son|parece|parecen)\s+de\s+que\b",
     "dequeísmo: ese verbo no rige «de que»"),
    (r"\b(dado|puesto|visto)\s+de\s+que\b", "dequeísmo: «dado que», no «dado de que»"),

    # --- queísmo ---
    (r"\b(a\s+pesar|a\s+partir|antes|despu[eé]s|adem[aá]s|en\s+caso|en\s+el\s+caso|"
     r"con\s+el\s+fin|a\s+fin|en\s+lugar|en\s+vez|el\s+hecho|la\s+idea|la\s+ventaja|"
     r"la\s+posibilidad|la\s+raz[oó]n|la\s+conclusi[oó]n)\s+que\b",
     "queísmo: falta «de» antes de «que»"),
    (r"\b(darse\s+cuenta|me\s+di\s+cuenta|se\s+dio\s+cuenta|estar\s+seguro|"
     r"depende|dependen|consta|informa|informan|convencer)\s+que\b",
     "queísmo: ese verbo rige «de que»"),

    # --- régimen preposicional ---
    (r"\ben\s+base\s+a\b", "«en base a» -> «con base en» o «a partir de»"),
    (r"\ben\s+relaci[oó]n\s+a\b", "«en relación a» -> «en relación con» o «con relación a»"),
    (r"\bde\s+acuerdo\s+a\b", "«de acuerdo a» -> «de acuerdo con»"),
    (r"\ba\s+nivel\s+de\b", "«a nivel de»: úsese solo con sentido de altura o jerarquía"),
    (r"\bcon\s+respecto\s+de\b", "«con respecto de» -> «con respecto a» o «respecto de»"),
    (r"\ben\s+torno\s+a\s+que\b", "revisar «en torno a que»"),
    (r"\bbajo\s+(el\s+)?(punto\s+de\s+vista|la\s+base)\b", "«bajo el punto de vista» -> «desde el punto de vista»"),
    (r"\bdiferente\s+a\b", "preferible «diferente de» o «distinto de»"),

    # --- concordancia y duplicaciones ---
    (r"\b(el|un)\s+\w+a\s+(es|son)\b(?!\w)", None),  # desactivada: demasiados falsos positivos
    (r"\b(de|que|la|el|los|las|en|y|a|se|por|con|su|un|una)\s+\1\b", "palabra duplicada"),
    (r"\buna\s+serie\s+de\s+\w+\s+(es|fue|ha)\b", "concordancia: «una serie de X» suele concordar en plural"),

    # --- gerundio de posterioridad ---
    (r",\s*(obteniendo|logrando|resultando|generando|produciendo|permitiendo|dando\s+lugar)\b",
     "posible gerundio de posterioridad: revisar si expresa consecuencia"),

    # --- puntuación ---
    # Las reglas de espaciado se omiten a propósito: sobre fuente LaTeX, la
    # propia limpieza de comandos introduce espacios y genera falsos positivos.
    (r"\betc\.\s*\.", "«etc.» con punto duplicado"),
    (r"\.{4,}", "puntos suspensivos mal formados"),

    # --- calcos y muletillas del inglés ---
    (r"\bjugar\s+un\s+(papel|rol)\b", "calco: «desempeñar un papel»"),
    (r"\bel\s+mismo\s+(es|fue|se)\b", "«el mismo» como pronombre: preferible repetir el sustantivo"),
    (r"\bmayormente\b", "anglicismo de registro: preferible «sobre todo» o «en su mayoría»"),
    (r"\badicionalmente\b", "preferible «además»"),
]


def limpiar(texto: str) -> str:
    for patron, reemplazo in LIMPIEZA:
        texto = re.sub(patron, reemplazo, texto, flags=re.DOTALL)
    return texto


def main() -> None:
    archivos = sorted(
        p for p in CAPS.rglob("*.tex")
        if "standalone" not in p.name and "preamble" not in p.name
    )
    total = 0
    for tex in archivos:
        crudo = tex.read_text(encoding="utf-8", errors="ignore")
        for numero, linea in enumerate(crudo.splitlines(), 1):
            prosa = limpiar(linea)
            for patron, nota in REGLAS:
                if nota is None:
                    continue
                for m in re.finditer(patron, prosa, re.IGNORECASE):
                    ini, fin = max(0, m.start() - 70), m.end() + 70
                    ctx = re.sub(r"\s+", " ", prosa[ini:fin]).strip()
                    print(f"{tex.relative_to(CAPS)}:{numero}  {nota}")
                    print(f"    ...{ctx}...\n")
                    total += 1
    print(f"total de hallazgos gramaticales: {total}")


if __name__ == "__main__":
    main()
