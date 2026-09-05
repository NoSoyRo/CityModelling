"""Detecta dedazos y tildes faltantes en la tesis, sin diccionario externo.

Se apoya en la consistencia del propio documento y solo reporta dos clases de
error, que son las de alta precision:

1. Transposiciones: dos letras contiguas intercambiadas respecto de una palabra
   que el texto usa varias veces. Asi se cazan "crecimeinto", "fucnion" o
   "requerimeintos". La flexion del espanol no produce transposiciones, asi que
   casi no hay falsos positivos.

2. Tildes faltantes: la palabra aparece sin tilde una o dos veces y con tilde
   muchas mas. Asi se caza "segun" frente a "segun" acentuado, o "analisis".

Deliberadamente NO reporta singular/plural ni conjugaciones, que es donde un
diccionario generico se llena de ruido.

Uso: python tools/revisar_ortografia.py
"""
import pathlib
import re
import unicodedata
from collections import Counter, defaultdict

RAIZ = pathlib.Path(__file__).resolve().parent.parent
CAPS = RAIZ / "report/tesis/tesis_indice_nuevo/caps_larraga"

LARGO_MINIMO = 5
FRECUENCIA_CONFIABLE = 3

PATRONES_LATEX = [
    (r"\\begin\{(equation|align|figure|table|longtable|tabular|algorithm|lstlisting|verbatim)\*?\}.*?\\end\{\1\*?\}", " "),
    (r"\$[^$]*\$", " "),
    # Cualquier variante de cita: \cite, \parencite, \textcite, \nocite...
    (r"\\[a-zA-Z]*cite[a-zA-Z]*\*?(\[[^]]*\])*\{[^}]*\}", " "),
    # Comandos cuyo argumento es un identificador, no prosa. Admiten argumento
    # opcional, como \includegraphics[width=...]{archivo}.
    (r"\\(ref|label|autoref|eqref|includegraphics|input|include|texttt|url|href|path|lstinline|caption\*)(\[[^]]*\])?\{[^}]*\}", " "),
    (r"\\[a-zA-Z@]+\*?", " "),
    (r"[{}\[\]]", " "),
    (r"%.*", " "),
]

# Palabras inglesas y nombres propios que conviven con formas espanolas parecidas.
IGNORAR = {
    "annual", "abstract", "abstracto", "urban", "growth", "value", "change",
    "model", "models", "sprawl", "merit", "figure", "modelling", "modeling",
    "index", "spatial", "cellular", "automata", "weights", "evidence",
    # Nombres propios de software que se escriben sin tilde.
    "dinamica", "idrisi", "selva", "terrset", "liberagis", "christaller",
}

ERRORES_GRAMATICALES = [
    (r"\bhan\s+habido\b", "«han habido» -> «ha habido»: haber impersonal no va en plural"),
    (r"\bhabian\s+habido\b", "«habían habido» -> «había habido»"),
    (r"\bhubieron\s+(muchos|varios|algunos|problemas|trabajos)\b", "«hubieron» impersonal -> «hubo»"),
    (r"\bdetras\s+de\s+que\b", "revisar: «detrás de que»"),
    (r"\ben\s+base\s+a\b", "«en base a» -> «con base en»"),
    (r"\bdebido\s+a\s+que\s+que\b", "«que» duplicado"),
    (r"\bde\s+de\b", "«de» duplicado"),
    (r"\bla\s+la\b|\bel\s+el\b|\blos\s+los\b|\blas\s+las\b", "articulo duplicado"),
    (r"\bque\s+que\b", "«que» duplicado"),
]


def limpiar(texto: str) -> str:
    for patron, reemplazo in PATRONES_LATEX:
        texto = re.sub(patron, reemplazo, texto, flags=re.DOTALL)
    return texto


def sin_tilde(palabra: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", palabra)
                   if unicodedata.category(c) != "Mn")


def transposiciones(palabra: str) -> set:
    return {palabra[:i] + palabra[i + 1] + palabra[i] + palabra[i + 2:]
            for i in range(len(palabra) - 1)
            if palabra[i] != palabra[i + 1]}


def main() -> None:
    archivos = [t for t in sorted(CAPS.rglob("*.tex")) if "standalone" not in t.name]
    frecuencia = Counter()
    apariciones = defaultdict(list)

    for tex in archivos:
        for numero, linea in enumerate(tex.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            for palabra in re.findall(r"[a-záéíóúüñA-ZÁÉÍÓÚÜÑ]+", limpiar(linea)):
                if len(palabra) < LARGO_MINIMO:
                    continue
                baja = palabra.lower()
                frecuencia[baja] += 1
                apariciones[baja].append((tex.relative_to(CAPS), numero, palabra))

    confiables = {p for p, n in frecuencia.items() if n >= FRECUENCIA_CONFIABLE}
    planas_confiables = defaultdict(list)
    for palabra in confiables:
        planas_confiables[sin_tilde(palabra)].append(palabra)

    print("=== 1. TRANSPOSICIONES DE LETRAS (dedazos) ===\n")
    dedazos = 0
    for palabra, n in sorted(frecuencia.items()):
        if n > 2 or palabra in confiables or palabra in IGNORAR:
            continue
        plana = sin_tilde(palabra)
        objetivos = set()
        for giro in transposiciones(plana):
            objetivos.update(planas_confiables.get(giro, []))
        objetivos = {o for o in objetivos if sin_tilde(o) != plana}
        if not objetivos:
            continue
        mejor = max(objetivos, key=lambda o: frecuencia[o])
        ruta, numero, original = apariciones[palabra][0]
        print(f"{ruta}:{numero}")
        print(f"    {original}  ->  {mejor}   (el texto usa «{mejor}» {frecuencia[mejor]} veces)")
        dedazos += 1
    if not dedazos:
        print("ninguno")

    print(f"\ntotal de dedazos: {dedazos}")

    print("\n\n=== 2. TILDES FALTANTES ===\n")
    faltantes = 0
    for palabra, n in sorted(frecuencia.items()):
        if palabra in IGNORAR or sin_tilde(palabra) != palabra:
            continue  # ya lleva tilde o es irrelevante
        hermanas = [h for h in planas_confiables.get(palabra, []) if h != palabra]
        if not hermanas:
            continue
        mejor = max(hermanas, key=lambda h: frecuencia[h])
        if frecuencia[mejor] < 3 * max(n, 1):
            continue  # la forma sin tilde no es claramente minoritaria
        for ruta, numero, original in apariciones[palabra]:
            print(f"{ruta}:{numero}")
            print(f"    {original}  ->  {mejor}   (el texto usa «{mejor}» {frecuencia[mejor]} veces)")
            faltantes += 1
    if not faltantes:
        print("ninguna")

    print(f"\ntotal de tildes faltantes: {faltantes}")

    print("\n\n=== 3. ERRORES GRAMATICALES CONOCIDOS ===\n")
    gramatica = 0
    for tex in archivos:
        for numero, linea in enumerate(tex.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            plana = sin_tilde(limpiar(linea)).lower()
            for patron, nota in ERRORES_GRAMATICALES:
                if re.search(patron, plana):
                    print(f"{tex.relative_to(CAPS)}:{numero}  {nota}")
                    gramatica += 1
    if not gramatica:
        print("ninguno")

    print(f"\ntotal gramatical: {gramatica}")


if __name__ == "__main__":
    main()
