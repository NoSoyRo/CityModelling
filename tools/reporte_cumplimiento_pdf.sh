#!/usr/bin/env bash
# Genera el PDF del reporte de cumplimiento a partir de su markdown.
set -euo pipefail

raiz="$(cd "$(dirname "$0")/.." && pwd)"
md="$raiz/revisiones/CUMPLIMIENTO_SESION_DRA_LARRAGA.md"
salida="$raiz/revisiones/CUMPLIMIENTO_SESION_DRA_LARRAGA.pdf"
cabecera="$raiz/revisiones/.reporte_header.tex"

# El primer encabezado del markdown pasa a ser el título del PDF, no una sección.
cuerpo="$(mktemp -t cumplimiento_cuerpo)"
trap 'rm -f "$cuerpo"' EXIT
tail -n +2 "$md" > "$cuerpo"

pandoc "$cuerpo" -o "$salida" \
  --from=markdown \
  --pdf-engine=xelatex \
  --toc --toc-depth=2 \
  --metadata title="Cumplimiento de la sesión con la Dra. Lárraga" \
  --metadata author="José Rodrigo Moreno López" \
  --metadata date="23 de agosto de 2026" \
  --include-in-header="$cabecera" \
  -V lang=es \
  -V geometry:margin=2.3cm \
  -V fontsize=11pt \
  -V colorlinks=true -V linkcolor=black -V urlcolor=black \
  -V mainfont="Palatino" \
  -V monofont="Menlo" -V monofontoptions="Scale=0.85"

echo "Generado: $salida"
