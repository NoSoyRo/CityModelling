# Resumen tesis_indice_nuevo — Revisión capítulo por capítulo

## Estructura actual

```
tesis_indice_nuevo/
├── back/
│   └── referencias.bib
└── caps_larraga/
    ├── main.tex
    ├── cap01_introduccion.tex ... cap06_conclusiones.tex
    ├── figures/
    │   ├── ac_simulation_quinquenal.png
    │   ├── architecture.png
    │   ├── binary_transformation_1984.png, 2000.png, 2020.png
    │   ├── temporal_quality_distribution.png
    │   ├── urban_growth_timeline.png
    │   ├── validation_metrics.png
    │   ├── woe_weights.png
    │   └── comparisons/
    │       └── quinquenal_2011_to_2012.png ... quinquenal_2015_to_2016.png
    └── thesis-commands.sty
```

## Capítulos y referencias

| Cap | Archivo | Figuras usadas |
|-----|---------|----------------|
| 1 | cap01_introduccion.tex | Ninguna |
| 2 | cap02_marco_teorico.tex | Ninguna |
| 3 | cap03_area_estudio.tex | temporal_quality_distribution, architecture, binary_transformation_1984/2000/2020 |
| 4 | cap04_modelo_crecimiento_urbano.tex | Ninguna |
| 5 | cap05_resultados_analisis.tex | urban_growth_timeline, woe_weights, quinquenal_* (5), ac_simulation_quinquenal, validation_metrics |
| 6 | cap06_conclusiones.tex | Ninguna |

## Scripts del proyecto (raíz)

| Script | Uso |
|--------|-----|
| generate_cap05_figures.py | Genera todas las figuras del cap 5 para la tesis |
| generate_quinquenal_summary_maps.py | Genera mapas resumen (inicio\|obs\|pred\|diff) desde resultados existentes |
| process_imagen_2015.py | Convierte imagen_2015.png → 2015.npy (ya ejecutado) |
| run_all_quinquenal_validations.py | Ejecuta las 5 validaciones quinquenales |
| validate_quinquenal_2011_2016.py | Validación única 2011→2016 |
| train_woe_1984_2010.py | Entrena WoE 1984-2010 |
| train_woe_pooled.py | Entrena WoE pooled |

## Compilación

```bash
cd report/tesis/tesis_indice_nuevo/caps_larraga
pdflatex main.tex && biber main && pdflatex main.tex && pdflatex main.tex
```

## Limpieza realizada

- Archivos LaTeX aux (.aux, .bbl, .log, etc.) eliminados de caps_larraga (se regeneran al compilar)
- Scripts en raíz: todos en uso (generate_cap05, generate_quinquenal_summary_maps, process_imagen_2015, run_all, validate_quinquenal, train_woe_*)
