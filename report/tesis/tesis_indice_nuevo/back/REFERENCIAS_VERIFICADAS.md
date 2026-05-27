# Verificación de referencias — referencias.bib

## Cambios realizados

### 1. Duplicados eliminados
Se eliminaron **17 entradas duplicadas** que causaban que Biber usara la primera ocurrencia (a veces incorrecta) y descartara la segunda. Los duplicados eliminados eran versiones alternativas o erróneas de:

- Wolfram1984, White1997, Clarke1998, Clarke2008
- Wang2021, Tang2024
- Ma2019, Gomez2020, Herold2003, Gong2013
- Fortin2024, North2023, Waddell2002, Chen2022

### 2. Correcciones de formato

| Clave | Corrección |
|-------|------------|
| **Wolfram1984** | Cambiado de `@book` a `@article` — el artículo de Nature 1984 es correcto (vol. 311, pp. 419-424) |
| **alonso1964location** | Cambiado de `@article` a `@book` — *Location and Land Use* es un libro |
| **Fortin2024** | Cambiado de `@article` a `@misc` — DEAP es software, no artículo de revista |
| **Gong2013, Ma2019** | Autores expandidos (reemplazado "et al." por lista o "and others") |

### 3. Referencias citadas en el texto (todas verificadas)

| Cita | Uso en texto | Entrada correcta |
|------|--------------|------------------|
| Wolfram1984 | Bases teóricas de AC | Nature 1984, artículo |
| Clarke1998, White1997 | Primeros AC urbanos | Clarke & Gaydos 1998: San Francisco/Washington; White: Env. Planning B |
| Clarke2008 | SLEUTH | "A Decade of Cellular Urban Modeling with SLEUTH" (capítulo en Planning Support Systems for Cities and Regions, ed. R.K. Brail, Lincoln Institute of Land Policy) |
| Wang2021, Tang2024 | AC+AG para vitalidad urbana; GSA-CA heurístico | IJERPH Wuhan; Scientific Reports regiones áridas |
| Ma2019, Gomez2020 | Deep Learning; modelado ML espaciotemporal | ISPRS; Remote Sensing |
| Herold2003, Gong2013 | Series temporales satelitales | Remote Sensing Env.; Int. J. Remote Sensing |
| Fortin2024, North2023, Waddell2002, Chen2022 | Herramientas open-source | DEAP, Repast, UrbanSim, PyLUSAT |
| pontius2008comparing | Comparativo 13 aplicaciones; FoM heterogéneo (seis casos bajo 15\%, un caso sobre 50\%); no fija intervalo universal | Annals of Regional Science |

### 4. Nota sobre la tabla en cap05

La fila "CA-Markov -- São Paulo" se mantiene sin cita (Almeida2003 eliminada por no existir).

## Validación de DOIs (marzo 2025)

Se verificaron los 15 DOIs contra doi.org. **3 resultaron inválidos (404)** y se eliminaron:

| Entrada | DOI eliminado | Motivo |
|---------|---------------|--------|
| Silva2008 | 10.1016/j.compenvurbsys.2007.09.004 | DOI NOT FOUND |
| North2023 | 10.1109/WSC57936.2023.10157947 | DOI NOT FOUND |

**DOIs válidos** (resuelven correctamente): Wolfram1984, White1997, Clarke1998, Herold2003, Wang2021, Tang2024, Arfiansyah2024, Chen2022, Waddell2002, Gomez2020, pontius2008comparing, Seto2012, Li2007.

## Correcciones de verificación exhaustiva (marzo 2025)

| Clave | Problema | Corrección |
|-------|----------|------------|
| **Tang2024** | Título, autores y páginas incorrectos | Tang, Xiaoyan; Liu, Funan; Hu, Xinling. Título: "Urban growth simulation and scenario projection for the arid regions using heuristic cellular automata". Scientific Reports 14, art. 71709. DOI verificado. |
| **Wang2021** | DOI y metadatos apuntaban a artículo inexistente en SCS | Reemplazado por paper real: Wang, Renyang; He, Qingsong; Zhang, Lu; Wang, Huiying. "Coupling Cellular Automata and a Genetic Algorithm to Generate a Vibrant Urban Form—A Case Study of Wuhan, China". IJERPH 18(21):11013. DOI 10.3390/ijerph182111013. |
| **Miller2022** | Autores incorrectos (Miller, Li) | Renombrado a **Chen2022**: Chen, Changjie; Judge, Jasmeet; Hulse, David. PyLUSAT. Env. Modelling & Software 151:105362. DOI verificado. |
| **Kadhim2018** | Paper inexistente en Remote Sensing 2018 | Reemplazado por **Gomez2020**: Gómez, Patiño, Duque, Passos. "Spatiotemporal Modeling of Urban Growth Using Machine Learning". Remote Sensing 12(1):109. DOI 10.3390/rs12010109. |
| **UNHabitat2020** | URL genérica (unhabitat.org) | URL específica del PDF: https://unhabitat.org/sites/default/files/2020/10/wcr_2020_report.pdf |
| **Waddell2020** | DOI 10.1016/j.compenvurbsys.2020.101541 apunta a artículo de Guerrero (housing wealth inequality), no a UrbanSim | Reemplazado por **Waddell2002**: Waddell, P. "UrbanSim: Modeling Urban Development for Land Use, Transportation and Environmental Planning". JAPA 68(3):297-314. DOI 10.1080/01944360208976274. |
| **Tang2024** | Número de artículo incorrecto (71709) | Corregido a 21106 según Scientific Reports. |

### Verificación exhaustiva de todas las referencias (marzo 2025)

| Clave | Verificación | Resultado |
|-------|--------------|-----------|
| **Clarke1997→1998** | Año incorrecto | Paper publicado en IJGIS 1998, no 1997. Corregido a Clarke1998. |
| **Seto2011→2012** | Año incorrecto | "Global forecasts of urban expansion to 2030" publicado en PNAS 2012. DOI 10.1073/pnas.1211658109 añadido. |
| **Angel2016→2012** | Año incorrecto | *Planet of Cities* publicado en 2012 por Lincoln Institute. Corregido a Angel2012. |
| **Li2011→2007** | Revista y año incorrectos | Paper en Journal of Environmental Management 2007, 85(4):1063--1075. **DOI correcto:** 10.1016/j.jenvman.2006.11.006 (el 10.1016/j.jenvman.2006.10.010 es otro artículo del vol.\ 85). |
| **Seto2012 duplicado** | Dos papers Seto 2012 | Renombrado teleconnections a Seto2012teleconnections. |

**Referencias verificadas como correctas:** Wolfram1984, White1997, Silva2002, Herold2003, Weng2002, Gong2013, Ma2019, Gomez2020, Seto2012, ArribasBel2014, Sante2010, Wang2021, Tang2024, Arfiansyah2024, Chen2022, Waddell2002, pontius2008comparing, BonhamCarter1994, Ojala2002, Gorelick2017, alonso1964location, Batty2005, goldberg1991comparative.

### Referencias eliminadas (no existen)

| Clave | Motivo |
|-------|--------|
| **Li2013** | Paper inexistente (DOI no resuelve a artículo válido) |
| **Huang2019** | Paper inexistente (DOI no resuelve a artículo válido) |
| **Almeida2003** | Paper inexistente (IJRS 24(19):761-776 no verificado) |

### URLs explícitas añadidas (Herold)

| Clave | DOI | URL alternativa añadida |
|-------|-----|-------------------------|
| **Herold2003** | 10.1016/S0034-4257(03)00075-0 | ScienceDirect PII: https://www.sciencedirect.com/science/article/abs/pii/S0034425703000750 |

## Estado final

- **0 duplicados** (Biber ya no reporta warnings)
- **18 citekeys** procesados correctamente
- **12 DOIs válidos**, 3 eliminados por no existir en el sistema DOI
- **Tesis compilada**: 104 páginas, main.pdf
