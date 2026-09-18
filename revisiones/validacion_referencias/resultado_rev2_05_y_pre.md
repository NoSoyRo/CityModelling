# Resultados de los lotes rev2_05, rev2pre_01, rev2pre_02 y rev2pre_03

Los cuatro agentes de estos lotes corrieron en modo de solo lectura y no pudieron
escribir su archivo. Aquí queda el registro de lo que reportaron y de qué se hizo
con cada hallazgo. El detalle completo de cada validación vive en el transcript de
cada agente; esto es el acta de disposición.

## Cobertura

| Lote | Alcance | Refs | Afirmaciones | Correctas | Con observación |
|---|---|---|---|---|---|
| rev2_05 | cap03 (Tang, Siddiqi, Dinamica, SLEUTH) | 12 | 30 | 25 | 5 |
| rev2pre_01 | cap01, cap02 (lote A) | 18 | 26 | 22 | 4 |
| rev2pre_02 | cap01, cap02 (lote B) | 18 | 25 | 20 | 5 |
| rev2pre_03 | cap01, cap02 (lote C) | 18 | 23 | 23 | 0 |

Ninguno de los cuatro lotes encontró referencias inexistentes.

## Hallazgos y disposición

### Corregidos en el texto

| # | Etiqueta | Dónde | Qué decía | Qué dice ahora |
|---|---|---|---|---|
| 1 | INCORRECTA | cap03, Tang2024 | El relieve y el agua se incorporan «como capas de exclusión explícitas» | La elevación es uno de los nueve factores de conducción; solo el agua es restricción excluida |
| 2 | INCORRECTA | cap02, Lloyd1982 | El esquema por lotes lo «formalizó después» Lloyd | Lloyd lo formuló en 1957, una década antes de MacQueen; solo se publicó en 1982 |
| 3 | NO VERIFICABLE | cap02, exceso de rojo | Atribuido a Meyer et al. (1998), *Transactions of the ASAE*, que trata de textura | Atribuido a Meyer, Hindman y Laksmi (1999), SPIE 3543:327–335, donde aparece la formulación 1,4R−G que usa el código |
| 4 | SOBREEXTENDIDA | cap01, pontius2008comparing | Pontius sostiene que la validación en ventanas desplazadas detecta sobreajuste | Pontius documenta que el FoM sube con el cambio neto observado; la inferencia sobre ventanas desplazadas se presenta como propia |
| 5 | SOBREEXTENDIDA | cap01, ONUHabitat2016 | Cocitado para cifras que SEDATU publicó en 2023 y 2024 | Retirado de esa cita; reubicado donde sí sostiene la afirmación (patrón de crecimiento disperso) |
| 6 | SOBREEXTENDIDA | cap02, Herold2003 | Enunciaba una jerarquía entre resoluciones que el artículo no plantea | Describe el caso de Santa Barbara: setenta y dos años de fotografía aérea e IKONOS |
| 7 | SOBREEXTENDIDA | cap02, Wolfram1984 | «en el sentido formalizado por» | «los caracteriza como»; la formalización viene de von Neumann y Ulam, y Wolfram1984 es unidimensional |
| 8 | SOBREEXTENDIDA | cap02, Silva2002 | El acrónimo «se fijó más tarde con» | «queda definido explícitamente en»; el artículo lo presenta como nombre ya vigente |
| 9 | NO VERIFICABLE | cap02, White1993 | Los AC probabilísticos urbanos «desde White1993» | «la línea de modelos urbanos que abre el autómata restringido de»; el término de perturbación estocástica es de White, Engelen y Uljee (1997) |
| 10 | SOBREEXTENDIDA | cap03, ClarkeHoppen1997 | La dependencia de trayectoria como afirmación del artículo | Se cita lo que el artículo sí dice (el autómata se adapta a las circunstancias que genera) y la consecuencia se presenta aparte |
| 11 | SOBREEXTENDIDA | cap03, JantzGoetz2005 | «demuestran que un conjunto muy ajustado a una escala no se transfiere a otra» | Se añade el resultado que faltaba: la tasa de crecimiento sí se reproduce en todas las resoluciones; lo que varía es el patrón y la sensibilidad de las reglas |
| 12 | SOBREEXTENDIDA | cap03, Dietzel2007 | La discusión sobre calibración óptima «permanece abierta» y «se repite en cada trabajo reportado» | Dietzel propone una métrica combinada; se elimina el cuantificador universal |
| 13 | SOBREEXTENDIDA | cap03, Dietzel2007 | La divergencia de proyecciones a largo plazo | Atribuida a Beven2006; Dietzel queda para la multiplicidad de combinaciones equivalentes |
| 14 | NO VERIFICABLE | cap03, Silva2002 | El intervalo $[0,100]$ y las $10^{10}$ combinaciones | Atribuidos a Clarke2008 y a las implementaciones de referencia; Silva no los enuncia |
| 15 | SOBREEXTENDIDA | cap03, Legendre1993 | La sobreestimación del desempeño en validación | Legendre queda para la violación del supuesto de independencia en pruebas estadísticas; Brenning2012 para la validación de modelos predictivos |
| 16 | SOBREEXTENDIDA | cap03, BonhamCarter1989 | «cartografía asistida por computadora» | «sistemas de información geográfica», que es el término del artículo |
| 17 | INCORRECTA | cap03, licenciamiento | «la mayoría se distribuye como software de interfaz gráfica y código cerrado», sin fuente | «software gratuito de interfaz gráfica cuyo código fuente no se publica»; CLUE es freeware, y el argumento de la tesis es sobre apertura del código, no sobre precio |
| 18 | SOBREEXTENDIDA | cap01 y cap03, Wahyudi2016 | «América Latina» | «América del Sur», que es lo que dice la revisión |

### Confirmados contra la fuente primaria, sin cambio

Estos son los datos de mayor riesgo del bloque. Los cuatro agentes los verificaron
con cita textual:

- La semana de cómputo en el clúster Beowulf de dieciséis nodos de Jantz et al.
  («Over a week of processing time was required to complete the calibration»).
- La tasa media anual del 2,8 % para la ZMQ, verificada en el Cuadro 5a del propio
  documento de CONAPO.
- El millón y medio de habitantes de la ZMQ en el censo 2020: se cumple en las dos
  delimitaciones vigentes (1 530 820 con cuatro municipios, 1 594 212 con Apaseo el Alto).
- La escala de cinco tramos del Information Value de Siddiqi, tramo por tramo y
  etiqueta por etiqueta, incluido el «suspicious» del tramo superior.
- Las vecindades de Moore y Von Neumann en el artículo bidimensional de Packard y
  Wolfram (1985), no en el unidimensional de 1984.
- El ISBN del informe de ONU-Hábitat, leído de la página de créditos del PDF.
- La definición de Kappa de Cohen (1960) y la escala de Landis y Koch, correctamente
  separadas.
- El margen máximo y el margen blando de Cortes y Vapnik (1995).
- El HSV atribuido a Smith (1978) y no al manual de González y Woods.
- El 2G−R−B de Woebbecke et al. (1995).

### Observaciones de higiene que se dejaron como están

- La clave `Jantz2003` tiene `year = 2004`, que es el año correcto. La salida impresa
  es correcta; renombrar la clave es cosmético.
- La clave `CONABIO2024Porque` tiene `year = 2022`. Mismo caso.
- GEOSCAN fecha el GSC Paper 89-9 en 1990. La cita convencional como 1989 es la
  estándar y es la que usan las fuentes que lo citan.
- La cláusula «formulado originalmente para el diagnóstico médico» del WoE no lleva
  cita propia; el `\parencite` cubre la parte minera. Si un sinodal lo pregunta, la
  referencia habitual es Spiegelhalter y Knill-Jones (1984).

## Estado tras aplicar todo

Compilación limpia, 135 páginas, sin errores de LaTeX ni citas indefinidas.
82 entradas en el `.bib`, 82 impresas en la bibliografía, 82 citadas en el cuerpo:
cero huérfanas, cero fantasma, cero duplicadas. Los revisores de ortografía y
gramática no devuelven nada.
