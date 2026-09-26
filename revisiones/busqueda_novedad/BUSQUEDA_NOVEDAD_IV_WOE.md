# Búsqueda adversarial de antecedentes: WoE ponderado por Information Value

Fecha: 18 de septiembre de 2026
Estado: cerrada. Quedan pendientes las bases sin acceso listadas al final.

## 1. Qué se puso a prueba

La regla de transición de la tesis no combina los pesos de evidencia con la suma
simple del WoE clásico. Usa una suma ponderada por el Information Value
normalizado de cada variable:

```
S(x) = suma_k [ IV_k / suma_j IV_j ] * W_k(b_k(x))
P(x) = sigmoide(S(x))
```

El código está en `validate_quinquenal_2011_2016.py:209-221`. Los pesos vigentes
están en `data/processed/quinquenal_best_config.json`, campo `iv_weights`.

La pregunta no era si el acoplamiento WoE con autómatas celulares es nuevo. No lo
es, y la tesis nunca lo afirmó. La pregunta era si alguien ya había usado el
Information Value como ponderador explícito de los pesos de evidencia dentro de
la regla de transición.

## 2. Cómo se buscó

La búsqueda se planteó como falsación, no como confirmación. La consigna de cada
agente fue encontrar un antecedente que **destruyera** la reclamación de novedad,
no uno que la apoyara. Se abrieron cinco frentes independientes, cada uno con una
estrategia distinta, para que un falso negativo en uno no contaminara al resto.

| Frente | Estrategia | Por qué |
|---|---|---|
| 1 | Término literal: WoE junto con Information Value | El camino obvio. Cubre deslizamientos, prospección mineral y credit scoring |
| 2 | Literatura de cambio de uso de suelo y crecimiento urbano | El dominio de la tesis. Incluye Dinamica EGO, FLUS, PLUS, CLUE, SLEUTH |
| 3 | Equivalencia matemática bajo otro nombre | El frente crítico. Busca la estructura, no el vocabulario |
| 4 | Grafo de citas hacia atrás y hacia adelante | Atrapa lo que la búsqueda por términos no ve |
| 5 | Literatura gris y no anglosajona | Tesis, actas y publicaciones en chino, portugués, español, persa y turco |

El frente 3 era el importante. Si solo se busca por nombre, el resultado puede ser
un falso negativo cómodo: nadie lo llama así, luego es nuevo. Ese frente buscó la
operación, o sea un promedio ponderado de log razones de verosimilitud con pesos
normalizados derivados de una medida de discriminación de la propia variable, sin
exigir que el texto dijera Weight of Evidence ni Information Value.

Criterio de clasificación para cada candidato: idéntico, matemáticamente
equivalente, parecido pero distinto, o no relacionado.

## 3. Detalle por frente: dónde buscó cada uno, cuánto y qué encontró

Las cifras de volumen son las que reportó cada frente sobre su propio trabajo. Lo
que sí se comprobó de forma independiente son las referencias de la sección 7.

### Frente 1. Término literal: WoE junto con Information Value

**Dónde buscó.** Google Scholar, Crossref, Semantic Scholar, ResearchGate, arXiv y
web abierta. Dominios cubiertos: susceptibilidad a deslizamientos, prospección
mineral, credit scoring y epidemiología espacial.

**Cuánto.** 17 cadenas de búsqueda, cerca de 82 resultados inspeccionados, texto
completo de 3 artículos y solo resumen del resto.

**Qué encontró.** Ningún equivalente. Tres primos estructurales. El primero son los
modelos AHP-IV y GWIV, que ponderan el Information Value con juicio de expertos o
con importancia de Random Forest, o sea la jugada inversa a la de la tesis: usan el
IV como puntaje y traen el peso de fuera. El segundo es un modelo que acopla WoE
con entropía de Shannon, donde el peso sí se deriva de los mismos datos y suma uno,
lo que lo vuelve el más cercano de este frente. El tercero es la familia china de
Weighted Weights of Evidence, cuya forma `ln O(D) + suma_i c_i W_i` es idéntica en
álgebra, pero donde `c_i` corrige dependencia condicional entre capas y no mide
poder predictivo.

**Qué descartó.** Decenas de trabajos que comparan WoE contra IV como métodos
rivales, o que promedian sus mapas de salida. Era la trampa anticipada: parecen
hallazgos y no lo son.

**Qué no pudo ver.** Los tres trabajos chinos de Weighted Weights of Evidence en
texto completo, y el de WoE con entropía, cuyo OCR llegó con las fórmulas
corrompidas.

### Frente 2. Cambio de uso de suelo y crecimiento urbano

**Dónde buscó.** Documentación oficial de Dinamica EGO, no solo los artículos, más
publicaciones del grupo de Soares-Filho, Batty y Xie, FLUS, PLUS, CLUE y casos de
Brasil, India, Irán, Turquía y China.

**Cuánto.** Cerca de 15 búsquedas, unos 90 resultados revisados y alrededor de 20
páginas o PDF leídos con detalle.

**Qué encontró.** El dato más útil de toda la operación: Dinamica EGO no permite
ponderar, con cita textual de Varnier y Weber. El software ofrece prueba de
significancia del contraste y medidas para eliminar variables correlacionadas, pero
nunca un coeficiente de importancia por variable. También encontró la arquitectura
más parecida a la de la tesis fuera del dominio urbano, un modelo que combina WoE
con regresión logística y calcula `sigmoide(suma_k beta_k * W_k)`, donde los
coeficientes se estiman por máxima verosimilitud y no están acotados ni suman uno.

**Un hallazgo negativo que vale.** Buscó si alguien discute que ponderar rompe la
interpretación de log momios. Localizó la fuente canónica de esa justificación, el
trabajo de I. J. Good, pero no encontró a nadie en la literatura de uso de suelo que
discuta la ruptura. Es un hueco real, y por eso la tesis gana si lo aborda ella.

### Frente 3. Equivalencia matemática bajo otro nombre

Este fue el frente decisivo y el único que sí tumbó parte de la reclamación.

**Dónde buscó.** Análisis multicriterio con SIG y combinación lineal ponderada,
método de ponderación por entropía, clasificación bayesiana ponderada por atributos,
credit scoring, y medidas de discriminación como ganancia de información,
información mutua, chi cuadrada y Cramér.

**Cuánto.** 17 cadenas de búsqueda, con lectura de texto completo de las fuentes
clave.

**Qué encontró.** Dos antecedentes clasificados como matemáticamente equivalentes,
ambos de clasificación bayesiana ponderada por atributos: uno de 2004 que pondera
con gain ratio y otro de 2011 que pondera con una medida de Kullback-Leibler. Y la
pieza que cierra el argumento, en dos trabajos independientes: el Information Value
es exactamente la divergencia de Jeffreys.

**Qué descartó con argumento.** La hipótesis de partida era que esto sería
combinación lineal ponderada de análisis multicriterio. Resultó falsa, y por una
razón precisa: en esa técnica el peso es un juicio subjetivo de experto, no una
medida estadística de discriminación. El método de ponderación por entropía también
quedó descartado, porque mide dispersión del criterio entre alternativas y no
divergencia respecto al evento. Son falsos amigos cercanos.

**Qué declaró como no encontrado.** Que alguien use el IV como peso del scorecard en
credit scoring en vez de como filtro, y que alguien use chi cuadrada o Cramér
normalizados como peso en naive Bayes. Las reportó como ausencias, no como
refutaciones.

**Aporte extra.** Localizó el contraargumento de Zaidi y colegas en JMLR 2013 contra
ponderar por poder predictivo.

### Frente 4. Grafo de citas

**Dónde buscó.** API de OpenAlex para rastreo de citantes, Semantic Scholar con
límite de tasa, y agregadores para las fuentes chinas.

**Cuánto.** Del artículo de Agterberg y Cheng sobre independencia condicional revisó
104 títulos de sus 209 citantes y profundizó en 15. De Xia y colegas revisó los 8
citantes. Sumó unos 15 trabajos de variantes con nombre propio y 6 fuentes técnicas
de credit scoring.

**Por qué ese punto de partida.** El nodo de independencia condicional se eligió a
propósito: es donde alguien introduciría una ponderación como remedio. Y en efecto
ahí vive la familia de Weighted Weights of Evidence, BoostWofE y los modelos de
dependencia condicional ajustada. Todos ponderan, ninguno pondera por IV.

**Qué encontró.** Verificó en texto completo la ecuación de un modelo que acopla WoE
con entropía de Shannon, con pesos normalizados que suman uno. Y encontró el trabajo
más cercano al linaje de la tesis, un modelo de paisaje de la familia Dinamica EGO
que sí calcula el IV explícitamente, pero solo para descartar variables por debajo
de un umbral, igual que en credit scoring. La probabilidad de transición sigue
siendo la suma simple.

**Qué no pudo ver.** No hizo el rastreo completo de citantes de Bonham-Carter, que
tiene miles. Sin Scopus ni Connected Papers. Los trabajos chinos, solo en resumen.

### Frente 5. Literatura gris y no anglosajona

**Dónde buscó.** En seis idiomas: inglés, chino, portugués, español, persa y turco.
Repositorios y bases: CNKI a través de agregadores, TESIUNAM, repositorio de la
UNAL, eprints de la UANL, tesis de la USP, Redalyc, SciELO, el repositorio de la
Universidad de Tübingen, arXiv, EarthArXiv y actas de ISPRS, AGILE y GIScience.

**Qué encontró.** Cuatro antecedentes parecidos y ninguno equivalente. Dos modelos de
valor de información ponderado, uno con entropía de Shannon y otro con agrupamiento
gris. El trabajo de la familia Dinamica EGO que usa el IV solo como filtro. Y una
tesis doctoral que sí combina mapas normalizados de evidencia con pesos que suman
uno, pero fijados por juicio experto y prueba y error, cosa que la propia tesis
reconoce como limitación.

**Un hallazgo terminológico útil.** Confirmó que en la literatura de deslizamientos
los términos Information Value y Statistical Index a veces nombran lo mismo que WoE
y a veces cosas distintas. Esa confusión podía esconder un antecedente, y por eso el
frente la revisó explícitamente.

**Qué dejó como pista sin verificar.** Tres trabajos citados de segunda mano que no
pudo leer de primera. Los separó del resto en vez de reportarlos como hallazgos, que
es lo correcto.

**Qué no pudo ver.** CNKI de forma nativa, ProQuest, DART-Europe, OpenThesis, SSRN y
un artículo de Springer de pago.

## 4. Qué salió

### En crecimiento urbano no hay antecedente

Los cinco frentes coinciden. La práctica estándar es la suma simple. Se revisaron
Bauru, Ningbo, Atakum, casos de India e Irán, y los modelos FLUS y PLUS.

El dato más contundente es sobre Dinamica EGO, el software de referencia del campo
para WoE con autómatas celulares. No es que nadie haya ponderado: el programa no
lo permite. Varnier y Weber lo dicen literalmente:

> In the Dinamica Ego software, the adjustment of evidence weights occurs
> automatically, not allowing any user definition.

### En deslizamientos y minería existe la arquitectura, con otro ponderador

Hay una familia documentada desde 2009 que hace `suma_i w_i * evidencia_i` con
pesos que suman uno. La diferencia está en el origen del peso: entropía de
Shannon, AHP, AHP difuso, PCA, agrupamiento gris o correlación condicional entre
capas.

En ningún caso el peso sale del Information Value de la propia variable. Varios de
esos trabajos ya calculaban el IV y aun así eligieron un ponderador externo.

### En aprendizaje automático sí hay antecedente, y es el hallazgo relevante

Sumar pesos de evidencia es naive Bayes en forma de log momios. Existe una línea
de clasificación bayesiana ponderada por atributos que pondera cada log
verosimilitud por una medida de relevancia normalizada de la propia variable.
Zhang y Sheng lo hacen con gain ratio en 2004. Lee, Gutierrez y Dou lo hacen con
una medida de Kullback-Leibler en 2011.

El eslabón que cierra el argumento es que el Information Value **es** la
divergencia de Jeffreys, o sea la divergencia de Kullback-Leibler simetrizada. No
es una heurística de credit scoring sin fundamento, es una divergencia con
propiedades conocidas, y la identidad está demostrada formalmente.

Encadenado: ponderar WoE por IV normalizado equivale a ponderar naive Bayes por
una divergencia de Kullback-Leibler normalizada. Eso es el método de Lee y colegas
de 2011 con la versión simétrica de la divergencia.

## 5. Veredicto

El mecanismo no es nuevo como matemática. Lo que no aparece documentado es su
traslado al modelado de crecimiento urbano.

Eso sigue siendo una contribución, del tipo traslado de dominio, y el traslado no
es obvio: el software de referencia del campo ni siquiera admite la ponderación, y
la familia vecina de deslizamientos llegó a la idea de ponderar pero tomó otro
camino teniendo el IV disponible.

Formulación defendible, que no reclama primacía y se adelanta a la objeción:

> En la revisión realizada no se identificaron trabajos que empleen el Information
> Value normalizado como ponderador de los pesos de evidencia dentro de la regla de
> transición de un autómata celular de crecimiento urbano. El mecanismo es
> formalmente análogo al de la clasificación bayesiana ponderada por atributos,
> donde el ponderador se deriva de una medida de divergencia entre las
> distribuciones condicionadas al evento. Su traslado al modelado de crecimiento
> urbano no se encontró documentado.

## 6. Dos objeciones previsibles

**La ponderación rompe la interpretación bayesiana.** La suma de pesos de
evidencia es el log de momios posteriores menos el de los previos, bajo
independencia condicional. Por eso se aplica la sigmoide: es una derivación, no
una conveniencia. Como los pesos IV normalizados suman uno, el resultado es un
promedio ponderado y no una suma de razones de verosimilitud. El código tampoco
suma el log momio previo. La salida, entonces, no es una probabilidad posterior.
Conviene llamarla índice o puntaje de aptitud.

Esto además explica un ajuste que hoy figura en la tesis como resultado empírico
sin mecanismo: al promediar en vez de sumar, el puntaje se comprime hacia el
promedio de los pesos en lugar de acumular evidencia, la sigmoide ronda 0,5, y el
término de vecindad basta para rebasar umbrales bajos. De ahí la sobre-predicción
masiva con umbrales de 0,5 a 0,6 y la necesidad de subir a 0,75.

**Hay literatura que argumenta en contra de ponderar por poder predictivo.** Zaidi
y colegas sostienen en JMLR 2013 que el peso debería servir para mitigar
violaciones de independencia condicional, no para premiar variables predictivas.
La tesis queda del lado que ese trabajo critica, junto con Zhang y Sheng y con Lee
y colegas. Conviene conocerlo antes del examen.

## 7. Referencias verificadas

Todas se comprobaron contra Crossref o contra arXiv, una por una. Se verificaron
porque entre los reportes hubo al menos una discrepancia de autoría: un trabajo
atribuido a Xu y colegas resultó ser de Cao, Li y Zhu.

| Referencia | DOI | Para qué sirve |
|---|---|---|
| Zhang y Sheng (2004), Learning weighted naive Bayes with accurate ranking, ICDM | 10.1109/ICDM.2004.10030 | Antecedente del mecanismo, con gain ratio |
| Lee, Gutierrez y Dou (2011), Calculating Feature Weights in Naive Bayes with Kullback-Leibler Measure, ICDM | 10.1109/ICDM.2011.29 | Antecedente más cercano, con divergencia KL |
| Rojas, Alvarez y Rojas, Statistical Hypothesis Testing for Information Value | arXiv:2309.13183 | Establece que el IV es la divergencia de Jeffreys |
| Sudjianto y Burakov, An Information-Theoretic Framework for Credit Risk Modeling | arXiv:2509.09855 | Demuestra la misma identidad |
| Varnier y Weber (2025), Land 14(3) 560 | 10.3390/land14030560 | Dinamica EGO no permite ponderar |
| Cao, Li y Zhu (2022), Sustainability 14(17) 11092 | 10.3390/su141711092 | Familia de IV ponderado, con AHP difuso y PCA |
| Ba y colegas (2017), ISPRS Int. J. Geo-Inf. 6(1) 18 | 10.3390/ijgi6010018 | IV ponderado por agrupamiento gris |
| Malczewski (2000), Transactions in GIS 4(1) | 10.1111/1467-9671.00035 | Combinación lineal ponderada en SIG |
| Hall (2007), Knowledge-Based Systems 20(2) | 10.1016/j.knosys.2006.11.008 | Ponderación de atributos en naive Bayes |

Zaidi, Cerquides, Carman y Webb (2013), JMLR 14, 1947-1988, no se comprobó contra
Crossref porque JMLR no asigna DOI. Debe verificarse antes de citarla.

## 8. Límites de la búsqueda

Se declaran para que la afirmación quede acotada a lo que realmente se revisó.

- CNKI no fue accesible. La cobertura china salió de resúmenes indexados y sitios
  agregadores de tesis. Es el hueco más grande.
- Tres trabajos chinos sobre Weighted Weights of Evidence, de 2009 y 2012, solo se
  vieron en resumen. Si su coeficiente de corrección resultara equivalente al IV
  normalizado, el veredicto cambiaría. No se pudo descartar.
- Varios artículos de Springer y Wiley quedaron tras muro de pago. De ellos solo se
  leyó el resumen.
- No se consultaron ProQuest, DART-Europe ni Scopus de forma nativa.

## 9. Lo que falta

No hay ablación. No se ha comparado la ponderación por IV contra la suma simple ni
contra pesos iguales, así que hoy no hay evidencia de que la ponderación mejore el
desempeño. Sin eso, el traslado de dominio es una variante, no una aportación
demostrada. Es la primera pregunta que sigue a la reclamación de novedad.

El script está escrito en `tools/ablacion_ponderacion.py` y no reentrena el WoE:
solo cambia el vector de pesos. Requiere `tools/variables_rapidas.py`, que
reimplementa dos de las siete variables espaciales de forma vectorizada y baja el
costo de un paso de más de seis minutos a 0,9 segundos. Seis de las siete
variables salen idénticas; `nearest_cluster_size` difiere en 0,15 por ciento de
las celdas por desempate entre clusters equidistantes, y es la que menos pesa, con
1,2 por ciento.

Nota aparte, detectada durante este trabajo: `validate_quinquenal_2011_2016.py` no
fija semilla aleatoria, así que las cifras publicadas provienen de una corrida
estocástica que no se puede repetir, y la tesis las reporta con tres decimales.
