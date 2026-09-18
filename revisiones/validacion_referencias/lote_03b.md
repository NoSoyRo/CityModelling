# Lote 3b: 5 referencias

## MacQueen1967

### Entrada bibliografica tal como esta en referencias.bib

```bibtex
@inproceedings{MacQueen1967,
  author    = {MacQueen, James B.},
  title     = {Some Methods for Classification and Analysis of Multivariate Observations},
  booktitle = {Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability},
  editor    = {Le Cam, Lucien M. and Neyman, Jerzy},
  volume    = {1},
  pages     = {281--297},
  year      = {1967},
  publisher = {University of California Press},
  address   = {Berkeley, CA}
}
```

### Afirmaciones que la tesis le atribuye (1)

1. `cap02_marco_teorico/cap02_marco_teorico.tex:61`
   > El agrupamiento K-Means, propuesto por \textcite{MacQueen1967}, es un método no supervisado que particiona un conjunto de observaciones en $k$ grupos, minimizando la suma de distancias cuadradas de cada punto al centroide de su grupo.

---

## RamirezHernandez2021

### Entrada bibliografica tal como esta en referencias.bib

```bibtex
@book{RamirezHernandez2021,
  author    = {Ramírez Hernández, Roberto},
  title     = {Zona Metropolitana de la {Ciudad de México}: crecimiento y expansión al 2040. Prospectiva territorial usando modelos de simulación urbana},
  publisher = {Universidad Nacional Autónoma de México, Instituto de Investigaciones Económicas},
  year      = {2021},
  doi       = {10.22201/iiec.9786073044349e.2021}
}
```

### Afirmaciones que la tesis le atribuye (3)

1. `cap01_introduccion/cap01_introduccion.tex:119`
   > Los trabajos representativos para ciudades mexicanas \parencite{SuarezDelgado2007,RamirezHernandez2021,Chihuahua2023} satisfacen parcialmente R1 y R2, pero no R3 ni R4 de la tabla de requerimientos.

2. `cap03_estado_del_arte/cap03_estado_del_arte.tex:408`
   > \textcite{RamirezHernandez2021}, en un libro del Instituto de Investigaciones Económicas de la UNAM, construye para la ZMCM lo que él mismo clasifica como un modelo econométrico con simulación espacial: un autómata celular sobre celdas territoriales con vecindad de Moore, cuyas probabilidades de transición se estiman con regresiones logísticas binomial y multinomial y se resuelven mediante una rutina de Monte Carlo.

3. `cap03_estado_del_arte/cap03_estado_del_arte.tex:472`
   > \begin{table}[H] \centering \caption{Posicionamiento de la literatura mexicana frente a R1--R4.} \label{tab:cuadrante-mexico} \begin{tabular}{lcccc} \toprule \textbf{Trabajo} & \textbf{R1} & \textbf{R2} & \textbf{R3} & \textbf{R4} \\ \midrule \textcite{SuarezDelgado2007} & no & \checkmark & no & \checkmark \\ \textcite{RamirezHernandez2021} & no & \checkmark & no & parcial \\ \textcite{Chihuahua2023} & parcial & \checkmark & no & parcial \\ \textcite{JimenezLopez2018} & parcial & \checkmark & parcial & parcial \\ \textcite{JimenezLopez2021} & parcial & \checkmark & \checkmark & parcial \\ \bottomrule \end{tabular} \end{table}

---

## Silva2002

### Entrada bibliografica tal como esta en referencias.bib

```bibtex
@article{Silva2002,
  author    = {Silva, Elisabete A. and Clarke, Keith C.},
  title     = {Calibration of the SLEUTH urban growth model for Lisbon and Porto, Portugal},
  journal   = {Computers, Environment and Urban Systems},
  year      = {2002},
  volume    = {26},
  number    = {6},
  pages     = {525--552},
  doi       = {10.1016/S0198-9715(01)00014-X}
}
```

### Afirmaciones que la tesis le atribuye (2)

1. `cap03_estado_del_arte/cap03_estado_del_arte.tex:18`
   > El siguiente nivel del enfoque llegó con el modelo que hoy se conoce como SLEUTH, aplicado primero a la bahía de San Francisco y al corredor Washington-Baltimore \parencite{Clarke1998}; el acrónimo se consolidó con la aplicación a Lisboa y Oporto \parencite{Silva2002}.

2. `cap03_estado_del_arte/cap03_estado_del_arte.tex:86`
   > Alo largo del Capítulo~\ref{ch:marco-teorico} se define el proceso de calibración SLEUTH, no obstante a grandes razgos se resuelve mediante búsqueda por fuerza bruta sobre un espacio de parámetros de sus cinco coeficientes de crecimiento, cada uno acotado a un intervalo fijo, lo que da del orden de $a^{b}$ combinaciones posibles \parencite{Silva2002,Clarke2008}, cada una evaluada con simulaciones Monte Carlo de múltiples iteraciones, claramente el tiempo de sintonización es probablemente mayor a lo que el acuerdo de nivel de servicio debe exigir para que sea operacional.

---

## SoaresFilho2004

### Entrada bibliografica tal como esta en referencias.bib

```bibtex
@article{SoaresFilho2004,
  author  = {Soares-Filho, Britaldo Silveira and Alencar, Ane and Nepstad, Daniel and Cerqueira, Gustavo and Vera Diaz, Maria del Carmen and Rivero, Sergio and Sol{\'o}rzano, Luis and Voll, Eliane},
  title   = {Simulating the response of land-cover changes to road paving and governance along a major {Amazon} highway: the {Santar{\'e}m}--{Cuiab{\'a}} corridor},
  journal = {Global Change Biology},
  year    = {2004},
  volume  = {10},
  number  = {5},
  pages   = {745--764},
  doi     = {10.1111/j.1529-8817.2003.00769.x}
}
```

### Afirmaciones que la tesis le atribuye (1)

1. `cap03_estado_del_arte/cap03_estado_del_arte.tex:270`
   > Conviene precisar la genealogía, porque a menudo se resume mal: la formulación original de \textcite{SoaresFilho2002} calcula las probabilidades de transición mediante regresión logística, y la arquitectura admite indistintamente ese método o el de Pesos de Evidencia; fue en las aplicaciones urbanas de \textcite{Almeida2003} donde la ruta WoE se desarrolló y estimó empíricamente, y en \textcite{SoaresFilho2004} donde quedó incorporada al uso corriente del marco para escenarios de cambio de cobertura.

---

## Wahyudi2016

### Entrada bibliografica tal como esta en referencias.bib

```bibtex
@article{Wahyudi2016,
  author  = {Wahyudi, Agung and Liu, Yan},
  title   = {Cellular Automata for Urban Growth Modelling: A Review on Factors Defining Transition Rules},
  journal = {International Review for Spatial Planning and Sustainable Development},
  year    = {2016},
  volume  = {4},
  number  = {2},
  pages   = {60--75},
  doi     = {10.14246/irspsd.4.2_60}
}
```

### Afirmaciones que la tesis le atribuye (1)

1. `cap03_estado_del_arte/cap03_estado_del_arte.tex:511`
   > A esa heterogeneidad se suma un desequilibrio geográfico: la revisión de \textcite{Wahyudi2016}, que clasifica por región ochenta y ocho aplicaciones de autómatas celulares urbanos publicadas entre 1993 y 2012, encuentra que al menos la mitad se concentra en ciudades de Estados Unidos y China, mientras que las aplicaciones en América Latina, África y el resto de Asia son escasas.

---

