# Actualización de avance, Tesis

Estimada Dra. Lárraga:

Le mando esta nota para ponerla al corriente de cómo quedó el documento antes de que lo revise.

---

## Sobre la autoría

Lo primero que quiero dejar claro: la escritura es mía. Evité delegar redacción a herramientas de IA generativa; el único apoyo digital que usé fue corrección ortográfica y de estilo, en los términos que usted misma señaló como admisibles. Las decisiones metodológicas, el argumento y la redacción de fondo son propios.

---

## La historia que cuenta la tesis

El documento hoy tiene un hilo único. Las ciudades intermedias mexicanas crecen de forma dispersa y poco planificada, y para la Zona Metropolitana de Querétaro no existe un modelo de crecimiento urbano interpretable celda a celda cuya capacidad predictiva se haya probado en varias ventanas temporales independientes con código y datos abiertos. De esa brecha concreta se desprenden los requerimientos y las hipótesis. Y queda explícito desde el inicio: el acoplamiento WoE–autómata celular no es la aportación en sí.

A partir de ahí el documento avanza de forma encadenada: el marco teórico asienta las bases formales, el estado del arte demuestra que los problemas de fondo —calibración, validación temporal, interpretabilidad, transferibilidad— siguen abiertos en la literatura, el modelo formaliza la regla de transición tal como está implementada en el código, los resultados la ponen a prueba en cinco ventanas quinquenales independientes, y las conclusiones cierran sin sobrevender. La brecha que planteo al inicio es la misma que documento en la literatura, la que resuelve el modelo, la que pruebo y la que concluyo. No hay desvíos.

---

## Cómo quedaron sus observaciones

Atendí todo lo que señaló en su revisión. Los puntos de fondo:

- **Novedad del trabajo.** Reorienté el discurso completo: la contribución es el protocolo de validación reproducible multi-ventana sobre una ciudad intermedia mexicana, y ese mensaje es consistente de la introducción a las conclusiones.
- **Percepción remota.** Eliminé la idea de "NDVI aproximado" a partir de imágenes RGB y la sustituí por un índice cromático correcto, explicando por qué el NDVI no se puede calcular sin banda infrarroja.
- **Validación sin circularidad.** La concordancia del clasificador quedó reformulada como coherencia interna frente a sus propias pseudoetiquetas. La validación real del trabajo es la evaluación temporal del modelo.
- **Hipótesis.** Las reescribí como afirmaciones falsables, sin referencias a archivos internos, y las verifiqué en las conclusiones.
- **Estado del arte.** Lo reorganicé por problemas metodológicos persistentes en lugar del esquema "autor → método → limitación".
- **Rigor en las citas.** Completé las referencias que señaló y anclé las afirmaciones generales a su fuente.
- **Tono.** Quité el lenguaje defensivo, las muletillas y los pies de figura que afirmaban causalidad donde solo hay correlación.
- **Honestidad metodológica.** Documenté las fuentes de incertidumbre, reconocí los resultados más bajos como límites reales sin minimizarlos, y justifiqué el pipeline de clasificación dado que no hay verdad terreno para los 37 años.

---

## Lo que agregué por iniciativa propia

Más allá de responder su revisión, hay cosas que hice porque me parecieron necesarias para que el trabajo fuera mas sólido.

En la parte de datos y método: construí manualmente una serie anual de 37 años de la ZMQ,un insumo reutilizable en sí mismo, y diseñé la validación en cinco ventanas quinquenales independientes en lugar de un solo intervalo, que es precisamente lo que permite hablar de estabilidad y no de un ajuste afortunado. Reporté la descomposición del error siguiendo el marco de Pontius, y todo el pipeline corre en CPU estándar sin software propietario ni GPU.

En la escritura: cuidé que el documento se lea como un razonamiento continuo, no como capítulos sueltos. En el estado del arte fui más allá de reorganizar: discutí el sobreajuste de calibración en modelos como SLEUTH y el sesgo por autocorrelación espacial del WoE, anclados a su literatura. En el capítulo de modelo argumenté por qué cada componente está ahí, por qué WoE y no regresión logística, por qué siete variables derivadas solo del mapa, por qué un autómata celular y no el mapa estático, y separé tres planos (modelo conceptual, implementación y configuración) para evitar confusiones en la defensa. En los resultados di sentido urbano a los números y mostré que el crecimiento de la ZMQ es por contagio borde a borde y no por difusión aleatoria, lo que justifica el modelo elegido. Las conclusiones mapean objetivos e hipótesis con su cumplimiento y cierran con un mensaje concreto sobre la ZMQ en lugar de un final genérico. Y elaboré figuras propias a lo largo del documento para hacer autoexplicativos los capítulos: el pipeline completo, el mapa binario, la transición, la vecindad de Moore, un paso del autómata celular y los componentes del *Figure of Merit*.

---

## Estado del documento

Compila sin errores con todas las referencias cruzadas, citas y figuras resueltas. Tiene su material preliminar: portada institucional UNAM–PCIC, dedicatoria, resumen, *abstract* y lista de acrónimos.

---

## Lo que le pido

Antes de que revise el documento completo, le pido que valide si esta nota refleja fielmente la esencia del trabajo: la historia que cuento, dónde ubico la aportación y cómo incorporé sus observaciones. Si algo no está bien encuadrado —el alcance de la contribución, el énfasis en algún punto, cualquier matiz de fondo— lo ajusto antes de entregarle la versión completa, para que llegue ya conforme a lo que usted espera.

Con aprecio,
Rodrigo Moreno López