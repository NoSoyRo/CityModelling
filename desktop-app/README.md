# Querétaro Urban Lab

> Aplicación de escritorio para inspeccionar, ejecutar y validar el modelo de
> crecimiento urbano WoE+AC sobre la Zona Metropolitana de Querétaro
> (1984–2020).

`Querétaro Urban Lab` es la interfaz visual del pipeline de la tesis. Su
propósito es que cualquier paso del modelo —desde un PNG crudo hasta una
métrica FoM de validación— sea inspeccionable, reproducible y trazable sin
tener que abrir un cuaderno de Jupyter o un terminal.

El espíritu es el de [NetLogo](https://ccl.northwestern.edu/netlogo/): una
herramienta donde el modelo, los datos y las gráficas conviven en la misma
ventana, y donde el usuario puede *ver* qué está pasando dentro del sistema.

---

## 1. Filosofía

Tres ideas guían el diseño:

1. **Cada etapa es un artefacto con metadatos**. El pipeline produce
   artefactos discretos (`.npy`, `.pkl`, `.json`, `.png`). La aplicación los
   trata como objetos de primera clase: cada uno tiene ruta, tipo, tamaño,
   forma y fecha; cada uno se puede previsualizar; cada uno se puede comparar
   con otra corrida.
2. **El control de flujo es explícito**. Las cinco etapas
   (Etapa 0–Etapa 5) están enumeradas en el árbol lateral. El usuario sabe en
   todo momento en qué punto del modelo se encuentra y qué entradas/salidas
   tiene cada paso.
3. **El código que ejecuta es exactamente el código de la tesis**. La aplicación
   no es una simulación didáctica: es un *front-end* sobre el paquete
   `tesis_ac` que ya existe en `src/`. Lo que ejecutas aquí es lo que
   produce los números reportados en el documento.

---

## 2. Arquitectura

```
desktop-app/
├── backend/                FastAPI + uvicorn (Python 3.11+)
│   ├── main.py             Punto de entrada HTTP / WebSocket
│   ├── inspector.py        Lectura de artefactos (npy, pkl, json)
│   ├── runner.py           Wrapper sobre tesis_ac (pipeline + validación)
│   ├── models.py           Schemas Pydantic v2
│   ├── catalog.py          Descubrimiento de artefactos en data/
│   └── ws.py               Bus de eventos para streaming de logs
│
├── frontend/               Vite + React + TypeScript
│   ├── src/
│   │   ├── App.tsx
│   │   ├── styles.css
│   │   ├── design/         Tokens del sistema de diseño
│   │   ├── components/     Sidebar / StepDetail / ArtifactViewer / ...
│   │   ├── lib/            cliente HTTP / WS
│   │   └── state/          store zustand
│   └── vite.config.ts
│
├── scripts/
│   └── launch.py           `python launch.py` arranca backend + abre la UI
│
└── docs/
    ├── architecture.md     Diagrama, decisiones técnicas y trade-offs
    ├── pipeline.md         Catálogo de etapas con E/S por paso
    └── api.md              Referencia de endpoints y eventos
```

La comunicación es estricta entre dos procesos:

| Canal | Tecnología | Uso |
|---|---|---|
| HTTP `:8765` | FastAPI / REST | Catálogo de artefactos, inspección, lanzamiento de jobs, snapshots |
| WS `:8765/ws` | WebSocket | Logs, progreso por paso, eventos de finalización |

---

## 3. Etapas del pipeline (catálogo navegable)

| ID | Etapa | Entrada | Salida | Módulo |
|---|---|---|---|---|
| E0 | Imágenes crudas | — | `data/raw/imagen_YYYY.png` | (manual) |
| E1.a | Extracción de 23 features | PNG RGB | matriz `(n_pixels, 23)` | `tesis_ac.pipeline.feature_extraction` |
| E1.b | Standard scaler + PCA → 8 comp. | matriz 23-D | matriz 8-D | `FeaturePreprocessor` |
| E1.c | K-Means $k=2$ + SVM lineal | matriz 8-D | mapa binario crudo | `tesis_ac.pipeline.clustering` |
| E2  | Estandarización de labels (NDVI / centro) | mapa binario crudo | `data/processed/standardized_maps/YYYY.npy` | `tesis_ac.historical.standardize_labels` |
| E3  | Variables espaciales + WoE pooled (1984–2010) | mapas E2 | `woe_pooled_1984_2010.pkl` | `train_woe_pooled.py` + `tesis_ac.woe.woe` |
| E4  | Simulación AC quinquenal 2011→2016 (×5 ventanas) | pickle WoE + estado inicial | `predicted_YYYY.npy` por año | `validate_quinquenal_*.py` + `tesis_ac.ca.rules` |
| E5  | Métricas (FoM, Kappa, IoU, descomposición) | predicción + observado | `validation_results.json` | mismo script |

Cada celda de esta tabla corresponde a una vista en la barra lateral. La
documentación detallada por etapa está en `docs/pipeline.md`.

---

## 4. Instalación

### 4.1 Backend

El backend reutiliza el `.venv` del proyecto principal. Solo se añaden tres
dependencias.

```bash
# Desde la raíz del repositorio
.venv/bin/pip install -r desktop-app/backend/requirements.txt
```

### 4.2 Frontend

```bash
cd desktop-app/frontend
npm install
```

---

## 5. Uso

### 5.1 Modo producción (un comando)

```bash
.venv/bin/python desktop-app/scripts/launch.py
```

Esto:

1. Construye el frontend si no está construido (`npm run build`).
2. Arranca el backend FastAPI en `localhost:8765` sirviendo el frontend
   estático.
3. Abre `http://localhost:8765` en el navegador por defecto.

### 5.2 Modo desarrollo (dos terminales)

Terminal A — backend:

```bash
.venv/bin/python -m uvicorn backend.main:app --reload --port 8765 \
    --app-dir desktop-app
```

Terminal B — frontend (con hot reload):

```bash
cd desktop-app/frontend
npm run dev
```

El frontend de desarrollo corre en `localhost:5173` y hace proxy al backend
en `localhost:8765`.

---

## 6. Convenciones de la interfaz

- **Densidad alta sobre estilo florido**. La aplicación está pensada para
  trabajar con ella, no para una captura de pantalla.
- **Datos siempre visibles**. Cuando una operación produce un array, su
  forma, dtype, min/max y conteo aparecen junto al render visual.
- **Tiempos a la vista**. Cada paso muestra cuánto tardó la última vez que se
  ejecutó, comparable con la corrida histórica si existe.
- **Sin magia oculta**. Cada vista de la UI documenta qué archivo del
  paquete `tesis_ac` ejecutó y con qué parámetros.

---

## 7. Estado del proyecto

Esta es la primera versión usable de la aplicación. El alcance actual es:

- [x] Descubrimiento automático de artefactos existentes en `data/`
- [x] Inspector universal de `.npy`, `.pkl`, `.json`, `.png`
- [x] Visor especializado de pickles WoE (IV, bins, woe_values por variable)
- [x] Dashboard de métricas quinquenales (FoM/Kappa/IoU por ventana)
- [x] Ejecución de la Etapa 1 (clasificación) sobre una imagen con streaming
- [ ] Ejecución de la Etapa 3 (entrenamiento WoE) desde la UI
- [ ] Ejecución de las Etapas 4–5 (validación) desde la UI
- [ ] Comparador de corridas (diff entre dos `validation_results.json`)
- [ ] Exportación de figuras LaTeX-ready

Las casillas vacías son las próximas iteraciones.

---

## 8. Licencia

Mismo régimen que el proyecto principal: uso académico.
