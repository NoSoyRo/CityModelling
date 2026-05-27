# Arquitectura

## Vista general

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Querétaro Urban Lab                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────────┐    HTTP / WS    ┌──────────────────────────┐   │
│   │  Frontend SPA   │ ◀────────────▶ │  FastAPI backend (uvicorn)│   │
│   │  React 18 + TS  │                 │  Python 3.11+             │   │
│   │  Vite + Tailwind│                 │                           │   │
│   └─────────────────┘                 └────────────┬──────────────┘   │
│                                                    │                  │
│                                                    │ import           │
│                                                    ▼                  │
│                                       ┌──────────────────────────┐    │
│                                       │  tesis_ac (src/)         │    │
│                                       │  pipeline / woe / ca /…  │    │
│                                       └────────────┬─────────────┘    │
│                                                    │                  │
│                                                    │ lee / escribe    │
│                                                    ▼                  │
│                                       ┌──────────────────────────┐    │
│                                       │  data/raw  data/processed│    │
│                                       └──────────────────────────┘    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

## Decisiones técnicas

### Por qué FastAPI

- Soporte nativo de WebSockets sin librerías adicionales.
- Validación automática vía Pydantic v2, lo que nos da contratos tipados
  con TypeScript en el frontend (los modelos espejan `backend/models.py`).
- Compatibilidad directa con `asyncio` para fan-out de eventos.

### Por qué Vite + React (en lugar de Streamlit, Gradio o tkinter)

- Streamlit/Gradio están optimizados para *demos de IA*; visualmente delatan
  la herramienta y limitan la densidad de información.
- tkinter no permite el tipo de visualización (Recharts, mapas) que pide la
  tesis.
- Vite + React permite tener un design system propio, scrollbars y
  tipografías controlados, y un bundle de salida estático servido por el
  propio backend en producción.

### Por qué reusar `tesis_ac` en lugar de reescribir

La aplicación no debe ser un fork del modelo: cualquier corrección que se
haga al pipeline desde la tesis debe verse aquí sin tocar la UI. El backend
hace `sys.path.insert(0, project_root / "src")` y llama directamente a los
mismos `FeatureExtractor`, `SatelliteImageProcessor`, `WoECalculator` que
los scripts de la raíz.

### Streaming vía WebSocket único

Todos los jobs publican en el mismo `EventBus` y los WebSockets reciben
todo. El cliente filtra por `job_id` cuando quiere enfocarse. Esto evita
abrir un canal nuevo por job y mantiene la lógica de reconexión simple.

### Modos de despliegue

| Modo            | Backend                       | Frontend                  |
|-----------------|-------------------------------|---------------------------|
| `launch.py`     | uvicorn :8765 sirve API + SPA | bundle Vite estático      |
| Desarrollo      | uvicorn :8765                 | `vite dev` :5173 + proxy  |

## Layout de carpetas

Ver `README.md` en la raíz de `desktop-app/`.

## Convenciones de logs/eventos

Cada evento publicado al WebSocket tiene la forma de `JobEvent` (ver
`backend/models.py`):

- `level`: `info | debug | warn | error | metric | artifact | progress | done`
- `step`: nombre canónico del paso (`load_image`, `features`, `pca`, …)
- `payload`: diccionario libre con datos estructurados (shapes, métricas,
  tiempos, rutas)

El frontend renderiza `metric` con flecha y color teal, `artifact` con
rombo verde y `error` con cruz roja. `done` cierra el job.

## Límites conocidos

- El runner sólo expone E1+E2 en esta iteración. E3 (entrenamiento WoE) y
  E4–E5 (simulación AC + métricas) están planificados pero aún se ejecutan
  fuera de la aplicación con los scripts de la raíz.
- No hay autenticación. La aplicación está pensada para correr en local
  (`127.0.0.1`), nunca expuesta a Internet.
- La inspección de pickles es lectura completa en memoria; archivos muy
  grandes pueden tardar.
