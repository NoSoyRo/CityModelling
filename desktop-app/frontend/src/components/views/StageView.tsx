import { useEffect, useState } from "react";
import { api } from "../../lib/api";
import type { StageDescriptor, StageId } from "../../lib/types";
import { ViewHeader } from "./ViewHeader";

const stageColors: Record<StageId, string> = {
  E0: "bg-stage-E0",
  E1: "bg-stage-E1",
  E2: "bg-stage-E2",
  E3: "bg-stage-E3",
  E4: "bg-stage-E4",
  E5: "bg-stage-E5",
};

export function StageView({ stageId }: { stageId: StageId }) {
  const [stages, setStages] = useState<StageDescriptor[]>([]);

  useEffect(() => {
    api.stages().then(setStages);
  }, []);

  const stage = stages.find((s) => s.id === stageId);
  if (!stage) {
    return <div className="p-6 text-ink-muted">Cargando descripción…</div>;
  }

  return (
    <div>
      <ViewHeader
        eyebrow={`Etapa ${stage.id}`}
        title={stage.title}
        subtitle={stage.one_liner}
        right={
          <span
            className={
              "w-3 h-3 rounded-sm " + stageColors[stage.id]
            }
          />
        }
      />

      <div className="p-6 grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Panel title="Entrada">
          <ul className="font-mono text-sm space-y-1">
            {stage.inputs.map((i) => (
              <li key={i} className="text-ink-muted">
                <span className="text-ink-subtle mr-2">›</span>
                {i}
              </li>
            ))}
          </ul>
        </Panel>

        <Panel title="Salida">
          <ul className="font-mono text-sm space-y-1">
            {stage.outputs.map((o) => (
              <li key={o} className="text-ink">
                <span className="text-ink-subtle mr-2">·</span>
                {o}
              </li>
            ))}
          </ul>
        </Panel>

        <Panel title="Código">
          <div className="font-mono text-sm text-ink break-words">
            {stage.code_module}
          </div>
        </Panel>

        <Panel title="Parámetros">
          {stage.parameters.length === 0 ? (
            <div className="text-sm text-ink-subtle italic">
              No requiere parámetros.
            </div>
          ) : (
            <ul className="font-mono text-sm space-y-1">
              {stage.parameters.map((p) => (
                <li key={p} className="text-ink-muted">
                  <span className="text-ink-subtle mr-2">·</span>
                  {p}
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>

      <div className="px-6 pb-6">
        <p className="text-xs text-ink-subtle max-w-3xl">
          Para explorar los artefactos producidos por esta etapa, expande
          el nodo correspondiente en la barra lateral y selecciona uno. Cada
          artefacto se abre con la inspección apropiada (estadísticas
          numéricas, árbol de pickle, gráficos WoE o vista previa de imagen).
        </p>
      </div>
    </div>
  );
}

function Panel({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="panel">
      <header className="panel-header">{title}</header>
      <div className="p-4">{children}</div>
    </section>
  );
}
