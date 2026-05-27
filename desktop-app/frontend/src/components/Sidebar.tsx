import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { useApp } from "../state/store";
import type { CatalogResponse, StageDescriptor, StageId } from "../lib/types";
import {
  Boxes,
  ChevronDown,
  ChevronRight,
  Gauge,
  PlayCircle,
  Terminal,
} from "lucide-react";

const stageColors: Record<StageId, string> = {
  E0: "bg-stage-E0",
  E1: "bg-stage-E1",
  E2: "bg-stage-E2",
  E3: "bg-stage-E3",
  E4: "bg-stage-E4",
  E5: "bg-stage-E5",
};

export function Sidebar() {
  const { selection, setSelection } = useApp();
  const [stages, setStages] = useState<StageDescriptor[]>([]);
  const [catalog, setCatalog] = useState<CatalogResponse | null>(null);
  const [expanded, setExpanded] = useState<Set<StageId>>(
    new Set(["E0", "E3", "E5"]),
  );

  useEffect(() => {
    api.stages().then(setStages).catch(() => setStages([]));
    refreshCatalog();
  }, []);

  function refreshCatalog() {
    api.catalog().then(setCatalog).catch(() => setCatalog(null));
  }

  function toggleStage(id: StageId) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  const byStage = new Map<StageId, CatalogResponse["artifacts"]>();
  if (catalog) {
    for (const a of catalog.artifacts) {
      const arr = byStage.get(a.stage) || [];
      arr.push(a);
      byStage.set(a.stage, arr);
    }
  }

  const isActiveStage = (id: StageId) =>
    selection.type === "stage" && selection.stageId === id;
  const isActiveArtifact = (id: string) =>
    selection.type === "artifact" && selection.ref.id === id;

  return (
    <aside className="border-r border-line-soft bg-surface/30 flex flex-col min-h-0">
      <div className="px-3 py-3 border-b border-line-soft flex items-center justify-between">
        <div className="text-2xs uppercase tracking-[0.14em] text-ink-subtle">
          Pipeline
        </div>
        <button
          className="text-2xs text-ink-subtle hover:text-ink-muted"
          onClick={refreshCatalog}
        >
          refrescar
        </button>
      </div>

      <nav className="flex-1 overflow-y-auto px-2 py-2 space-y-0.5">
        <SidebarLink
          icon={<Gauge className="w-3.5 h-3.5" />}
          label="Dashboard de métricas"
          active={selection.type === "metrics"}
          onClick={() => setSelection({ type: "metrics" })}
        />
        <SidebarLink
          icon={<PlayCircle className="w-3.5 h-3.5" />}
          label="Ejecutar clasificación"
          active={selection.type === "runner"}
          onClick={() => setSelection({ type: "runner" })}
        />
        <SidebarLink
          icon={<Terminal className="w-3.5 h-3.5" />}
          label="Stream de logs"
          active={selection.type === "logs"}
          onClick={() => setSelection({ type: "logs" })}
        />

        <div className="divider" />

        {stages.map((s) => {
          const open = expanded.has(s.id);
          const artifacts = byStage.get(s.id) || [];
          return (
            <div key={s.id}>
              <div
                className={
                  "group flex items-center gap-1.5 px-1.5 py-1 rounded-sm cursor-pointer " +
                  (isActiveStage(s.id)
                    ? "bg-surface3 text-ink"
                    : "hover:bg-surface3/60 text-ink-muted")
                }
              >
                <button
                  className="p-0.5 text-ink-subtle hover:text-ink-muted"
                  onClick={() => toggleStage(s.id)}
                >
                  {open ? (
                    <ChevronDown className="w-3.5 h-3.5" />
                  ) : (
                    <ChevronRight className="w-3.5 h-3.5" />
                  )}
                </button>
                <span
                  className={
                    "w-2 h-2 rounded-sm flex-shrink-0 " + stageColors[s.id]
                  }
                />
                <button
                  className="flex-1 flex items-baseline gap-2 text-left"
                  onClick={() =>
                    setSelection({ type: "stage", stageId: s.id })
                  }
                >
                  <span className="font-mono text-xs text-ink-subtle">
                    {s.id}
                  </span>
                  <span className="text-sm truncate">{s.title}</span>
                </button>
                <span className="pill text-2xs">{artifacts.length}</span>
              </div>

              {open && (
                <ul className="ml-6 mt-0.5 mb-1 border-l border-line-soft">
                  {artifacts.slice(0, 60).map((a) => (
                    <li key={a.id}>
                      <button
                        className={
                          "w-full text-left text-2xs font-mono px-2 py-1 hover:text-ink hover:bg-surface3/50 truncate " +
                          (isActiveArtifact(a.id)
                            ? "text-ink bg-surface3"
                            : "text-ink-muted")
                        }
                        onClick={() =>
                          setSelection({ type: "artifact", ref: a })
                        }
                        title={a.path}
                      >
                        {a.label}
                      </button>
                    </li>
                  ))}
                  {artifacts.length > 60 && (
                    <li className="text-2xs text-ink-subtle px-2 py-1 italic">
                      … {artifacts.length - 60} más
                    </li>
                  )}
                  {artifacts.length === 0 && (
                    <li className="text-2xs text-ink-subtle px-2 py-1 italic">
                      sin artefactos
                    </li>
                  )}
                </ul>
              )}
            </div>
          );
        })}

        <div className="divider" />
        <div className="px-2 py-1 flex items-center gap-1.5 text-2xs text-ink-subtle">
          <Boxes className="w-3.5 h-3.5" />
          <span>
            {catalog?.total ?? 0} artefactos descubiertos
          </span>
        </div>
      </nav>
    </aside>
  );
}

function SidebarLink(props: {
  icon: React.ReactNode;
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      className={
        "w-full flex items-center gap-2 px-2 py-1.5 rounded-sm text-sm " +
        (props.active
          ? "bg-surface3 text-ink"
          : "text-ink-muted hover:bg-surface3/60 hover:text-ink")
      }
      onClick={props.onClick}
    >
      <span className="text-ink-subtle">{props.icon}</span>
      <span>{props.label}</span>
    </button>
  );
}
