import { useMemo, useState } from "react";
import { ViewHeader } from "./ViewHeader";
import { useApp } from "../../state/store";
import type { JobEventLevel } from "../../lib/types";

const LEVELS: JobEventLevel[] = [
  "info",
  "metric",
  "artifact",
  "progress",
  "warn",
  "error",
  "done",
];

export function LogsView() {
  const { events, clearEvents } = useApp();
  const [enabled, setEnabled] = useState<Set<JobEventLevel>>(
    new Set(LEVELS),
  );
  const [query, setQuery] = useState("");

  const filtered = useMemo(
    () =>
      events.filter(
        (e) =>
          enabled.has(e.level) &&
          (query === "" ||
            (e.message + " " + (e.step || "") + " " + e.job_id)
              .toLowerCase()
              .includes(query.toLowerCase())),
      ),
    [events, enabled, query],
  );

  function toggle(l: JobEventLevel) {
    setEnabled((prev) => {
      const next = new Set(prev);
      if (next.has(l)) next.delete(l);
      else next.add(l);
      return next;
    });
  }

  return (
    <div>
      <ViewHeader
        eyebrow="Diagnóstico"
        title="Stream de logs"
        subtitle="Todos los eventos de WebSocket recibidos en esta sesión, filtrables por nivel."
      />
      <div className="p-6 space-y-3">
        <div className="flex items-center gap-2 flex-wrap">
          {LEVELS.map((l) => (
            <button
              key={l}
              onClick={() => toggle(l)}
              className={
                "pill " +
                (enabled.has(l)
                  ? "border-accent/50 text-accent"
                  : "text-ink-subtle")
              }
            >
              {l}
            </button>
          ))}
          <div className="flex-1" />
          <input
            placeholder="filtrar texto…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="input w-64"
          />
          <button className="btn-ghost" onClick={clearEvents}>
            limpiar
          </button>
        </div>

        <section className="panel">
          <header className="panel-header flex items-center justify-between">
            <span>{filtered.length} eventos</span>
            <span className="font-mono">/ws</span>
          </header>
          <div className="font-mono text-xs leading-relaxed">
            {filtered.length === 0 && (
              <div className="p-4 text-ink-subtle italic">
                No hay eventos que coincidan con el filtro.
              </div>
            )}
            {filtered.map((e, i) => (
              <div
                key={i}
                className="grid grid-cols-[80px_60px_80px_80px_minmax(0,1fr)] gap-3 px-3 py-1.5 border-b border-line-soft/60 hover:bg-surface3/30"
              >
                <span className="text-ink-subtle">
                  {e.ts.slice(11, 19)}
                </span>
                <span
                  className={
                    e.level === "error"
                      ? "text-danger"
                      : e.level === "warn"
                        ? "text-warn"
                        : e.level === "metric"
                          ? "text-accent"
                          : e.level === "artifact"
                            ? "text-ok"
                            : "text-ink-muted"
                  }
                >
                  {e.level}
                </span>
                <span className="text-ink-muted truncate">
                  {e.step || "—"}
                </span>
                <span className="text-ink-subtle truncate">
                  {e.job_id.slice(0, 8)}
                </span>
                <span className="text-ink whitespace-pre-wrap break-words">
                  {e.message}
                </span>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
