import { useEffect, useMemo, useRef } from "react";
import { useApp } from "../state/store";
import type { JobEventLevel } from "../lib/types";

const levelColor: Record<JobEventLevel, string> = {
  info: "text-ink-muted",
  debug: "text-ink-subtle",
  warn: "text-warn",
  error: "text-danger",
  metric: "text-accent",
  artifact: "text-ok",
  progress: "text-ink",
  done: "text-ink-muted",
};

const levelGlyph: Record<JobEventLevel, string> = {
  info: "·",
  debug: "·",
  warn: "!",
  error: "×",
  metric: "→",
  artifact: "◆",
  progress: "›",
  done: "✓",
};

export function RightPanel() {
  const { events, activeJobId, clearEvents } = useApp();

  const filtered = useMemo(
    () =>
      activeJobId ? events.filter((e) => e.job_id === activeJobId) : events,
    [events, activeJobId],
  );

  const scrollRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [filtered.length]);

  return (
    <aside className="border-l border-line-soft bg-surface/30 flex flex-col min-h-0">
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-line-soft">
        <div className="text-2xs uppercase tracking-[0.14em] text-ink-subtle">
          Event stream
        </div>
        <div className="flex items-center gap-2">
          <span className="text-2xs text-ink-subtle font-mono">
            {activeJobId ? activeJobId.slice(0, 8) : "global"}
          </span>
          <button
            className="text-2xs text-ink-subtle hover:text-ink-muted"
            onClick={clearEvents}
          >
            limpiar
          </button>
        </div>
      </div>

      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto font-mono text-2xs leading-relaxed px-3 py-2 space-y-1"
      >
        {filtered.length === 0 && (
          <div className="text-ink-subtle italic">
            Esperando eventos. Lanza un job desde "Ejecutar clasificación".
          </div>
        )}
        {filtered.map((e, i) => (
          <div key={i} className="flex gap-2">
            <span className="text-ink-subtle shrink-0 w-[64px]">
              {formatTime(e.ts)}
            </span>
            <span className={"w-3 shrink-0 " + levelColor[e.level]}>
              {levelGlyph[e.level]}
            </span>
            <span className="text-ink-subtle w-[64px] shrink-0 truncate">
              {e.step || "—"}
            </span>
            <span className={levelColor[e.level] + " whitespace-pre-wrap break-words"}>
              {e.message}
            </span>
          </div>
        ))}
      </div>

      <div className="border-t border-line-soft px-3 py-1.5 text-2xs text-ink-subtle flex items-center justify-between">
        <span>{filtered.length} eventos</span>
        <span className="font-mono">/ws</span>
      </div>
    </aside>
  );
}

function formatTime(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toISOString().slice(11, 19);
  } catch {
    return iso.slice(0, 19);
  }
}
