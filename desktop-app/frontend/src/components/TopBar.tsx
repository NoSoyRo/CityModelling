import { useApp } from "../state/store";
import { Activity, Cpu, FolderOpenDot } from "lucide-react";

export function TopBar() {
  const { projectRoot, wsConnected, events, activeJobId } = useApp();
  const liveEvents = activeJobId
    ? events.filter((e) => e.job_id === activeJobId).length
    : events.length;

  return (
    <header className="border-b border-line-soft bg-surface/60 backdrop-blur">
      <div className="flex items-center gap-4 px-4 h-12">
        <div className="flex items-center gap-2.5">
          <Logo />
          <div className="flex items-baseline gap-2">
            <h1 className="text-sm tracking-tightish font-medium">
              Querétaro Urban Lab
            </h1>
            <span className="text-2xs text-ink-subtle font-mono">v0.1.0</span>
          </div>
        </div>

        <div className="h-5 w-px bg-line-soft" />

        <div className="flex items-center gap-1.5 text-2xs text-ink-muted">
          <FolderOpenDot className="w-3.5 h-3.5 text-ink-subtle" />
          <span className="font-mono">
            {projectRoot
              ? projectRoot.replace(/^.*\/([^/]+\/[^/]+)$/, "…/$1")
              : "—"}
          </span>
        </div>

        <div className="flex-1" />

        <div className="flex items-center gap-3 text-2xs text-ink-muted">
          <div className="flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5 text-ink-subtle" />
            <span>tesis_ac</span>
          </div>
          <div className="flex items-center gap-1.5">
            <Activity
              className={
                "w-3.5 h-3.5 " +
                (wsConnected ? "text-ok" : "text-danger")
              }
            />
            <span className="font-mono">
              {wsConnected ? "stream ok" : "sin stream"}
            </span>
          </div>
          <div className="pill-strong">
            <span className="num">{liveEvents}</span>
            <span className="text-ink-subtle">events</span>
          </div>
        </div>
      </div>
    </header>
  );
}

function Logo() {
  return (
    <svg
      width="22"
      height="22"
      viewBox="0 0 32 32"
      fill="none"
      className="text-accent"
    >
      <rect width="32" height="32" rx="6" fill="#13161a" />
      <g
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="square"
        fill="none"
      >
        <path d="M5 22 L5 11 L11 11 L11 22 Z" />
        <path d="M13 22 L13 14 L19 14 L19 22 Z" />
        <path d="M21 22 L21 17 L27 17 L27 22 Z" />
        <path d="M5 25 L27 25" />
      </g>
    </svg>
  );
}
