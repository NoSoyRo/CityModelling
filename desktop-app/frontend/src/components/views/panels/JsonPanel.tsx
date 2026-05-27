import { useState } from "react";
import { ChevronRight, ChevronDown } from "lucide-react";

export function JsonPanel({ content }: { content: unknown }) {
  return (
    <section className="panel">
      <header className="panel-header">JSON</header>
      <div className="p-3 font-mono text-xs leading-relaxed">
        <Node name="(root)" value={content} depth={0} initiallyOpen />
      </div>
    </section>
  );
}

function Node({
  name,
  value,
  depth,
  initiallyOpen = false,
}: {
  name: string;
  value: unknown;
  depth: number;
  initiallyOpen?: boolean;
}) {
  const [open, setOpen] = useState(initiallyOpen || depth < 1);

  if (value === null) return <Leaf name={name} render="null" tone="muted" />;
  if (typeof value === "number")
    return <Leaf name={name} render={String(value)} tone="num" />;
  if (typeof value === "boolean")
    return <Leaf name={name} render={String(value)} tone="bool" />;
  if (typeof value === "string")
    return (
      <Leaf
        name={name}
        render={'"' + value + '"'}
        tone="str"
      />
    );
  if (Array.isArray(value)) {
    return (
      <Group
        name={name}
        kind={`[ ${value.length} ]`}
        open={open}
        setOpen={setOpen}
      >
        {value.map((v, i) => (
          <Node key={i} name={`[${i}]`} value={v} depth={depth + 1} />
        ))}
      </Group>
    );
  }
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    return (
      <Group
        name={name}
        kind={`{ ${entries.length} }`}
        open={open}
        setOpen={setOpen}
      >
        {entries.map(([k, v]) => (
          <Node key={k} name={k} value={v} depth={depth + 1} />
        ))}
      </Group>
    );
  }
  return <Leaf name={name} render={String(value)} tone="muted" />;
}

function Group({
  name,
  kind,
  open,
  setOpen,
  children,
}: {
  name: string;
  kind: string;
  open: boolean;
  setOpen: (b: boolean) => void;
  children: React.ReactNode;
}) {
  return (
    <div>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 hover:text-ink"
      >
        <span className="text-ink-subtle">
          {open ? (
            <ChevronDown className="w-3 h-3" />
          ) : (
            <ChevronRight className="w-3 h-3" />
          )}
        </span>
        <span className="text-ink">{name}</span>
        <span className="text-ink-subtle ml-2">{kind}</span>
      </button>
      {open && <div className="pl-5 border-l border-line-soft ml-1">{children}</div>}
    </div>
  );
}

function Leaf({
  name,
  render,
  tone,
}: {
  name: string;
  render: string;
  tone: "muted" | "num" | "bool" | "str";
}) {
  const toneClass = {
    muted: "text-ink-muted",
    num: "text-accent",
    bool: "text-warn",
    str: "text-ok",
  }[tone];
  return (
    <div className="flex gap-2">
      <span className="text-ink">{name}</span>
      <span className="text-ink-subtle">:</span>
      <span className={toneClass}>{render}</span>
    </div>
  );
}
