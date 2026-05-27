import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import type { PickleNode } from "../../../lib/types";

export function PickleTreePanel({ node }: { node: PickleNode }) {
  return (
    <section className="panel">
      <header className="panel-header">Estructura del pickle</header>
      <div className="p-3 font-mono text-xs">
        <Row node={node} depth={0} initiallyOpen />
      </div>
    </section>
  );
}

function Row({
  node,
  depth,
  initiallyOpen = false,
}: {
  node: PickleNode;
  depth: number;
  initiallyOpen?: boolean;
}) {
  const [open, setOpen] = useState(initiallyOpen || depth < 1);
  const hasChildren = node.children.length > 0;

  return (
    <div>
      <button
        onClick={() => hasChildren && setOpen(!open)}
        className="flex items-baseline gap-2 hover:text-ink w-full text-left"
      >
        <span className="text-ink-subtle">
          {hasChildren ? (
            open ? (
              <ChevronDown className="w-3 h-3" />
            ) : (
              <ChevronRight className="w-3 h-3" />
            )
          ) : (
            <span className="inline-block w-3 h-3 text-center">·</span>
          )}
        </span>
        <span className="text-ink">{node.name}</span>
        <span className="text-ink-subtle">: {node.type}</span>
        {node.summary && (
          <span className="text-ink-muted ml-2">{node.summary}</span>
        )}
      </button>

      {hasChildren && open && (
        <div className="pl-5 border-l border-line-soft ml-1">
          {node.children.map((c, i) => (
            <Row key={i} node={c} depth={depth + 1} />
          ))}
        </div>
      )}
    </div>
  );
}
