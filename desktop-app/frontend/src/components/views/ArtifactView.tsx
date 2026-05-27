import { useEffect, useState } from "react";
import { api } from "../../lib/api";
import type { ArtifactRef, InspectionResponse } from "../../lib/types";
import { ViewHeader } from "./ViewHeader";
import { NumpyPanel } from "./panels/NumpyPanel";
import { JsonPanel } from "./panels/JsonPanel";
import { PickleTreePanel } from "./panels/PickleTreePanel";
import { WoePanel } from "./panels/WoePanel";
import { ImagePanel } from "./panels/ImagePanel";
import { fmtBytes, fmtDate } from "../../lib/format";
import { Loader2 } from "lucide-react";

export function ArtifactView({ ref }: { ref: ArtifactRef }) {
  const [data, setData] = useState<InspectionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setData(null);
    api
      .inspect(ref)
      .then((r) => !cancelled && setData(r))
      .catch((e) => !cancelled && setError(String(e)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [ref]);

  return (
    <div>
      <ViewHeader
        eyebrow={`${ref.stage} · ${ref.kind.replace(/_/g, " ")}`}
        title={ref.label}
        subtitle={ref.path}
        right={
          <div className="text-right text-2xs font-mono text-ink-subtle">
            <div>{fmtBytes(ref.size_bytes)}</div>
            <div>{fmtDate(ref.modified)}</div>
          </div>
        }
      />

      {loading && (
        <div className="p-6 flex items-center gap-2 text-ink-muted text-sm">
          <Loader2 className="w-4 h-4 animate-spin" />
          Inspeccionando artefacto…
        </div>
      )}

      {error && (
        <div className="p-6 text-sm text-danger font-mono">{error}</div>
      )}

      {data && (
        <div className="p-6 space-y-4">
          {data.notes.length > 0 && (
            <div className="panel-quiet p-3 text-xs text-ink-muted space-y-1">
              {data.notes.map((n, i) => (
                <div key={i}>· {n}</div>
              ))}
            </div>
          )}

          {data.numpy && (
            <NumpyPanel
              stats={data.numpy}
              previewB64={data.image_preview_b64}
            />
          )}

          {data.woe && <WoePanel woe={data.woe} />}

          {data.pickle_tree && !data.woe && (
            <PickleTreePanel node={data.pickle_tree} />
          )}

          {data.json_content && (
            <JsonPanel content={data.json_content.content} />
          )}

          {data.image_preview_b64 && !data.numpy && (
            <ImagePanel b64={data.image_preview_b64} caption={ref.label} />
          )}
        </div>
      )}
    </div>
  );
}
