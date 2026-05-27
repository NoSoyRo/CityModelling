import { fmtBytes, fmtNumber, fmtShape } from "../../../lib/format";
import type { NumpyStats } from "../../../lib/types";

export function NumpyPanel({
  stats,
  previewB64,
}: {
  stats: NumpyStats;
  previewB64: string | null;
}) {
  return (
    <section className="panel">
      <header className="panel-header">Array (numpy)</header>
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-4 p-4">
        <div className="kv">
          <div className="kv-label">shape</div>
          <div className="kv-value">{fmtShape(stats.shape)}</div>
          <div className="kv-label">dtype</div>
          <div className="kv-value">{stats.dtype}</div>
          <div className="kv-label">size</div>
          <div className="kv-value">{stats.size.toLocaleString()}</div>
          <div className="kv-label">memory</div>
          <div className="kv-value">{fmtBytes(stats.nbytes)}</div>
          <div className="kv-label">min</div>
          <div className="kv-value">{fmtNumber(stats.min)}</div>
          <div className="kv-label">max</div>
          <div className="kv-value">{fmtNumber(stats.max)}</div>
          <div className="kv-label">mean</div>
          <div className="kv-value">{fmtNumber(stats.mean)}</div>
          {stats.unique_count !== null && (
            <>
              <div className="kv-label">únicos</div>
              <div className="kv-value">{stats.unique_count}</div>
            </>
          )}
          {stats.distribution && (
            <>
              <div className="kv-label">histograma</div>
              <div className="kv-value">
                <div className="space-y-1">
                  {Object.entries(stats.distribution).map(([k, v]) => (
                    <Bar
                      key={k}
                      label={k}
                      value={v}
                      total={stats.size}
                    />
                  ))}
                </div>
              </div>
            </>
          )}
        </div>

        {previewB64 && (
          <div>
            <div className="text-2xs uppercase tracking-[0.1em] text-ink-subtle mb-2">
              Preview
            </div>
            <div className="bg-base border border-line-soft rounded-sm overflow-hidden">
              <img
                src={`data:image/png;base64,${previewB64}`}
                alt="preview"
                className="w-full h-auto block"
              />
            </div>
            <div className="mt-2 text-2xs text-ink-subtle">
              Downsampled. Para análisis cuantitativo, usa el array directo.
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

function Bar({
  label,
  value,
  total,
}: {
  label: string;
  value: number;
  total: number;
}) {
  const pct = total > 0 ? (value / total) * 100 : 0;
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-12 font-mono text-ink-muted">{label}</span>
      <div className="flex-1 h-2 bg-surface3 rounded-sm overflow-hidden">
        <div
          className="h-full bg-accent/60"
          style={{ width: `${Math.min(100, pct)}%` }}
        />
      </div>
      <span className="w-24 text-right num text-ink-muted">
        {value.toLocaleString()} · {pct.toFixed(2)}%
      </span>
    </div>
  );
}
