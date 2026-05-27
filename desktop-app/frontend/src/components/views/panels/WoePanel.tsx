import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type {
  IvStrength,
  WoeModelInspection,
  WoeVariableSummary,
} from "../../../lib/types";
import { fmtNumber } from "../../../lib/format";

const ivColors: Record<IvStrength, string> = {
  no_predictive: "#5f6368",
  weak: "#9aa0a6",
  medium: "#fcd34d",
  strong: "#5eead4",
  very_strong: "#f0abfc",
};
const ivLabel: Record<IvStrength, string> = {
  no_predictive: "sin valor",
  weak: "débil",
  medium: "medio",
  strong: "fuerte",
  very_strong: "muy fuerte",
};

export function WoePanel({ woe }: { woe: WoeModelInspection }) {
  const [selected, setSelected] = useState<string>(
    woe.variables[0]?.name ?? "",
  );
  const current = woe.variables.find((v) => v.name === selected);

  return (
    <section className="panel">
      <header className="panel-header">Modelo WoE</header>
      <div className="p-4 grid grid-cols-1 lg:grid-cols-[280px_minmax(0,1fr)] gap-4">
        <div className="space-y-3">
          <div className="kv">
            <div className="kv-label">entrenamiento</div>
            <div className="kv-value">{woe.trained_years || "—"}</div>
            <div className="kv-label">períodos</div>
            <div className="kv-value">{woe.n_periods ?? "—"}</div>
            <div className="kv-label">transiciones</div>
            <div className="kv-value">
              {woe.total_transitions?.toLocaleString() ?? "—"}
            </div>
            <div className="kv-label">variables</div>
            <div className="kv-value">{woe.variables.length}</div>
          </div>
          <div className="divider" />
          <div className="space-y-1">
            <div className="field-label">Information Value</div>
            {woe.variables.map((v) => (
              <button
                key={v.name}
                onClick={() => setSelected(v.name)}
                className={
                  "w-full px-2 py-1.5 rounded-sm flex items-center gap-2 text-left text-xs " +
                  (selected === v.name
                    ? "bg-surface3 text-ink"
                    : "hover:bg-surface3/60 text-ink-muted hover:text-ink")
                }
              >
                <span
                  className="w-2 h-2 rounded-sm flex-shrink-0"
                  style={{ background: ivColors[v.iv_strength] }}
                />
                <span className="font-mono flex-1 truncate">{v.name}</span>
                <span className="num text-2xs text-ink-subtle">
                  {fmtNumber(v.iv_total, 3)}
                </span>
              </button>
            ))}
          </div>
        </div>

        <div className="min-w-0">
          {current ? <Detail v={current} /> : null}
        </div>
      </div>
    </section>
  );
}

function Detail({ v }: { v: WoeVariableSummary }) {
  const data = v.woe_values.map((value, i) => ({
    bin: i,
    range:
      v.bin_edges[i] !== undefined && v.bin_edges[i + 1] !== undefined
        ? `[${fmtNumber(v.bin_edges[i], 3)}, ${fmtNumber(v.bin_edges[i + 1], 3)})`
        : `bin ${i}`,
    woe: value,
  }));

  return (
    <div className="space-y-3">
      <div className="flex items-baseline justify-between">
        <h3 className="font-mono text-base text-ink">{v.name}</h3>
        <div className="text-2xs text-ink-subtle font-mono">
          IV={fmtNumber(v.iv_total, 4)} ·{" "}
          <span style={{ color: ivColors[v.iv_strength] }}>
            {ivLabel[v.iv_strength]}
          </span>
        </div>
      </div>

      <div className="h-64 panel-quiet">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 12, right: 16, bottom: 8, left: 4 }}>
            <CartesianGrid stroke="#1c2127" vertical={false} />
            <XAxis
              dataKey="bin"
              stroke="#5f6368"
              fontSize={10}
              tickLine={false}
              axisLine={{ stroke: "#262b32" }}
            />
            <YAxis
              stroke="#5f6368"
              fontSize={10}
              tickLine={false}
              axisLine={{ stroke: "#262b32" }}
            />
            <Tooltip
              cursor={{ fill: "rgba(94, 234, 212, 0.06)" }}
              contentStyle={{
                background: "#13161a",
                border: "1px solid #262b32",
                fontSize: 11,
              }}
              formatter={(value: number) => fmtNumber(value, 4)}
              labelFormatter={(_, payload) =>
                payload?.[0]?.payload?.range ?? ""
              }
            />
            <Bar dataKey="woe" radius={[2, 2, 0, 0]}>
              {data.map((d, i) => (
                <Cell
                  key={i}
                  fill={d.woe >= 0 ? "#5eead4" : "#f87171"}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="kv text-xs">
        <div className="kv-label">bins</div>
        <div className="kv-value">{v.n_bins}</div>
        <div className="kv-label">positivas</div>
        <div className="kv-value">{v.positive_samples.toLocaleString()}</div>
        <div className="kv-label">negativas</div>
        <div className="kv-value">{v.negative_samples.toLocaleString()}</div>
      </div>
    </div>
  );
}
