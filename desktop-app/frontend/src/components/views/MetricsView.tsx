import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../../lib/api";
import type { MetricsDashboard } from "../../lib/types";
import { ViewHeader } from "./ViewHeader";
import { fmtNumber } from "../../lib/format";

export function MetricsView() {
  const [data, setData] = useState<MetricsDashboard | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .metrics()
      .then(setData)
      .catch((e) => setError(String(e)));
  }, []);

  if (error) {
    return <div className="p-6 text-danger font-mono text-sm">{error}</div>;
  }
  if (!data) {
    return <div className="p-6 text-ink-muted text-sm">Cargando…</div>;
  }

  const chartData = data.runs.map((r) => ({
    window: r.window,
    FoM: r.fom,
    Kappa: r.kappa,
    IoU: r.iou,
    F1: r.f1,
    accuracy: r.accuracy,
  }));

  return (
    <div>
      <ViewHeader
        eyebrow="Etapa E5 · agregada"
        title="Dashboard de métricas"
        subtitle="Resultados de las cinco ventanas quinquenales 2011–2020 leídos directamente de validation_results.json."
      />

      <div className="p-6 space-y-4">
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          <Stat label="FoM medio" value={data.aggregate.fom} />
          <Stat label="Kappa medio" value={data.aggregate.kappa} />
          <Stat label="IoU medio" value={data.aggregate.iou} />
          <Stat label="Accuracy media" value={data.aggregate.accuracy} />
        </div>

        <section className="panel">
          <header className="panel-header">FoM, Kappa, IoU por ventana</header>
          <div className="h-80 p-4">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid stroke="#1c2127" vertical={false} />
                <XAxis
                  dataKey="window"
                  stroke="#5f6368"
                  fontSize={11}
                  tickLine={false}
                  axisLine={{ stroke: "#262b32" }}
                />
                <YAxis
                  stroke="#5f6368"
                  fontSize={11}
                  tickLine={false}
                  axisLine={{ stroke: "#262b32" }}
                  domain={[0, 1]}
                />
                <Tooltip
                  contentStyle={{
                    background: "#13161a",
                    border: "1px solid #262b32",
                    fontSize: 11,
                  }}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line
                  type="monotone"
                  dataKey="FoM"
                  stroke="#5eead4"
                  strokeWidth={2}
                  dot={{ r: 3, fill: "#5eead4" }}
                />
                <Line
                  type="monotone"
                  dataKey="Kappa"
                  stroke="#fcd34d"
                  strokeWidth={2}
                  dot={{ r: 3, fill: "#fcd34d" }}
                />
                <Line
                  type="monotone"
                  dataKey="IoU"
                  stroke="#f0abfc"
                  strokeWidth={2}
                  dot={{ r: 3, fill: "#f0abfc" }}
                />
                <Line
                  type="monotone"
                  dataKey="F1"
                  stroke="#9aa0a6"
                  strokeWidth={1.5}
                  strokeDasharray="3 3"
                  dot={{ r: 2 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="panel">
          <header className="panel-header">
            Crecimiento urbano observado vs predicho
          </header>
          <div className="h-72 p-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={data.runs.map((r) => ({
                  window: r.window,
                  observado: r.growth_observed,
                  predicho: r.growth_predicted,
                }))}
              >
                <CartesianGrid stroke="#1c2127" vertical={false} />
                <XAxis
                  dataKey="window"
                  stroke="#5f6368"
                  fontSize={11}
                  axisLine={{ stroke: "#262b32" }}
                  tickLine={false}
                />
                <YAxis
                  stroke="#5f6368"
                  fontSize={11}
                  axisLine={{ stroke: "#262b32" }}
                  tickLine={false}
                />
                <Tooltip
                  contentStyle={{
                    background: "#13161a",
                    border: "1px solid #262b32",
                    fontSize: 11,
                  }}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="observado" fill="#5eead4">
                  {data.runs.map((_, i) => (
                    <Cell key={i} fill="#5eead4" />
                  ))}
                </Bar>
                <Bar dataKey="predicho" fill="#fcd34d" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="panel">
          <header className="panel-header">Detalle por ventana</header>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-2xs uppercase tracking-[0.1em] text-ink-subtle">
                <tr>
                  <th className="text-left px-4 py-2">Ventana</th>
                  <th className="text-right px-4 py-2">FoM</th>
                  <th className="text-right px-4 py-2">Kappa</th>
                  <th className="text-right px-4 py-2">IoU</th>
                  <th className="text-right px-4 py-2">F1</th>
                  <th className="text-right px-4 py-2">Acc</th>
                  <th className="text-right px-4 py-2">Threshold</th>
                  <th className="text-right px-4 py-2">Δ urbano (obs)</th>
                  <th className="text-right px-4 py-2">Δ urbano (pred)</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {data.runs.map((r) => (
                  <tr
                    key={r.window}
                    className="border-t border-line-soft hover:bg-surface3/30"
                  >
                    <td className="px-4 py-2">{r.window}</td>
                    <td className="px-4 py-2 text-right num text-accent">
                      {fmtNumber(r.fom, 4)}
                    </td>
                    <td className="px-4 py-2 text-right num">
                      {fmtNumber(r.kappa, 4)}
                    </td>
                    <td className="px-4 py-2 text-right num">
                      {fmtNumber(r.iou, 4)}
                    </td>
                    <td className="px-4 py-2 text-right num">
                      {fmtNumber(r.f1, 4)}
                    </td>
                    <td className="px-4 py-2 text-right num">
                      {fmtNumber(r.accuracy, 4)}
                    </td>
                    <td className="px-4 py-2 text-right num text-ink-muted">
                      {fmtNumber(r.threshold, 3)}
                    </td>
                    <td className="px-4 py-2 text-right num text-ink-muted">
                      {r.growth_observed.toLocaleString()}
                    </td>
                    <td className="px-4 py-2 text-right num text-ink-muted">
                      {r.growth_predicted.toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
}: {
  label: string;
  value: number | undefined;
}) {
  return (
    <div className="panel p-3">
      <div className="field-label">{label}</div>
      <div className="mt-1 text-2xl num text-ink">
        {value !== undefined ? fmtNumber(value, 4) : "—"}
      </div>
    </div>
  );
}
