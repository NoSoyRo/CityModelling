import { useEffect, useMemo, useState } from "react";
import { api } from "../../lib/api";
import type { ArtifactRef, JobEvent } from "../../lib/types";
import { ViewHeader } from "./ViewHeader";
import { useApp } from "../../state/store";
import { Play, Loader2 } from "lucide-react";
import { fmtNumber } from "../../lib/format";

export function RunnerView() {
  const { events, activeJobId, setActiveJobId } = useApp();
  const [rawImages, setRawImages] = useState<ArtifactRef[]>([]);
  const [selectedImage, setSelectedImage] = useState<string>("");
  const [sampleSize, setSampleSize] = useState<number>(5000);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .catalog()
      .then((c) => {
        const imgs = c.artifacts.filter((a) => a.kind === "raw_image");
        setRawImages(imgs);
        if (imgs.length > 0 && !selectedImage) setSelectedImage(imgs[0].path);
      })
      .catch(() => setRawImages([]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const jobEvents = useMemo(
    () => events.filter((e) => e.job_id === activeJobId),
    [events, activeJobId],
  );

  const isRunning =
    activeJobId !== null &&
    !jobEvents.some((e) => e.level === "done");

  async function launch() {
    if (!selectedImage) return;
    setSubmitting(true);
    setError(null);
    try {
      const ack = await api.submitJob("classify_image", {
        image: selectedImage,
        sample_size: sampleSize,
      });
      setActiveJobId(ack.job_id);
    } catch (e) {
      setError(String(e));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div>
      <ViewHeader
        eyebrow="Etapa E1 + E2 · ejecución en vivo"
        title="Ejecutar clasificación de una imagen"
        subtitle="Lanza el pipeline 23 features → PCA(8) → K-Means → SVM → estandarización NDVI sobre un PNG crudo. Cada paso emite métricas que ves a la derecha."
      />

      <div className="p-6 grid grid-cols-1 lg:grid-cols-[400px_minmax(0,1fr)] gap-4">
        <section className="panel">
          <header className="panel-header">Parámetros</header>
          <div className="p-4 space-y-4">
            <div>
              <div className="field-label mb-1">Imagen RGB</div>
              <select
                className="input"
                value={selectedImage}
                onChange={(e) => setSelectedImage(e.target.value)}
              >
                {rawImages.map((a) => (
                  <option key={a.path} value={a.path}>
                    {a.label} — {a.path.split("/").slice(-2).join("/")}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <div className="field-label mb-1">SVM sample size</div>
              <input
                type="number"
                className="input"
                value={sampleSize}
                min={500}
                max={50000}
                step={500}
                onChange={(e) => setSampleSize(Number(e.target.value))}
              />
              <div className="text-2xs text-ink-subtle mt-1">
                Píxeles que se usan para entrenar el SVM lineal por imagen.
                Más alto = más exacto, más lento.
              </div>
            </div>

            <div>
              <button
                className="btn-primary w-full justify-center"
                onClick={launch}
                disabled={submitting || isRunning || !selectedImage}
              >
                {submitting || isRunning ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Play className="w-4 h-4" />
                )}
                {isRunning ? "Ejecutando…" : "Lanzar job"}
              </button>
              {error && (
                <div className="mt-2 text-2xs text-danger font-mono">
                  {error}
                </div>
              )}
            </div>

            <div className="divider" />

            <div className="text-2xs text-ink-subtle leading-relaxed">
              Cada job se guarda en{" "}
              <span className="font-mono text-ink-muted">
                desktop-app/runs/&lt;job_id&gt;/
              </span>{" "}
              con un manifest JSON, los dos arrays binarios producidos
              (crudo y estandarizado) y los tiempos por paso.
            </div>
          </div>
        </section>

        <section className="panel">
          <header className="panel-header">
            {activeJobId
              ? `Job ${activeJobId.slice(0, 12)} · ${jobEvents.length} eventos`
              : "Sin job activo"}
          </header>
          <div className="p-4 space-y-3">
            <Timeline events={jobEvents} />
            <ResultsSummary events={jobEvents} />
          </div>
        </section>
      </div>
    </div>
  );
}

const stepOrder = [
  "setup",
  "load_image",
  "features",
  "pca",
  "clustering",
  "standardize",
  "persist",
];
const stepLabel: Record<string, string> = {
  setup: "Setup",
  load_image: "Cargar imagen",
  features: "Extraer 23 features",
  pca: "Scaler + PCA(8)",
  clustering: "K-Means + SVM",
  standardize: "Estandarizar labels",
  persist: "Persistir artefactos",
};

function Timeline({ events }: { events: JobEvent[] }) {
  return (
    <ul className="space-y-2">
      {stepOrder.map((step) => {
        const ev = events.find(
          (e) => e.step === step && e.level === "metric",
        );
        const errored = events.find(
          (e) => e.step === step && e.level === "error",
        );
        const status = errored
          ? "error"
          : ev
            ? "done"
            : events.find((e) => e.step === step)
              ? "running"
              : "pending";
        const elapsed = ev?.payload?.elapsed_ms as number | undefined;

        return (
          <li
            key={step}
            className="flex items-center gap-3 text-sm border border-line-soft rounded-sm px-3 py-2 bg-surface/30"
          >
            <StatusDot status={status} />
            <span className="font-mono text-2xs text-ink-subtle w-[60px]">
              {step}
            </span>
            <span className="flex-1">{stepLabel[step] || step}</span>
            <span className="num text-2xs text-ink-muted">
              {elapsed !== undefined ? `${elapsed} ms` : ""}
            </span>
          </li>
        );
      })}
    </ul>
  );
}

function StatusDot({
  status,
}: {
  status: "pending" | "running" | "done" | "error";
}) {
  const cls = {
    pending: "bg-line",
    running: "bg-warn animate-pulse",
    done: "bg-ok",
    error: "bg-danger",
  }[status];
  return <span className={"w-2 h-2 rounded-sm " + cls} />;
}

function ResultsSummary({ events }: { events: JobEvent[] }) {
  const featuresEv = events.find(
    (e) => e.step === "features" && e.level === "metric",
  );
  const pcaEv = events.find((e) => e.step === "pca" && e.level === "metric");
  const clusteringEv = events.find(
    (e) => e.step === "clustering" && e.level === "metric",
  );
  const stdEv = events.find(
    (e) => e.step === "standardize" && e.level === "metric",
  );
  const persistEv = events.find(
    (e) => e.step === "persist" && e.level === "artifact",
  );

  if (!featuresEv && !pcaEv) return null;

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mt-2">
      <Card label="Features extraídas">
        {featuresEv ? (
          <>
            <div className="num text-lg">
              {featuresEv.payload?.n_features as number}
            </div>
            <div className="text-2xs text-ink-subtle">
              X: {JSON.stringify(featuresEv.payload?.X_shape)}
            </div>
          </>
        ) : (
          <Pending />
        )}
      </Card>
      <Card label="PCA varianza">
        {pcaEv ? (
          <>
            <div className="num text-lg">
              {(
                ((pcaEv.payload?.explained_variance_ratio as number) || 0) *
                100
              ).toFixed(1)}
              %
            </div>
            <div className="text-2xs text-ink-subtle">
              {JSON.stringify(pcaEv.payload?.X_pca_shape)}
            </div>
          </>
        ) : (
          <Pending />
        )}
      </Card>
      <Card label="SVM test acc">
        {clusteringEv ? (
          <>
            <div className="num text-lg">
              {fmtNumber(
                clusteringEv.payload?.svm_test_accuracy as number,
                4,
              )}
            </div>
            <div className="text-2xs text-ink-subtle">
              inertia ={" "}
              {fmtNumber(clusteringEv.payload?.inertia as number, 0)}
            </div>
          </>
        ) : (
          <Pending />
        )}
      </Card>
      <Card label="% urbano">
        {stdEv ? (
          <>
            <div className="num text-lg text-accent">
              {fmtNumber(stdEv.payload?.urban_percentage as number, 2)}%
            </div>
            <div className="text-2xs text-ink-subtle">
              flip = {String(stdEv.payload?.flipped ?? false)}
            </div>
          </>
        ) : (
          <Pending />
        )}
      </Card>
      {persistEv && (
        <div className="col-span-2 lg:col-span-4 panel-quiet p-3 text-2xs font-mono text-ink-muted">
          <div className="text-ink-subtle field-label mb-1">Manifest</div>
          {String(persistEv.payload?.manifest)}
        </div>
      )}
    </div>
  );
}

function Card({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="panel p-3">
      <div className="field-label">{label}</div>
      <div className="mt-1">{children}</div>
    </div>
  );
}

function Pending() {
  return <div className="text-sm text-ink-subtle">—</div>;
}
