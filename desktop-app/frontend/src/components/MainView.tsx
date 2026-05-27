import { useApp } from "../state/store";
import { StageView } from "./views/StageView";
import { ArtifactView } from "./views/ArtifactView";
import { MetricsView } from "./views/MetricsView";
import { RunnerView } from "./views/RunnerView";
import { LogsView } from "./views/LogsView";

export function MainView() {
  const { selection } = useApp();
  return (
    <main className="overflow-y-auto min-h-0">
      {selection.type === "stage" && <StageView stageId={selection.stageId} />}
      {selection.type === "artifact" && <ArtifactView ref={selection.ref} />}
      {selection.type === "metrics" && <MetricsView />}
      {selection.type === "runner" && <RunnerView />}
      {selection.type === "logs" && <LogsView />}
    </main>
  );
}
