import { useEffect } from "react";
import { TopBar } from "./components/TopBar";
import { Sidebar } from "./components/Sidebar";
import { RightPanel } from "./components/RightPanel";
import { MainView } from "./components/MainView";
import { api, connectEventStream } from "./lib/api";
import { useApp } from "./state/store";

export function App() {
  const { setProjectRoot, appendEvent, setWsConnected } = useApp();

  useEffect(() => {
    api
      .health()
      .then((h) => setProjectRoot(h.project_root))
      .catch(() => setProjectRoot(""));
  }, [setProjectRoot]);

  useEffect(() => {
    const ws = connectEventStream((e) => appendEvent(e));
    ws.onopen = () => setWsConnected(true);
    ws.onclose = () => setWsConnected(false);
    ws.onerror = () => setWsConnected(false);
    return () => ws.close();
  }, [appendEvent, setWsConnected]);

  return (
    <div className="h-screen flex flex-col">
      <TopBar />
      <div className="flex-1 grid grid-cols-[260px_minmax(0,1fr)_340px] min-h-0">
        <Sidebar />
        <MainView />
        <RightPanel />
      </div>
    </div>
  );
}
