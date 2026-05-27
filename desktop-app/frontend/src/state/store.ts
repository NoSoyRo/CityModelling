import { create } from "zustand";
import type { ArtifactRef, JobEvent, StageId } from "../lib/types";

export type Selection =
  | { type: "stage"; stageId: StageId }
  | { type: "artifact"; ref: ArtifactRef }
  | { type: "metrics" }
  | { type: "runner" }
  | { type: "logs" };

interface AppState {
  selection: Selection;
  setSelection: (s: Selection) => void;

  projectRoot: string;
  setProjectRoot: (r: string) => void;

  events: JobEvent[];
  appendEvent: (e: JobEvent) => void;
  clearEvents: () => void;

  activeJobId: string | null;
  setActiveJobId: (id: string | null) => void;

  wsConnected: boolean;
  setWsConnected: (b: boolean) => void;
}

export const useApp = create<AppState>((set) => ({
  selection: { type: "stage", stageId: "E0" },
  setSelection: (s) => set({ selection: s }),

  projectRoot: "",
  setProjectRoot: (r) => set({ projectRoot: r }),

  events: [],
  appendEvent: (e) =>
    set((state) => ({
      events:
        state.events.length >= 4000
          ? [...state.events.slice(-3999), e]
          : [...state.events, e],
    })),
  clearEvents: () => set({ events: [] }),

  activeJobId: null,
  setActiveJobId: (id) => set({ activeJobId: id }),

  wsConnected: false,
  setWsConnected: (b) => set({ wsConnected: b }),
}));
