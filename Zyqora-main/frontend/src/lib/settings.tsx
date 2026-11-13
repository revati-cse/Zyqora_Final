import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { setApiBase } from "./api";
import type { Lang } from "./i18n";

export type Theme = "light" | "dark" | "system";
export type VoiceMode = "off" | "browser" | "device";

export interface Settings {
  theme: Theme;
  lang: Lang;
  apiBase: string; // "" = same origin
  pollSeconds: number; // device telemetry interval
  userId: string; // GreenScore participant for new classifications ("" = none)
  voice: VoiceMode;
  visual3d: boolean; // decorative 3D illustration on Overview (never rendered in the Pi's own browser)
}

const DEFAULTS: Settings = { theme: "system", lang: "en", apiBase: "", pollSeconds: 3, userId: "", voice: "browser", visual3d: true };
const KEY = "zyqora.settings";

function load(): Settings {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return DEFAULTS;
    const parsed = JSON.parse(raw) as Partial<Settings>;
    return { ...DEFAULTS, ...parsed, pollSeconds: Math.min(30, Math.max(2, Number(parsed.pollSeconds) || 3)) };
  } catch {
    return DEFAULTS;
  }
}

interface Ctx {
  settings: Settings;
  update: (patch: Partial<Settings>) => void;
}
const SettingsContext = createContext<Ctx | null>(null);

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<Settings>(load);
  setApiBase(settings.apiBase);

  useEffect(() => {
    try {
      localStorage.setItem(KEY, JSON.stringify(settings));
      localStorage.setItem("zyqora.theme", settings.theme);
    } catch {
      /* storage unavailable: settings stay in memory */
    }
    const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
    const apply = () =>
      document.documentElement.classList.toggle("dark", settings.theme === "dark" || (settings.theme === "system" && !!mq?.matches));
    apply();
    document.documentElement.lang = settings.lang;
    mq?.addEventListener?.("change", apply);
    return () => mq?.removeEventListener?.("change", apply);
  }, [settings]);

  const update = useCallback((patch: Partial<Settings>) => setSettings((s) => ({ ...s, ...patch })), []);
  const value = useMemo(() => ({ settings, update }), [settings, update]);
  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>;
}

export function useSettings(): Ctx {
  const ctx = useContext(SettingsContext);
  if (!ctx) throw new Error("useSettings must be used inside SettingsProvider");
  return ctx;
}

export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(() => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false);
  useEffect(() => {
    const mq = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    const on = () => setReduced(mq.matches);
    mq?.addEventListener?.("change", on);
    return () => mq?.removeEventListener?.("change", on);
  }, []);
  return reduced;
}
