import { Activity, Menu, Moon, Sun, X } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { NavLink, useLocation } from "react-router";
import { Button } from "@/components/ui/button";
import { useHealth } from "@/hooks/useApi";
import { useSettings } from "@/lib/settings";
import { cn } from "@/lib/utils";
import { NAV } from "./nav";


function Pill({ ok, okText, badText, title }: { ok: boolean | null; okText: string; badText: string; title?: string }) {
  return (
    <span title={title}
      className={cn("inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 font-mono text-[10px] font-medium tracking-wide",
        ok === null ? "bg-surface-2 text-muted-foreground" : ok ? "bg-accept-bg text-accept" : "bg-danger-bg text-danger")}>
      <span aria-hidden className={cn("size-1.5 rounded-full", ok === null ? "bg-muted-foreground" : ok ? "bg-accept" : "bg-danger")} />
      {ok === null ? "CHECKING…" : ok ? okText : badText}
    </span>
  );
}

export function StatusPills() {
  const health = useHealth();
  const connected = health.isSuccess ? true : health.isError ? false : null;
  const modelOk = health.data ? health.data.model_ready : connected === false ? false : null;
  return (
    <div className="flex flex-wrap items-center gap-1.5" aria-live="polite">
      <Pill ok={connected} okText={health.data?.is_raspberry_pi ? "PI CONNECTED" : "DEVICE CONNECTED"} badText="DEVICE DISCONNECTED"
        title={health.data ? `${health.data.device_model ?? health.data.machine}` : undefined} />
      <Pill ok={modelOk} okText="MODEL READY" badText="MODEL UNAVAILABLE" title={health.data?.model_code ?? health.data?.profile ?? undefined} />
    </div>
  );
}

function ThemeButton() {
  const { settings, update } = useSettings();
  const dark = settings.theme === "dark" || (settings.theme === "system" && !!window.matchMedia?.("(prefers-color-scheme: dark)").matches);
  return (
    <Button variant="ghost" size="icon" aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}
      onClick={() => update({ theme: dark ? "light" : "dark" })} data-theme={settings.theme}>
      {dark ? <Sun /> : <Moon />}
    </Button>
  );
}

function LangButton() {
  const { settings, update } = useSettings();
  return (
    <Button variant="ghost" size="sm" aria-label="Switch language" onClick={() => update({ lang: settings.lang === "en" ? "ta" : "en" })}
      className="font-mono">
      {settings.lang === "en" ? "EN" : <span className="font-tamil">தமிழ்</span>}
    </Button>
  );
}

function Brand() {
  return (
    <div className="flex items-center gap-2.5">
      <svg viewBox="0 0 32 32" className="size-7" aria-hidden>
        <rect width="32" height="32" rx="8" className="fill-primary" />
        <path d="M10 11h12l-9 10h9" fill="none" className="stroke-primary-foreground" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      <div className="leading-tight">
        <p className="text-sm font-semibold tracking-tight">Zyqora AI</p>
        <p className="font-mono text-[10px] text-muted-foreground">mixed-waste AI · 4.0</p>
      </div>
    </div>
  );
}

function NavItems({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <nav aria-label="Main" className="flex flex-col gap-0.5">
      {NAV.map(({ to, label, icon: Icon }) => (
        <NavLink key={to} to={to} end={to === "/"} onClick={onNavigate}
          className={({ isActive }) => cn("flex items-center gap-2.5 rounded-md px-3 py-2 text-sm transition-colors",
            isActive ? "bg-surface-2 font-medium text-foreground" : "text-muted-foreground hover:bg-surface-2 hover:text-foreground")}>
          <Icon className="size-4" aria-hidden />
          {label}
        </NavLink>
      ))}
    </nav>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const location = useLocation();
  useEffect(() => setOpen(false), [location.pathname]);
  return (
    <div className="min-h-dvh lg:grid lg:grid-cols-[240px_1fr] lg:bg-[linear-gradient(to_right,var(--surface)_240px,transparent_240px)]">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 focus:rounded focus:bg-surface focus:px-3 focus:py-2">
        Skip to content
      </a>
      <aside className="sticky top-0 hidden h-dvh flex-col gap-6 bg-surface px-3 py-5 lg:flex">
        <div className="px-2"><Brand /></div>
        <NavItems />
        <div className="mt-auto px-2 text-[11px] leading-relaxed text-muted-foreground">
          <Activity className="mb-1 size-3.5" aria-hidden />
          Inference runs on the connected device with ExecuTorch. No cloud services.
        </div>
      </aside>
      <div className="min-w-0 lg:border-l lg:border-border">
        <header className="sticky top-0 z-30 flex items-center gap-2 border-b border-border bg-background/90 px-4 py-2.5 backdrop-blur sm:px-6">
          <Button variant="ghost" size="icon" className="lg:hidden" aria-label={open ? "Close menu" : "Open menu"} aria-expanded={open}
            onClick={() => setOpen((o) => !o)}>
            {open ? <X /> : <Menu />}
          </Button>
          <div className="lg:hidden"><Brand /></div>
          <div className="ml-auto flex items-center gap-1 sm:gap-2">
            <div className="hidden sm:block"><StatusPills /></div>
            <LangButton />
            <ThemeButton />
          </div>
        </header>
        {open && (
          <div className="border-b border-border bg-surface px-3 py-3 lg:hidden">
            <div className="mb-3 px-1 sm:hidden"><StatusPills /></div>
            <NavItems onNavigate={() => setOpen(false)} />
          </div>
        )}
        <main id="main" className="mx-auto w-full max-w-6xl px-4 py-6 sm:px-6 sm:py-8">{children}</main>
      </div>
    </div>
  );
}
