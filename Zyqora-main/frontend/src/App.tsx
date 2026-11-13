import { QueryClientProvider } from "@tanstack/react-query";
import { lazy, Suspense } from "react";
import { BrowserRouter, Link, Route, Routes } from "react-router";
import { EmptyState } from "@/components/common";
import { AppShell } from "@/components/layout/AppShell";
import { Skeleton } from "@/components/ui/misc";
import { makeQueryClient } from "@/lib/queryClient";
import { SettingsProvider } from "@/lib/settings";

// Each page is its own chunk; charts and the 3D scene load only where used.
const Overview = lazy(() => import("@/pages/Overview"));
const Live = lazy(() => import("@/pages/Live"));
const Batch = lazy(() => import("@/pages/Batch"));
const Analytics = lazy(() => import("@/pages/Analytics"));
const GreenScore = lazy(() => import("@/pages/GreenScore"));
const ModelPerformance = lazy(() => import("@/pages/ModelPerformance"));
const Device = lazy(() => import("@/pages/Device"));
const History = lazy(() => import("@/pages/History"));
const Settings = lazy(() => import("@/pages/Settings"));

function NotFound() {
  return <EmptyState title="Page not found"><Link className="underline" to="/">Back to overview</Link></EmptyState>;
}

export function AppRoutes() {
  return (
    <AppShell>
      <Suspense fallback={<div className="space-y-4"><Skeleton className="h-8 w-48" /><Skeleton className="h-64" /></div>}>
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/classify" element={<Live />} />
          <Route path="/batch" element={<Batch />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/greenscore" element={<GreenScore />} />
          <Route path="/model" element={<ModelPerformance />} />
          <Route path="/device" element={<Device />} />
          <Route path="/history" element={<History />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </Suspense>
    </AppShell>
  );
}

const queryClient = makeQueryClient();

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <SettingsProvider>
        <BrowserRouter>
          <AppRoutes />
        </BrowserRouter>
      </SettingsProvider>
    </QueryClientProvider>
  );
}
