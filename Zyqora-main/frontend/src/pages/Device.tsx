import { useEffect, useRef, useState } from "react";
import { Sparkline } from "@/components/charts/Charts";
import { ErrorState, PageHeader, ScopeTag } from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/misc";
import { useSystemMetrics } from "@/hooks/useApi";
import { useSettings } from "@/lib/settings";
import { fmtMs, fmtUptime } from "@/lib/utils";
import type { SystemMetrics } from "@/types/mixed";

type Sample = { t: number; cpu: number | null; mem: number | null; temp: number | null; rss: number | null };
const NA = <span className="text-sm font-normal text-muted-foreground">not available on this device</span>;

function Stat({ label, value, children }: { label: string; value: React.ReactNode; children?: React.ReactNode }) {
  return (
    <Card>
      <CardContent className="p-4">
        <p className="text-xs text-muted-foreground">{label}</p>
        <p className="mt-1 font-mono text-2xl font-semibold tabular">{value}</p>
        {children && <div className="mt-2">{children}</div>}
      </CardContent>
    </Card>
  );
}

export default function Device() {
  const { settings } = useSettings();
  const q = useSystemMetrics();
  const d: SystemMetrics | undefined = q.data;
  const [samples, setSamples] = useState<Sample[]>([]);
  const last = useRef<number>(0);

  useEffect(() => {
    if (!d || d.sampled_at === last.current) return;
    last.current = d.sampled_at;
    setSamples((s) => [...s.slice(-59), { t: d.sampled_at, cpu: d.cpu_percent, mem: d.memory.used_percent, temp: d.temperature_c, rss: d.process_rss_mb }]);
  }, [d]);

  return (
    <>
      <PageHeader title="System Monitoring"
        description={<>Telemetry read from /proc and /sys on the device every {settings.pollSeconds} s (change in Settings); polling pauses while this tab is hidden. Sparklines show this browser session; samples are also stored for Analytics. <ScopeTag>live</ScopeTag></>} />
      {q.isError && <div className="mb-4"><ErrorState error={q.error} title="DEVICE DISCONNECTED" /></div>}
      {q.isLoading ? <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">{[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-28" />)}</div> : d && (
        <>
          <div className="mb-6 flex flex-wrap items-center gap-2 text-sm">
            <Badge variant={q.isError ? "danger" : "accept"}>{q.isError ? "DEVICE DISCONNECTED" : "DEVICE CONNECTED"}</Badge>
            <Badge variant={d.is_raspberry_pi ? "accept" : "review"}>{d.device_model ?? "unknown model"}{d.is_raspberry_pi ? "" : " · not a Raspberry Pi"}</Badge>
            <Badge variant={d.backend_status === "ok" ? "accept" : "danger"}>BACKEND {d.backend_status.toUpperCase()}</Badge>
            <Badge variant={d.model_ready ? "accept" : "danger"}>{d.model_ready ? "CLASSIFIER READY" : "CLASSIFIER UNAVAILABLE"}</Badge>
            <Badge variant={d.detector_ready ? "accept" : "danger"}>{d.detector_ready ? "DETECTOR READY" : "DETECTOR UNAVAILABLE"}</Badge>
            {d.model_profile && <Badge variant="outline">{d.model_profile}</Badge>}
          </div>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Stat label="CPU utilisation (all cores)" value={d.cpu_percent == null ? NA : `${d.cpu_percent.toFixed(1)}%`}>
              {samples.length > 1 && <Sparkline data={samples} dataKey="cpu" unit="%" domain={[0, 100]} />}
            </Stat>
            <Stat label="RAM used" value={d.memory.used_percent == null ? NA : `${d.memory.used_percent.toFixed(1)}%`}>
              <p className="text-xs text-muted-foreground">
                {d.memory.used_mb != null && d.memory.total_mb != null ? `${(d.memory.used_mb / 1024).toFixed(2)} / ${(d.memory.total_mb / 1024).toFixed(1)} GiB` : ""}
              </p>
              {samples.length > 1 && <Sparkline data={samples} dataKey="mem" unit="%" />}
            </Stat>
            <Stat label="CPU temperature" value={d.temperature_c == null ? NA : `${d.temperature_c.toFixed(1)} °C`}>
              {samples.length > 1 && d.temperature_c != null && <Sparkline data={samples} dataKey="temp" unit=" °C" />}
            </Stat>
            <Stat label="Zyqora process memory" value={d.process_rss_mb == null ? NA : `${d.process_rss_mb.toFixed(0)} MB`}>
              <p className="text-xs text-muted-foreground">peak {d.process_peak_rss_mb?.toFixed(0) ?? "—"} MB · includes Python, torch and ExecuTorch</p>
            </Stat>
          </div>
          <div className="mt-6 grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader><CardTitle>Inference runtime</CardTitle><CardDescription>Model loaded once in the API process. Single requests on an idle Pi run at a lower clock than back-to-back benchmark runs.</CardDescription></CardHeader>
              <CardContent>
                <dl className="grid grid-cols-2 gap-3 text-sm">
                  {[
                    ["Runtime", d.runtime ?? "—"], ["Backend", d.backend ?? "—"], ["Threads", d.threads ?? "—"],
                    ["Last inference", fmtMs(d.last_inference_ms, 2)], ["CPU cores", d.cpu_count ?? "—"],
                    ["Classifier load + warm-up", fmtMs(d.model_load_ms, 0)],
                    ["Detector", d.detector_profile ?? "not loaded"], ["Detector runtime", d.detector_runtime ?? "—"],
                    ["Detector load + warm-up", fmtMs(d.detector_load_ms, 0)],
                    ["CPU clock now", d.cpufreq?.cpu_mhz ? `${d.cpufreq.cpu_mhz.toFixed(0)} / ${d.cpufreq.cpu_max_mhz?.toFixed(0) ?? "?"} MHz` : "not available"],
                    ["Frequency governor", d.cpufreq?.governor ?? "not available"],
                    ["Load average", d.load_average ? d.load_average.map((x) => x.toFixed(2)).join(" / ") : "—"],
                  ].map(([k, v]) => <div key={String(k)}><dt className="text-xs text-muted-foreground">{k}</dt><dd className="font-mono">{v}</dd></div>)}
                </dl>
              </CardContent>
            </Card>
            <Card>
              <CardHeader><CardTitle>Hardware interfaces</CardTitle><CardDescription>Uptime {fmtUptime(d.uptime_s)}</CardDescription></CardHeader>
              <CardContent className="space-y-3 text-sm">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="w-20 text-xs text-muted-foreground">Camera</span>
                  <Badge variant="outline">{d.camera.configured_backend}</Badge>
                  <Badge variant={d.camera.open ? "accept" : "neutral"}>{d.camera.open ? (d.camera.simulated ? "open · SIMULATION" : "open") : "not opened yet"}</Badge>
                </div>
                {d.camera.last_error && <p className="text-xs text-danger">{d.camera.last_error}</p>}
                <div className="flex flex-wrap items-center gap-2">
                  <span className="w-20 text-xs text-muted-foreground">Sorter</span>
                  <Badge variant={d.sorter.hardware ? "review" : "neutral"}>{d.sorter.name}{d.sorter.hardware ? " · HARDWARE ENABLED" : " · SIMULATION"}</Badge>
                </div>
                <p className="text-xs text-muted-foreground">REVIEW items never actuate a bin. Physical sorting is not verified until tested on hardware.</p>
                <div className="flex flex-wrap items-center gap-2 border-t border-border pt-3">
                  <span className="w-20 text-xs text-muted-foreground">Power</span>
                  <Badge variant="outline">measured: {d.measured_power_watts != null ? `${d.measured_power_watts.toFixed(2)} W` : "no meter connected"}</Badge>
                  {d.pmic && <Badge variant="outline">PMIC rails: {d.pmic.rails_watts.toFixed(2)} W</Badge>}
                </div>
                <p className="text-xs text-muted-foreground">{d.pmic ? d.pmic.source : "PMIC rail readings unavailable (needs vcgencmd on a Raspberry Pi 5)."} Watts are never estimated from CPU load.</p>
              </CardContent>
            </Card>
          </div>
        </>
      )}
    </>
  );
}
