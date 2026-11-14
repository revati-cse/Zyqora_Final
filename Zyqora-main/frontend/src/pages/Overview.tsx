import { ArrowRight, Cpu, ScanSearch } from "lucide-react";
import { lazy, Suspense, useState } from "react";
import { Link } from "react-router";
import { CategoryDonut } from "@/components/charts/Charts";
import { CategoryBadge, DecisionBadge, EmptyState, ErrorState, MetricCard, PageHeader, ScopeTag } from "@/components/common";
import { BinIllustration } from "@/components/three/BinIllustration";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/misc";
import { useAnalytics, useAnalyticsAll, useDetections, useGreenScore, useHealth, useHistory, useModelStatus, useModelsStatus } from "@/hooks/useApi";
import { usePrefersReducedMotion, useSettings } from "@/lib/settings";
import { fmtMs, fmtNum, fmtPct, fmtTime } from "@/lib/utils";

const BinScene = lazy(() => import("@/components/three/BinScene"));

function hasWebGL() {
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") || c.getContext("webgl"));
  } catch {
    return false;
  }
}

export default function Overview() {
  const { settings } = useSettings();
  const analytics = useAnalytics();
  const health = useHealth();
  const model = useModelStatus();
  const recent = useHistory({ limit: 6 });
  const gs = useGreenScore();
  const reduced = usePrefersReducedMotion();
  const [webgl] = useState(hasWebGL);
  const a = analytics.data;
  const all = useAnalyticsAll();
  const models = useModelsStatus();
  const dets = useDetections({ limit: 5 });
  const d = all.data?.detections;
  const rec = d?.by_profile.recyclable?.counts ?? {};
  const food = d?.by_profile.food?.counts ?? {};
  // Do not spend the Pi's CPU/GPU on decoration when the dashboard is opened on the Pi itself.
  const onDevice = !!health.data?.is_raspberry_pi && ["localhost", "127.0.0.1", "::1"].includes(window.location.hostname);
  const show3d = settings.visual3d && webgl && !reduced && !onDevice;

  return (
    <>
      <div className="mb-8 grid items-center gap-6 rounded-xl border border-border bg-surface p-6 md:grid-cols-[1fr_260px]">
        <div>
          <p className="font-mono text-xs text-muted-foreground">NAMMA AI · PS3 · offline-first edge AI</p>
          <h1 className="mt-2 text-2xl font-semibold tracking-tight sm:text-3xl">Every item in a mixed-waste photo, sorted on the device.</h1>
          <p className="mt-2 max-w-xl text-sm text-muted-foreground">
            YOLO finds each object, a MobileNetV2 fine-tuned on the organiser's Organic / Recyclable data classifies every crop on a
            Raspberry Pi 5 (ExecuTorch + XNNPACK), and a transparent Green Score rates each reliable item. Uncertain items are shown as Unknown.
          </p>
          <div className="mt-4 flex flex-wrap gap-2">
            <Button asChild><Link to="/detect"><ScanSearch />Analyse mixed waste</Link></Button>
            <Button asChild variant="outline"><Link to="/model">Model comparison<ArrowRight /></Link></Button>
          </div>
        </div>
        <div className="hidden h-44 md:block">
          {show3d ? (
            <Suspense fallback={<BinIllustration />}><BinScene reducedMotion={reduced} /></Suspense>
          ) : <BinIllustration />}
        </div>
      </div>

      <PageHeader title="Overview" description={<>Mixed-waste analyses stored on this device. <ScopeTag>live</ScopeTag></>} />
      {all.isError && <div className="mb-4"><ErrorState error={all.error} /></div>}
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <MetricCard label="Analysed images" value={fmtNum(d?.images)} loading={all.isLoading} />
        <MetricCard label="Detected objects" value={fmtNum(d?.detected_objects)} loading={all.isLoading} hint={d ? `${d.waste_items} waste items · ${d.review_items} review` : undefined} />
        <MetricCard label="Recyclable / Non-Recyclable" value={d ? `${rec.Recyclable ?? 0} / ${rec["Non-Recyclable"] ?? 0}` : "—"} loading={all.isLoading}
          hint={d ? `${rec.Unknown ?? 0} Unknown (not reliable)` : undefined} />
        <MetricCard label="Food / Waste" value={d ? `${food.Food ?? 0} / ${food.Waste ?? 0}` : "—"} loading={all.isLoading} hint="profile mapping of model labels" />
        <MetricCard label="Average Green Score" value={d?.green_score.mean_image_score != null ? d.green_score.mean_image_score.toFixed(1) : "—"} loading={all.isLoading}
          hint={d ? `${d.green_score.scored_items} scored items` : undefined} />
        <MetricCard label="Median pipeline latency" value={fmtMs(d?.latency.pipeline.median_ms, 0)} loading={all.isLoading}
          hint={d?.latency.detector.median_ms != null ? `YOLO ${fmtMs(d.latency.detector.median_ms, 0)}` : "no analyses yet"} />
        <MetricCard label="Detector" value={models.data?.detector.ready ? "READY" : models.data ? "UNAVAILABLE" : "…"} mono={false} loading={models.isLoading}
          hint={models.data?.detector.profile ?? models.data?.detector.code ?? undefined} />
        <MetricCard label="Classifier" value={models.data?.classifier.ready ? "READY" : models.data ? "UNAVAILABLE" : "…"} mono={false} loading={models.isLoading}
          hint={models.data?.classifier.profile ?? undefined} />
      </div>
      <Card className="mt-6">
        <CardHeader className="flex-row items-center justify-between">
          <div><CardTitle>Recent detections</CardTitle><CardDescription>Latest mixed-waste analyses</CardDescription></div>
          <Button asChild variant="ghost" size="sm"><Link to="/history">History<ArrowRight /></Link></Button>
        </CardHeader>
        <CardContent>
          {dets.isLoading ? <Skeleton className="h-24" /> : dets.isError ? <ErrorState error={dets.error} /> : !dets.data?.items.length ? (
            <EmptyState title="No analyses yet"><Link className="underline" to="/detect">Analyse a mixed-waste image</Link>.</EmptyState>
          ) : (
            <ul className="divide-y divide-border">
              {dets.data.items.map((it) => (
                <li key={it.id} className="flex items-center gap-3 py-2.5 text-sm">
                  <span className="w-10 font-mono text-xs text-muted-foreground">#{it.id}</span>
                  <span className="min-w-0 flex-1 truncate">{it.objects.filter((o) => o.decision === "ACCEPT" || o.decision === "REVIEW").map((o) => o.organiser_category).join(" · ") || "no waste item"}</span>
                  <span className="whitespace-nowrap font-mono text-xs">GS {it.image_green_score?.toFixed(1) ?? "—"}</span>
                  <span className="hidden whitespace-nowrap font-mono text-xs text-muted-foreground sm:inline">{fmtMs(it.pipeline_ms, 0)}</span>
                  <span className="ml-2 hidden truncate text-xs text-muted-foreground md:inline">{fmtTime(it.created_at)}</span>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>

      <h2 className="mb-3 mt-10 text-lg font-semibold tracking-tight">Single-item classifications</h2>
      {analytics.isError && <div className="mb-4"><ErrorState error={analytics.error} /></div>}
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <MetricCard label="Images processed" value={fmtNum(a?.total)} loading={analytics.isLoading} />
        <MetricCard label="Accepted (bin recommended)" value={fmtNum(a?.accepted)} loading={analytics.isLoading} hint={a?.total ? fmtPct(a.accepted / a.total) : undefined} />
        <MetricCard label="Review required" value={fmtNum(a?.review)} loading={analytics.isLoading} hint={a?.review_rate != null ? `${fmtPct(a.review_rate)} review rate` : undefined} />
        <MetricCard label="Median model inference" value={fmtMs(a?.inference_latency.median_ms, 2)} loading={analytics.isLoading}
          hint={a?.inference_latency.p95_ms != null ? `p95 ${fmtMs(a.inference_latency.p95_ms, 2)} · ${a.inference_latency.count} runs` : "no runs yet"} />
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader className="flex-row items-center justify-between">
            <div><CardTitle>Recent classifications</CardTitle><CardDescription>Latest results from this device</CardDescription></div>
            <Button asChild variant="ghost" size="sm"><Link to="/history">History<ArrowRight /></Link></Button>
          </CardHeader>
          <CardContent>
            {recent.isLoading ? <div className="space-y-2">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-10" />)}</div>
              : recent.isError ? <ErrorState error={recent.error} />
              : !recent.data?.items.length ? (
                <EmptyState title="No classifications yet">Go to Single-item Classification and classify a waste image.</EmptyState>
              ) : (
                <ul className="divide-y divide-border">
                  {recent.data.items.map((it) => (
                    <li key={it.id} className="flex items-center gap-3 py-2.5 text-sm">
                      <span className="w-10 font-mono text-xs text-muted-foreground">#{it.id}</span>
                      <CategoryBadge category={it.predicted_category} lang={settings.lang} />
                      <span className="hidden font-mono text-xs tabular sm:inline">{fmtPct(it.confidence)}</span>
                      <DecisionBadge status={it.ecoguard_status} lang={settings.lang} />
                      <span className="ml-auto hidden truncate text-xs text-muted-foreground md:inline">{fmtTime(it.created_at)}</span>
                    </li>
                  ))}
                </ul>
              )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>Predicted categories</CardTitle><CardDescription>Distribution of all classifications</CardDescription></CardHeader>
          <CardContent>
            {analytics.isLoading ? <Skeleton className="h-52" /> : a && a.total ? <CategoryDonut data={a.by_category} /> : <EmptyState title="Nothing to chart yet" />}
          </CardContent>
        </Card>
      </div>

      <div className="mt-6 grid gap-6 md:grid-cols-3">
        <Card>
          <CardHeader><CardTitle>Model</CardTitle></CardHeader>
          <CardContent className="space-y-1.5 text-sm">
            {model.isLoading ? <Skeleton className="h-16" /> : model.data?.ready ? (
              <>
                <p className="font-mono text-xs">{model.data.profile}</p>
                <p className="text-muted-foreground">{model.data.classes?.join(" · ")} · {model.data.precision?.toUpperCase()} · output {JSON.stringify(model.data.output_shape)}</p>
                <p className="text-muted-foreground">{model.data.runtime} {model.data.executorch_version} · {model.data.xnnpack ? "XNNPACK" : "no XNNPACK"}</p>
              </>
            ) : <p className="text-danger">MODEL UNAVAILABLE{model.data?.code ? ` · ${model.data.code}` : ""}</p>}
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>Device</CardTitle></CardHeader>
          <CardContent className="space-y-1.5 text-sm">
            {health.isError ? <p className="text-danger">DEVICE DISCONNECTED</p> : health.data ? (
              <>
                <p className="flex items-center gap-1.5"><Cpu className="size-4" aria-hidden />{health.data.device_model ?? health.data.machine}</p>
                <p className="text-muted-foreground">{health.data.is_raspberry_pi ? "Raspberry Pi detected" : "Not a Raspberry Pi (development host)"}</p>
                <Link className="text-xs underline underline-offset-2" to="/device">Live telemetry</Link>
              </>
            ) : <Skeleton className="h-16" />}
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>Green Points (rewards)</CardTitle></CardHeader>
          <CardContent className="space-y-1.5 text-sm">
            {gs.isError ? <ErrorState error={gs.error} /> : gs.data ? (
              gs.data.users.length ? (
                <>
                  <p className="text-muted-foreground">{gs.data.users.length} participant(s); verified points only.</p>
                  <p className="font-mono">{gs.data.users[0].user_id}: {gs.data.users[0].total_points} pts</p>
                  <Link className="text-xs underline underline-offset-2" to="/rewards">Open Green Points</Link>
                </>
              ) : <p className="text-muted-foreground">No participants yet.</p>
            ) : <Skeleton className="h-16" />}
          </CardContent>
        </Card>
      </div>
      {a?.latest && (
        <p className="mt-6 text-xs text-muted-foreground">
          Latest: #{a.latest.id} {a.latest.predicted_category} ({a.latest.ecoguard_status}) at {fmtTime(a.latest.created_at)}.
        </p>
      )}
    </>
  );
}
