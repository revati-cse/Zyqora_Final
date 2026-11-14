import { Download } from "lucide-react";
import { useState } from "react";
import { CategoryDonut, CountBars, DecisionChart, LatencyChart, Lines, VolumeChart } from "@/components/charts/Charts";
import { EmptyState, ErrorState, MetricCard, PageHeader, ScopeTag } from "@/components/common";
import { ConfusionMatrix } from "@/components/common/ConfusionMatrix";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input, Label, Select } from "@/components/ui/input";
import { Skeleton, Table, Td, Th } from "@/components/ui/misc";
import { useAnalytics, useAnalyticsAll, useDevice, useModelStatus } from "@/hooks/useApi";
import { api } from "@/lib/api";
import { fmtMs, fmtNum, fmtPct, toIsoEnd, toIsoStart } from "@/lib/utils";

const short = (iso: string) => iso.slice(5, 16).replace("T", " ");

function DetectionAnalytics() {
  const [days, setDays] = useState<number | undefined>(undefined);
  const q = useAnalyticsAll(days);
  const d = q.data?.detections;
  return (
    <section className="mb-10" aria-labelledby="det-analytics">
      <div className="mb-3 flex flex-wrap items-end justify-between gap-3">
        <h2 id="det-analytics" className="text-lg font-semibold tracking-tight">Mixed-waste analyses <ScopeTag>stored on device</ScopeTag></h2>
        <div className="flex items-center gap-2">
          <Label htmlFor="a-days" className="mb-0">Period</Label>
          <Select id="a-days" value={days ?? ""} onChange={(e) => setDays(e.target.value ? Number(e.target.value) : undefined)}>
            <option value="">All time</option><option value="1">Last 24 h</option><option value="7">Last 7 days</option><option value="30">Last 30 days</option>
          </Select>
          <Button asChild size="sm" variant="outline"><a href={api.detectionsCsvUrl()} download><Download />CSV</a></Button>
        </div>
      </div>
      {q.isError ? <ErrorState error={q.error} /> : q.isLoading ? <Skeleton className="h-72" /> : !d?.images ? (
        <EmptyState title="No mixed-waste analyses yet">Charts appear after the first analysis on the AI Detection page.</EmptyState>
      ) : (
        <div className="space-y-6">
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <MetricCard label="Images" value={fmtNum(d.images)} />
            <MetricCard label="Objects / waste items" value={`${d.detected_objects} / ${d.waste_items}`} />
            <MetricCard label="Reliable items" value={fmtNum(d.accepted_items)} hint={`${d.review_items} review (Unknown)`} />
            <MetricCard label="Median pipeline" value={fmtMs(d.latency.pipeline.median_ms, 0)} hint={`p95 ${fmtMs(d.latency.pipeline.p95_ms, 0)}`} />
          </div>
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            {Object.entries(d.by_profile).map(([name, p]) => (
              <Card key={name} className="min-w-0"><CardHeader><CardTitle className="text-base">{p.title}</CardTitle>
                <CardDescription>Derived from stored model labels; review items count as Unknown.</CardDescription></CardHeader>
                <CardContent>{Object.keys(p.counts).length ? <CountBars data={p.counts} /> : <EmptyState title="No waste items" />}</CardContent></Card>
            ))}
          </div>
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <Card className="min-w-0"><CardHeader><CardTitle className="text-base">Material (detector hint)</CardTitle></CardHeader>
              <CardContent><CountBars data={d.by_material} /></CardContent></Card>
            <Card className="min-w-0"><CardHeader><CardTitle className="text-base">Disposal streams</CardTitle></CardHeader>
              <CardContent><CountBars data={d.by_stream} /></CardContent></Card>
            <Card className="min-w-0"><CardHeader><CardTitle className="text-base">Daily detections</CardTitle></CardHeader>
              <CardContent><Lines data={d.by_day} x="date" lines={[{ key: "images", name: "images" }, { key: "objects", name: "objects" }, { key: "waste_items", name: "waste items" }]} /></CardContent></Card>
            <Card className="min-w-0"><CardHeader><CardTitle className="text-base">Green Score trend</CardTitle><CardDescription>Daily mean image score</CardDescription></CardHeader>
              <CardContent><Lines data={d.by_day} x="date" lines={[{ key: "mean_green_score", name: "mean image Green Score" }]} domain={[0, 100]} /></CardContent></Card>
            <Card className="min-w-0"><CardHeader><CardTitle className="text-base">Inference latency</CardTitle><CardDescription>Per analysis, measured on the device</CardDescription></CardHeader>
              <CardContent><Lines data={d.latency.series.map((x) => ({ ...x, label: `#${x.id}` }))} x="label" unit=" ms"
                lines={[{ key: "pipeline_ms", name: "pipeline" }, { key: "detector_ms", name: "YOLO" }, { key: "classification_ms", name: "MobileNetV2 (all crops)" }]} /></CardContent></Card>
            <Card className="min-w-0"><CardHeader><CardTitle className="text-base">CPU and memory utilisation</CardTitle>
              <CardDescription>System samples stored at each analysis and while System Monitoring is open</CardDescription></CardHeader>
              <CardContent>{d.system_metrics.length ? <Lines data={d.system_metrics.map((x) => ({ ...x, t: short(x.timestamp) }))} x="t" unit="%" domain={[0, 100]}
                lines={[{ key: "cpu_percent", name: "CPU (all cores)" }, { key: "memory_percent", name: "RAM used" }]} /> : <EmptyState title="No samples yet" />}</CardContent></Card>
          </div>
        </div>
      )}
    </section>
  );
}

export default function Analytics() {
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [category, setCategory] = useState("");
  const filters = { date_from: toIsoStart(from), date_to: toIsoEnd(to), category: category || undefined };
  const q = useAnalytics(filters);
  const model = useModelStatus();
  const device = useDevice();
  const a = q.data;
  const cats = Object.keys(a?.by_category ?? {});

  return (
    <>
      <PageHeader title="Analytics" description={<>Only records stored by this device are charted; nothing is simulated. <ScopeTag>live</ScopeTag></>}
        actions={<Button asChild size="sm" variant="outline"><a href={api.historyCsvUrl(filters)} download><Download />Export classifications CSV</a></Button>} />
      <DetectionAnalytics />
      <h2 className="mb-3 text-lg font-semibold tracking-tight">Single-item classifications</h2>
      <Card className="mb-6">
        <CardContent className="grid gap-3 p-4 sm:grid-cols-3">
          <div><Label htmlFor="a-from">From</Label><Input id="a-from" type="date" value={from} onChange={(e) => setFrom(e.target.value)} /></div>
          <div><Label htmlFor="a-to">To</Label><Input id="a-to" type="date" value={to} onChange={(e) => setTo(e.target.value)} /></div>
          <div><Label htmlFor="a-cat">Predicted category</Label>
            <Select id="a-cat" value={category} onChange={(e) => setCategory(e.target.value)}>
              <option value="">All</option>{(model.data?.classes ?? []).map((c) => <option key={c} value={c}>{c}</option>)}
            </Select></div>
        </CardContent>
      </Card>

      {q.isError ? <ErrorState error={q.error} /> : q.isLoading ? <Skeleton className="h-96" /> : !a?.total ? (
        <EmptyState title="No data for these filters">Classify images (Live or Batch) to populate analytics. Nothing here is simulated.</EmptyState>
      ) : (
        <div className="space-y-6">
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <MetricCard label="Classifications" value={fmtNum(a.total)} />
            <MetricCard label="Review rate" value={fmtPct(a.review_rate)} hint={`${a.review} of ${a.total}`} />
            <MetricCard label="Median / p95 inference" value={fmtMs(a.inference_latency.median_ms, 2)} hint={`p95 ${fmtMs(a.inference_latency.p95_ms, 2)}`} />
            <MetricCard label="Median total on device" value={fmtMs(a.total_latency.median_ms, 1)} hint="decode + preprocess + inference + decision" />
          </div>
          <div className="grid gap-6 lg:grid-cols-3">
            <Card><CardHeader><CardTitle>Organic vs Recyclable</CardTitle></CardHeader><CardContent><CategoryDonut data={a.by_category} /></CardContent></Card>
            <Card className="lg:col-span-2"><CardHeader><CardTitle>Volume per day</CardTitle><CardDescription>Stacked by predicted category</CardDescription></CardHeader>
              <CardContent><VolumeChart data={a.by_day} categories={cats} /></CardContent></Card>
          </div>
          <div className="grid gap-6 lg:grid-cols-2">
            <Card><CardHeader><CardTitle>ACCEPT vs REVIEW</CardTitle></CardHeader><CardContent><DecisionChart data={a.by_day} /></CardContent></Card>
            <Card><CardHeader><CardTitle>Latency trend</CardTitle><CardDescription>Last {a.latency_series.length} classifications</CardDescription></CardHeader>
              <CardContent><LatencyChart data={a.latency_series} /></CardContent></Card>
          </div>
          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader><CardTitle>Batch throughput</CardTitle><CardDescription>Measured from batch timestamps (includes upload time from the browser)</CardDescription></CardHeader>
              <CardContent>
                {!a.batch_throughput.length ? <p className="text-sm text-muted-foreground">No batch runs with 2+ images yet.</p> : (
                  <Table>
                    <thead><tr><Th>Batch</Th><Th className="text-right">Images</Th><Th className="text-right">Seconds</Th><Th className="text-right">Images/s</Th></tr></thead>
                    <tbody>{a.batch_throughput.map((b) => (
                      <tr key={b.batch_id}><Td className="font-mono text-xs">{b.batch_id}</Td><Td className="text-right font-mono">{b.images}</Td>
                        <Td className="text-right font-mono">{b.seconds.toFixed(1)}</Td><Td className="text-right font-mono">{b.images_per_second.toFixed(2)}</Td></tr>
                    ))}</tbody>
                  </Table>
                )}
              </CardContent>
            </Card>
            <Card>
              <CardHeader><CardTitle>Device now</CardTitle><CardDescription>Current telemetry (history is shown on Device Monitoring)</CardDescription></CardHeader>
              <CardContent className="grid grid-cols-3 gap-3">
                <MetricCard label="CPU" value={device.data?.cpu_percent == null ? "n/a" : `${device.data.cpu_percent.toFixed(0)}%`} />
                <MetricCard label="RAM" value={device.data?.memory.used_percent == null ? "n/a" : `${device.data.memory.used_percent.toFixed(0)}%`} />
                <MetricCard label="Temp" value={device.data?.temperature_c == null ? "n/a" : `${device.data.temperature_c.toFixed(0)}°C`} />
              </CardContent>
            </Card>
          </div>
          <Card>
            <CardHeader><CardTitle>Category accuracy <ScopeTag>labelled uploads only</ScopeTag></CardTitle></CardHeader>
            <CardContent>
              {!a.labelled_accuracy ? (
                <p className="text-sm text-muted-foreground">No uploads with ground-truth labels in this range, so no accuracy is shown. Add labels in Batch Analysis.</p>
              ) : (
                <div className="grid gap-6 md:grid-cols-2">
                  <div className="space-y-2 text-sm">
                    <p><span className="font-mono">{a.labelled_accuracy.correct}/{a.labelled_accuracy.labelled_images}</span> correct · accuracy <span className="font-mono">{fmtPct(a.labelled_accuracy.accuracy)}</span></p>
                    <p>Accepted-only accuracy <span className="font-mono">{fmtPct(a.labelled_accuracy.accepted_only_accuracy)}</span></p>
                    <p className="text-xs text-muted-foreground">{a.labelled_accuracy.note}. Not a substitute for the independent TEST evaluation.</p>
                  </div>
                  <ConfusionMatrix rows={Object.keys(a.labelled_accuracy.metrics.confusion_matrix.matrix).filter((r) => Object.values(a.labelled_accuracy!.metrics.confusion_matrix.matrix[r]).some(Boolean))}
                    cols={Object.keys(Object.values(a.labelled_accuracy.metrics.confusion_matrix.matrix)[0] ?? {})}
                    value={(r, c) => a.labelled_accuracy!.metrics.confusion_matrix.matrix[r]?.[c] ?? 0} />
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </>
  );
}
