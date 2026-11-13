import { Download } from "lucide-react";
import { useState } from "react";
import { CategoryDonut, DecisionChart, LatencyChart, VolumeChart } from "@/components/charts/Charts";
import { EmptyState, ErrorState, MetricCard, PageHeader, ScopeTag } from "@/components/common";
import { ConfusionMatrix } from "@/components/common/ConfusionMatrix";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input, Label, Select } from "@/components/ui/input";
import { Skeleton, Table, Td, Th } from "@/components/ui/misc";
import { useAnalytics, useDevice, useModelStatus } from "@/hooks/useApi";
import { api } from "@/lib/api";
import { fmtMs, fmtNum, fmtPct, toIsoEnd, toIsoStart } from "@/lib/utils";

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
      <PageHeader title="Analytics" description={<>Aggregated from the persisted classification history on this device. <ScopeTag>live</ScopeTag></>}
        actions={<Button asChild size="sm" variant="outline"><a href={api.historyCsvUrl(filters)} download><Download />Export CSV</a></Button>} />
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
