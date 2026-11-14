import { AlertTriangle, ArrowRight } from "lucide-react";
import { ErrorState, MetricCard, PageHeader, ScopeTag } from "@/components/common";
import { ConfusionMatrix } from "@/components/common/ConfusionMatrix";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, Skeleton, Table, Td, Th } from "@/components/ui/misc";
import { useModelMetrics, useModelStatus } from "@/hooks/useApi";
import { cn, fmtBytes, fmtMs, fmtNum, fmtPct } from "@/lib/utils";
import type { ClassificationMetrics } from "@/types/api";

const STEPS = [
  ["Image", "upload, browser camera or Pi camera"],
  ["Preprocess", "EXIF fix → RGB → resize 224×224 → ImageNet normalise"],
  ["ExecuTorch", "MobileNetV2 .pte, XNNPACK CPU kernels"],
  ["Decode", "2 logits → softmax (0 Organic, 1 Recyclable)"],
  ["EcoGuard", "score ≥ threshold → ACCEPT, else REVIEW"],
  ["Action", "bin guidance + voice; REVIEW = supervisor"],
];

function MetricsTable({ m }: { m: ClassificationMetrics }) {
  return (
    <Table>
      <thead><tr><Th>Class</Th><Th className="text-right">Support</Th><Th className="text-right">Precision</Th><Th className="text-right">Recall</Th><Th className="text-right">F1</Th></tr></thead>
      <tbody>
        {Object.entries(m.per_class).filter(([, v]) => v.support || v.predicted).map(([c, v]) => (
          <tr key={c}>
            <Td>{c}{!v.predictable && <span className="ml-1 text-xs text-muted-foreground">(not a model class)</span>}</Td>
            <Td className="text-right font-mono tabular">{v.support}</Td>
            <Td className="text-right font-mono tabular">{v.precision == null ? "—" : v.precision.toFixed(4)}</Td>
            <Td className="text-right font-mono tabular">{v.recall == null ? "—" : v.recall.toFixed(4)}</Td>
            <Td className="text-right font-mono tabular">{v.f1 == null ? "—" : v.f1.toFixed(4)}</Td>
          </tr>
        ))}
      </tbody>
    </Table>
  );
}

export default function ModelPerformance() {
  const status = useModelStatus();
  const metrics = useModelMetrics();
  const s = status.data;
  const v = metrics.data?.validation;
  const test = metrics.data?.test;
  const bench = metrics.data?.benchmark;
  const live = metrics.data?.live;
  const threshold = s?.ecoguard?.threshold;

  return (
    <>
      <PageHeader title="Model Performance"
        description="What runs, how it was evaluated, and what has not been measured yet. Each figure is tagged with its source; validation, TEST and live figures are never mixed." />

      <Card className="mb-6">
        <CardHeader><CardTitle>Edge AI pipeline</CardTitle><CardDescription>Every live classification follows these steps on the device.</CardDescription></CardHeader>
        <CardContent>
          <ol className="grid gap-2 sm:grid-cols-2 lg:grid-cols-6">
            {STEPS.map(([title, text], i) => (
              <li key={title} className="relative rounded-md border border-border bg-surface-2 p-3">
                <p className="font-mono text-[10px] text-muted-foreground">{String(i + 1).padStart(2, "0")}</p>
                <p className="text-sm font-medium">{title}</p>
                <p className="mt-0.5 text-xs text-muted-foreground">{text}</p>
                {i < STEPS.length - 1 && <ArrowRight className="absolute -right-2.5 top-1/2 hidden size-4 -translate-y-1/2 text-muted-foreground lg:block" aria-hidden />}
              </li>
            ))}
          </ol>
        </CardContent>
      </Card>

      {status.isError && <div className="mb-6"><ErrorState error={status.error} title="DEVICE DISCONNECTED" /></div>}
      <Card className="mb-6">
        <CardHeader><CardTitle>Loaded model <ScopeTag>device</ScopeTag></CardTitle></CardHeader>
        <CardContent>
          {status.isLoading ? <Skeleton className="h-28" /> : !s?.ready ? (
            <Alert variant="danger" title="MODEL UNAVAILABLE"><span className="font-mono text-xs">{s?.code}</span> · {s?.message}</Alert>
          ) : (
            <dl className="grid gap-x-6 gap-y-3 text-sm sm:grid-cols-2 lg:grid-cols-3">
              {[
                ["Architecture", s.description],
                ["Profile", s.profile],
                ["Trained classes", s.classes?.join(", ")],
                ["Input", `${JSON.stringify(s.input_shape)} float32 NCHW`],
                ["Resize / crop", `${JSON.stringify(s.preprocessing?.resize)} / ${s.preprocessing?.center_crop ?? "none"}`],
                ["Output", JSON.stringify(s.output_shape)],
                ["Artifact", `${s.pte} · ${fmtBytes(s.pte_size_bytes)}`],
                ["SHA-256", s.sha256 ? `${s.sha256.slice(0, 16)}…${s.sha256_pinned ? (s.sha256_pinned === s.sha256 ? " (pinned ✓)" : " (pin mismatch)") : " (not pinned)"}` : "—"],
                ["Runtime", `${s.runtime} ${s.executorch_version}`],
                ["Backend", s.xnnpack ? "XNNPACK (registered)" : "XNNPACK not registered"],
                ["Precision / threads", `${s.precision?.toUpperCase()} · ${s.threads ?? "default"} threads`],
                ["EcoGuard threshold", `${threshold} · ${s.ecoguard?.calibrated ? "calibrated" : "provisional (validation table)"}`],
              ].map(([k, val]) => (
                <div key={k} className="min-w-0">
                  <dt className="text-xs text-muted-foreground">{k}</dt>
                  <dd className="truncate font-mono text-xs" title={String(val)}>{val}</dd>
                </div>
              ))}
            </dl>
          )}
          {s?.fallback_used && <Alert variant="danger" className="mt-4" title="Fallback model in use">{s.failed_profiles?.join("; ")}</Alert>}
        </CardContent>
      </Card>

      {metrics.isError && <ErrorState error={metrics.error} />}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Development validation <ScopeTag>validation · team-reported</ScopeTag></CardTitle>
            <CardDescription>{v?.source}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            {!v ? <Skeleton className="h-40" /> : (
              <>
                <div className="grid grid-cols-3 gap-3">
                  <MetricCard label="Accuracy" value={fmtPct(v.accuracy, 2)} hint={`${fmtNum(v.dataset.validation_images)} images`} />
                  {Object.entries(v.per_class_accuracy).map(([c, x]) => <MetricCard key={c} label={`${c} accuracy`} value={fmtPct(x, 1)} />)}
                </div>
                {v.pi_reproduction && (
                  <div className="rounded-md border border-border bg-surface-2 p-3 text-sm" data-testid="pi-reproduction">
                    <p className="flex flex-wrap items-center gap-2 font-medium">Reproduced on Raspberry Pi 5 <ScopeTag>pi · same validation set</ScopeTag></p>
                    <p className="mt-1 font-mono text-xs tabular">
                      {v.pi_reproduction.correct.toLocaleString()}/{v.pi_reproduction.total.toLocaleString()} correct ({fmtPct(v.pi_reproduction.accuracy, 1)})
                      {" · "}review {fmtPct(v.pi_reproduction.review_rate, 1)} · accepted-only {fmtPct(v.pi_reproduction.accepted_only_accuracy, 2)}
                      {" · "}inference median {fmtMs(v.pi_reproduction.inference_median_ms, 2)} · {fmtNum(v.pi_reproduction.throughput_images_per_second, 1)} img/s
                    </p>
                    <p className="mt-1 text-xs text-muted-foreground">{v.pi_reproduction.source} {v.pi_reproduction.note}</p>
                  </div>
                )}
                <ConfusionMatrix rows={v.confusion_matrix.labels} cols={v.confusion_matrix.labels}
                  value={(r, c) => v.confusion_matrix.matrix[v.confusion_matrix.labels.indexOf(r)][v.confusion_matrix.labels.indexOf(c)]}
                  caption="Validation confusion matrix (2,000 images)" />
                <Table>
                  <thead><tr><Th>Class</Th><Th className="text-right">Precision</Th><Th className="text-right">Recall</Th><Th className="text-right">F1</Th></tr></thead>
                  <tbody>{Object.entries(v.derived_from_confusion_matrix).map(([c, m]) => (
                    <tr key={c}><Td>{c}</Td><Td className="text-right font-mono">{m.precision.toFixed(4)}</Td><Td className="text-right font-mono">{m.recall.toFixed(4)}</Td><Td className="text-right font-mono">{m.f1.toFixed(4)}</Td></tr>
                  ))}</tbody>
                </Table>
                <div>
                  <p className="mb-2 text-xs font-medium">EcoGuard threshold trade-off (validation)</p>
                  <Table>
                    <thead><tr><Th>Threshold</Th><Th className="text-right">Review rate</Th><Th className="text-right">Accepted accuracy</Th></tr></thead>
                    <tbody>{v.ecoguard_threshold_table.map((r) => (
                      <tr key={r.threshold} className={cn(r.threshold === threshold && "bg-surface-2 font-medium")}>
                        <Td className="font-mono">{r.threshold.toFixed(2)}{r.threshold === threshold && " ← in use"}</Td>
                        <Td className="text-right font-mono">{fmtPct(r.review_rate, 2)}</Td>
                        <Td className="text-right font-mono">{fmtPct(r.accepted_accuracy, 2)}</Td>
                      </tr>
                    ))}</tbody>
                  </Table>
                </div>
              </>
            )}
          </CardContent>
        </Card>

        <div className="min-w-0 space-y-6">
          <Card>
            <CardHeader><CardTitle>Independent TEST set <ScopeTag>test</ScopeTag></CardTitle></CardHeader>
            <CardContent>
              {!metrics.data ? <Skeleton className="h-24" /> : !test ? (
                <Alert title="Not measured yet">
                  No TEST evaluation has been run on this device. Run on the Pi (labels from the O/ and R/ folders):
                  <pre className="mt-2 overflow-x-auto whitespace-pre-wrap break-all rounded bg-surface p-2 font-mono text-[11px]">python run_zyqora.py --profile waste_mobilenetv2 --images &lt;TEST dir&gt; --labels-from-folders --dataset-role test --quiet --out outputs/test_waste</pre>
                </Alert>
              ) : (
                <div className="space-y-4">
                  <p className="text-xs text-muted-foreground">{test.source} · {test.model_profile} · measured on {test.measured_on}</p>
                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                    <MetricCard label="Accuracy" value={fmtPct(test.overall_accuracy, 2)} hint={`${fmtNum(test.total_images)} images`} />
                    <MetricCard label="Review rate" value={fmtPct(test.review_rate, 2)} />
                    <MetricCard label="Accepted-only acc." value={fmtPct(test.accepted_only_accuracy, 2)} />
                    <MetricCard label="Wrong but accepted" value={fmtNum(test.accepted_incorrect)} />
                  </div>
                  {test.metrics && (
                    <>
                      <ConfusionMatrix rows={Object.keys(test.metrics.confusion_matrix.matrix)}
                        cols={Object.keys(Object.values(test.metrics.confusion_matrix.matrix)[0] ?? {})}
                        value={(r, c) => test.metrics!.confusion_matrix.matrix[r][c] ?? 0} caption="TEST confusion matrix" />
                      <MetricsTable m={test.metrics} />
                    </>
                  )}
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Benchmark <ScopeTag>{bench ? (bench.platform.is_raspberry_pi_5 ? "raspberry pi 5" : "host, not pi") : "pending"}</ScopeTag></CardTitle>
              {bench && <CardDescription>{bench.source} · {bench.platform.device_model ?? bench.platform.machine}</CardDescription>}
            </CardHeader>
            <CardContent>
              {!metrics.data ? <Skeleton className="h-20" /> : !bench ? (
                <Alert title="Not measured yet">Run <span className="break-all font-mono text-xs">python scripts/benchmark_pi.py --profile waste_mobilenetv2 --threads 1 2 3 4 --images &lt;dir&gt;</span> on the Pi.</Alert>
              ) : (
                <>
                  {bench.warning && <Alert variant="review" className="mb-3">{bench.warning}</Alert>}
                  <Table>
                    <thead><tr><Th>Threads</Th><Th className="text-right">Median</Th><Th className="text-right">p95</Th><Th className="text-right">CPU/inf</Th><Th className="text-right">Inf/s</Th><Th className="text-right">Peak RSS</Th></tr></thead>
                    <tbody>{bench.results.map((r) => (
                      <tr key={String(r.threads)}>
                        <Td className="font-mono">{r.threads ?? "default"}</Td>
                        <Td className="text-right font-mono">{fmtMs(r.median_ms, 2)}</Td>
                        <Td className="text-right font-mono">{fmtMs(r.p95_ms, 2)}</Td>
                        <Td className="text-right font-mono">{fmtMs(r.cpu_ms_per_inference, 2)}</Td>
                        <Td className="text-right font-mono">{fmtNum(r.throughput, 1)}</Td>
                        <Td className="text-right font-mono">{r.peak_rss_mb ? `${r.peak_rss_mb.toFixed(0)} MB` : "—"}</Td>
                      </tr>
                    ))}</tbody>
                  </Table>
                </>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Live on this device <ScopeTag>live</ScopeTag></CardTitle>
              <CardDescription>{live ? `${live.source} · ${live.measured_on}` : ""}</CardDescription>
            </CardHeader>
            <CardContent>
              {!live ? <Skeleton className="h-20" /> : live.total === 0 ? <p className="text-sm text-muted-foreground">No classifications yet.</p> : (
                <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
                  <MetricCard label="Classifications" value={fmtNum(live.total)} />
                  <MetricCard label="Median inference" value={fmtMs(live.inference_latency.median_ms, 2)} />
                  <MetricCard label="p95 inference" value={fmtMs(live.inference_latency.p95_ms, 2)} />
                  <MetricCard label="Review rate" value={fmtPct(live.review_rate)} />
                </div>
              )}
              {live?.labelled_accuracy && (
                <p className="mt-3 text-xs text-muted-foreground">
                  Labelled uploads: {live.labelled_accuracy.correct}/{live.labelled_accuracy.labelled_images} correct ({fmtPct(live.labelled_accuracy.accuracy)}). {live.labelled_accuracy.note}.
                </p>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      <Card className="mt-6">
        <CardHeader><CardTitle className="flex items-center gap-2"><AlertTriangle className="size-4 text-review" aria-hidden />Known limitations</CardTitle></CardHeader>
        <CardContent>
          <ul className="list-disc space-y-1.5 pl-5 text-sm text-muted-foreground">
            {(v?.limitations ?? []).map((l) => <li key={l}>{l}</li>)}
            <li>REVIEW is a decision (ask a supervisor), not a third class. Residual items are forced into Organic or Recyclable, sometimes with high confidence.</li>
          </ul>
        </CardContent>
      </Card>
    </>
  );
}
