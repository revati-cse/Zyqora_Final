import { useQueryClient } from "@tanstack/react-query";
import { CircleX, Download, Loader2, Play, RotateCcw, Square, Trash2 } from "lucide-react";
import { useRef, useState } from "react";
import { CategoryBadge, DecisionBadge, EmptyState, ErrorState, MetricCard, PageHeader, ScopeTag } from "@/components/common";
import { ConfusionMatrix } from "@/components/common/ConfusionMatrix";
import { Dropzone } from "@/components/inference/ImageInput";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label, Select } from "@/components/ui/input";
import { Alert, Progress, Table, Td, Th } from "@/components/ui/misc";
import { useModelStatus } from "@/hooks/useApi";
import { api, ApiError } from "@/lib/api";
import { useSettings } from "@/lib/settings";
import { downloadText, fmtMs, fmtPct, toCsv } from "@/lib/utils";
import type { AnalyticsSummary, Classification } from "@/types/api";

type Status = "queued" | "running" | "done" | "error";
interface Item { key: string; file: File; status: Status; truth: string; result?: Classification; error?: string }

const newBatchId = () => (crypto.randomUUID?.() ?? `${Date.now()}${Math.random()}`).replace(/[^a-zA-Z0-9]/g, "").slice(0, 16);

export default function Batch() {
  const { settings } = useSettings();
  const model = useModelStatus();
  const qc = useQueryClient();
  const [items, setItems] = useState<Item[]>([]);
  const [running, setRunning] = useState(false);
  const [batchId, setBatchId] = useState<string>(newBatchId);
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null);
  const [summaryError, setSummaryError] = useState<unknown>(null);
  const abort = useRef<AbortController | null>(null);
  const classes = model.data?.classes ?? [];
  const truthOptions = [...classes, ...(model.data?.unsupported_ground_truth ?? [])];

  const patch = (key: string, p: Partial<Item>) => setItems((xs) => xs.map((x) => (x.key === key ? { ...x, ...p } : x)));

  const run = async (only?: Status[]) => {
    const todo = items.filter((x) => (only ?? ["queued"]).includes(x.status));
    if (!todo.length) return;
    setRunning(true);
    setSummary(null);
    abort.current = new AbortController();
    for (const it of todo) {
      if (abort.current.signal.aborted) break;
      patch(it.key, { status: "running", error: undefined });
      try {
        const result = await api.classify(it.file, it.file.name, {
          source: "batch", batchId, groundTruth: it.truth || undefined, userId: settings.userId || undefined, signal: abort.current.signal,
        });
        patch(it.key, { status: "done", result });
      } catch (e) {
        if ((e as Error).name === "AbortError") { patch(it.key, { status: "queued" }); break; }
        patch(it.key, { status: "error", error: e instanceof ApiError ? `${e.code}: ${e.message}` : String(e) });
      }
    }
    setRunning(false);
    try {
      setSummary((await api.batch(batchId)).summary);
      setSummaryError(null);
    } catch (e) {
      setSummaryError(e);
    }
    for (const k of ["analytics", "history", "model-metrics"]) void qc.invalidateQueries({ queryKey: [k] });
  };

  const done = items.filter((x) => x.status === "done").length;
  const failed = items.filter((x) => x.status === "error").length;
  const progress = items.length ? ((done + failed) / items.length) * 100 : 0;
  const results = items.filter((x) => x.result).map((x) => x.result!);
  const acc = summary?.labelled_accuracy;

  const exportRows = () => items.map((x) => ({
    filename: x.file.name, status: x.status, error: x.error ?? "", predicted_category: x.result?.predicted_category ?? "",
    score: x.result?.confidence ?? "", ecoguard: x.result?.ecoguard.status ?? "", disposal_action: x.result?.disposal_action ?? "",
    inference_ms: x.result?.timing.inference_ms ?? "", ground_truth: x.result?.ground_truth ?? x.truth, correct: x.result?.correct ?? "",
    id: x.result?.id ?? "", model_profile: x.result?.model.profile ?? "",
  }));

  return (
    <>
      <PageHeader title="Batch Analysis"
        description="Classify many images with the same on-device model, one at a time. Add ground-truth labels to get accuracy; without labels only predictions and counts are shown."
        actions={items.length > 0 && (
          <>
            <Button size="sm" variant="outline" disabled={!results.length} onClick={() => downloadText(`zyqora_batch_${batchId}.csv`, toCsv(exportRows()), "text/csv")}><Download />CSV</Button>
            <Button size="sm" variant="outline" disabled={!results.length} onClick={() => downloadText(`zyqora_batch_${batchId}.json`, JSON.stringify({ batch_id: batchId, items: exportRows(), server_summary: summary }, null, 2), "application/json")}><Download />JSON</Button>
          </>
        )} />
      {model.data && !model.data.ready && <Alert variant="danger" className="mb-4" title="MODEL UNAVAILABLE">{model.data.code}: {model.data.message}</Alert>}

      <div className="grid grid-cols-1 gap-6 2xl:grid-cols-[minmax(0,320px)_1fr]">
        <Card>
          <CardHeader><CardTitle>Queue</CardTitle><CardDescription>Batch <span className="font-mono">{batchId}</span></CardDescription></CardHeader>
          <CardContent className="space-y-4">
            <Dropzone multiple disabled={running} onMany={(files) => setItems((xs) => [...xs, ...files.map((f, i) => ({ key: `${Date.now()}-${i}-${f.name}`, file: f, status: "queued" as Status, truth: "" }))])} />
            {items.length > 0 && (
              <>
                <div>
                  <Label htmlFor="truth-all">Ground truth for all queued images (optional)</Label>
                  <Select id="truth-all" disabled={running} defaultValue="" onChange={(e) => setItems((xs) => xs.map((x) => (x.status === "queued" ? { ...x, truth: e.target.value } : x)))}>
                    <option value="">No label</option>
                    {truthOptions.map((c) => <option key={c} value={c}>{c}{classes.includes(c) ? "" : " (not a model class)"}</option>)}
                  </Select>
                </div>
                <Progress value={progress} label="Batch progress" />
                <p className="font-mono text-xs text-muted-foreground">{done} done · {failed} failed · {items.length} total</p>
                <div className="flex flex-wrap gap-2">
                  {!running ? (
                    <Button disabled={!model.data?.ready || !items.some((x) => x.status === "queued")} onClick={() => run()}><Play />Classify queue</Button>
                  ) : (
                    <Button variant="outline" onClick={() => abort.current?.abort()}><Square />Stop</Button>
                  )}
                  <Button variant="outline" disabled={running || !failed} onClick={() => run(["error"])}><RotateCcw />Retry failed</Button>
                  <Button variant="ghost" disabled={running} onClick={() => { setItems([]); setSummary(null); setBatchId(newBatchId()); }}><Trash2 />Clear</Button>
                </div>
              </>
            )}
          </CardContent>
        </Card>

        <div className="min-w-0 space-y-6">
          {summary && (
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <MetricCard label="Classified" value={summary.total} />
              <MetricCard label="ACCEPT / REVIEW" value={`${summary.accepted} / ${summary.review}`} />
              <MetricCard label="Median inference" value={fmtMs(summary.inference_latency.median_ms, 2)} hint={`p95 ${fmtMs(summary.inference_latency.p95_ms, 2)}`} />
              <MetricCard label={acc ? "Accuracy (labelled)" : "Accuracy"} value={acc ? fmtPct(acc.accuracy, 1) : "—"}
                hint={acc ? `${acc.correct}/${acc.labelled_images} correct` : "needs ground-truth labels"} />
            </div>
          )}
          {summaryError != null && <ErrorState error={summaryError} title="Could not load batch summary" />}
          {acc && (
            <Card>
              <CardHeader><CardTitle>Labelled results <ScopeTag>this batch</ScopeTag></CardTitle>
                <CardDescription>Computed by the backend evaluator; REVIEW items stay in the denominator. Accepted-only accuracy {fmtPct(acc.accepted_only_accuracy, 1)}.</CardDescription></CardHeader>
              <CardContent>
                <ConfusionMatrix rows={Object.keys(acc.metrics.confusion_matrix.matrix).filter((r) => Object.values(acc.metrics.confusion_matrix.matrix[r]).some(Boolean))}
                  cols={Object.keys(Object.values(acc.metrics.confusion_matrix.matrix)[0] ?? {})} value={(r, c) => acc.metrics.confusion_matrix.matrix[r]?.[c] ?? 0} />
              </CardContent>
            </Card>
          )}
          {!items.length ? <EmptyState title="Queue is empty">Add JPEG, PNG or WebP images to start a batch.</EmptyState> : (
            <Card>
              <Table>
                <thead><tr><Th>File</Th><Th>Status</Th><Th className="hidden sm:table-cell">Truth</Th><Th>Category</Th><Th className="text-right">Score</Th><Th>Decision</Th><Th className="hidden text-right md:table-cell">Inference</Th></tr></thead>
                <tbody>
                  {items.map((x) => (
                    <tr key={x.key} data-testid="batch-row">
                      <Td className="max-w-40 truncate font-mono text-xs" title={x.file.name}>{x.file.name}</Td>
                      <Td className="text-xs">
                        {x.status === "running" ? <span className="inline-flex items-center gap-1"><Loader2 className="size-3.5 animate-spin" aria-hidden />running</span>
                          : x.status === "error" ? <span className="inline-flex items-center gap-1 text-danger" title={x.error}><CircleX className="size-3.5" aria-hidden />failed</span>
                          : x.status}
                        {x.error && <p className="mt-0.5 max-w-48 truncate text-[11px] text-danger" title={x.error}>{x.error}</p>}
                      </Td>
                      <Td className="hidden sm:table-cell">
                        <Select aria-label={`Ground truth for ${x.file.name}`} className="h-8 text-xs" value={x.truth} disabled={running || x.status === "done"}
                          onChange={(e) => patch(x.key, { truth: e.target.value })}>
                          <option value="">—</option>{truthOptions.map((c) => <option key={c} value={c}>{c}</option>)}
                        </Select>
                      </Td>
                      <Td>{x.result ? <CategoryBadge category={x.result.predicted_category} lang={settings.lang} /> : "—"}</Td>
                      <Td className="whitespace-nowrap text-right font-mono text-xs tabular">{x.result ? fmtPct(x.result.confidence, 2) : "—"}</Td>
                      <Td>{x.result ? <DecisionBadge status={x.result.ecoguard.status} lang={settings.lang} /> : "—"}</Td>
                      <Td className="hidden whitespace-nowrap text-right font-mono text-xs tabular md:table-cell">{x.result ? fmtMs(x.result.timing.inference_ms, 2) : "—"}</Td>
                    </tr>
                  ))}
                </tbody>
              </Table>
            </Card>
          )}
        </div>
      </div>
    </>
  );
}
