import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, Download, ImageOff, Trash2 } from "lucide-react";
import { Fragment, useState } from "react";
import { CategoryBadge, DecisionBadge, EmptyState, ErrorState, PageHeader } from "@/components/common";
import { DetectionReport } from "@/components/detection/Report";
import { ScoreBars } from "@/components/inference/ResultCard";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input, Label, Select } from "@/components/ui/input";
import { Skeleton, Table, Tabs, TabsContent, TabsList, TabsTrigger, Td, Th } from "@/components/ui/misc";
import { useDetection, useDetections, useHistory, useModelStatus } from "@/hooks/useApi";
import { api, apiUrl } from "@/lib/api";
import { useSettings } from "@/lib/settings";
import { fmtMs, fmtPct, fmtTime, toIsoEnd, toIsoStart } from "@/lib/utils";

const PAGE = 20;

const MATERIALS = ["Plastic", "Paper", "Cardboard", "Metal", "Glass", "Organic", "E-Waste", "Other", "Unknown"];

function Pager({ page, pages, total, noun, setPage }: { page: number; pages: number; total: number; noun: string; setPage: (f: (p: number) => number) => void }) {
  return (
    <div className="flex items-center justify-between border-t border-border px-4 py-3 text-xs text-muted-foreground">
      <span>{total} {noun}</span>
      <div className="flex items-center gap-2">
        <Button size="sm" variant="outline" disabled={page === 0} onClick={() => setPage((p) => p - 1)} aria-label="Previous page"><ChevronLeft /></Button>
        <span className="font-mono">{page + 1} / {pages}</span>
        <Button size="sm" variant="outline" disabled={page + 1 >= pages} onClick={() => setPage((p) => p + 1)} aria-label="Next page"><ChevronRight /></Button>
      </div>
    </div>
  );
}

function DetectionDetail({ id }: { id: number }) {
  const q = useDetection(id);
  if (q.isLoading) return <Skeleton className="h-64" />;
  if (q.isError) return <ErrorState error={q.error} />;
  const r = q.data!;
  return r.image_url ? <DetectionReport r={r} imageSrc={apiUrl(r.image_url)} /> : <p className="text-sm text-muted-foreground">No stored image for this analysis.</p>;
}

function DetectionHistory() {
  const qc = useQueryClient();
  const [search, setSearch] = useState("");
  const [material, setMaterial] = useState("");
  const [review, setReview] = useState(false);
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [page, setPage] = useState(0);
  const [open, setOpen] = useState<number | null>(null);
  const filters = { q: search.trim() || undefined, material: material || undefined, has_review: review, date_from: toIsoStart(from), date_to: toIsoEnd(to) };
  const q = useDetections({ ...filters, limit: PAGE, offset: page * PAGE });
  const del = useMutation({
    mutationFn: api.deleteDetection,
    onSuccess: () => { setOpen(null); for (const k of ["detections", "analytics-all"]) void qc.invalidateQueries({ queryKey: [k] }); },
  });
  const total = q.data?.total ?? 0;
  const pages = Math.max(1, Math.ceil(total / PAGE));
  return (
    <>
      <Card className="mb-4">
        <CardContent className="grid gap-3 p-4 sm:grid-cols-2 lg:grid-cols-6">
          <div className="lg:col-span-2"><Label htmlFor="d-q">Search</Label>
            <Input id="d-q" placeholder="file name, object, label or material" value={search} onChange={(e) => { setSearch(e.target.value); setPage(0); }} /></div>
          <div><Label htmlFor="d-mat">Material</Label>
            <Select id="d-mat" value={material} onChange={(e) => { setMaterial(e.target.value); setPage(0); }}>
              <option value="">All</option>{MATERIALS.map((m) => <option key={m}>{m}</option>)}
            </Select></div>
          <div><Label htmlFor="d-from">From</Label><Input id="d-from" type="date" value={from} onChange={(e) => { setFrom(e.target.value); setPage(0); }} /></div>
          <div><Label htmlFor="d-to">To</Label><Input id="d-to" type="date" value={to} onChange={(e) => { setTo(e.target.value); setPage(0); }} /></div>
          <label className="flex items-end gap-2 pb-2 text-sm"><input type="checkbox" className="size-4 accent-[var(--primary)]" checked={review}
            onChange={(e) => { setReview(e.target.checked); setPage(0); }} />Only with review items</label>
        </CardContent>
      </Card>
      <div className="mb-3 flex justify-end">
        <Button asChild variant="outline" size="sm"><a href={api.detectionsCsvUrl(filters)} download><Download />Export CSV (one row per object)</a></Button>
      </div>
      {del.isError && <div className="mb-3"><ErrorState error={del.error} title="Delete failed" /></div>}
      {q.isError ? <ErrorState error={q.error} /> : q.isLoading ? <Skeleton className="h-64" /> : !q.data?.items.length ? (
        <EmptyState title="No analyses match">Analyse an image on the AI Detection page, or clear the filters.</EmptyState>
      ) : (
        <Card>
          <Table>
            <thead><tr><Th className="w-14"><span className="sr-only">Image</span></Th><Th>#</Th><Th>Time</Th><Th>Detected objects</Th>
              <Th className="text-right">Green Score</Th><Th className="hidden text-right md:table-cell">Processing</Th><Th><span className="sr-only">Actions</span></Th></tr></thead>
            <tbody>
              {q.data.items.map((it) => (
                <Fragment key={it.id}>
                  <tr className="cursor-pointer hover:bg-surface-2" data-testid="detection-row" tabIndex={0} aria-expanded={open === it.id}
                    onClick={() => setOpen(open === it.id ? null : it.id)}
                    onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), setOpen(open === it.id ? null : it.id))}>
                    <Td>{it.image_url ? <img src={apiUrl(it.image_url)} alt="" loading="lazy" className="size-10 rounded object-cover" />
                      : <ImageOff className="size-4 text-muted-foreground" aria-label="no image stored" />}</Td>
                    <Td className="font-mono text-xs">{it.id}</Td>
                    <Td className="whitespace-nowrap text-xs">{fmtTime(it.created_at)}</Td>
                    <Td className="text-xs">
                      <span className="font-mono">{it.waste_items}</span> waste · {it.objects.filter((o) => o.decision === "ACCEPT" || o.decision === "REVIEW")
                        .map((o) => `${o.detection_label ?? "whole image"} → ${o.organiser_category}`).join(", ") || "none"}
                      {it.ignored_objects > 0 && <span className="text-muted-foreground"> · {it.ignored_objects} not waste</span>}
                    </Td>
                    <Td className="whitespace-nowrap text-right font-mono text-xs tabular">{it.image_green_score?.toFixed(1) ?? "—"}</Td>
                    <Td className="hidden whitespace-nowrap text-right font-mono text-xs tabular md:table-cell">{fmtMs(it.pipeline_ms, 0)}</Td>
                    <Td className="text-right">
                      <Button size="icon" variant="ghost" aria-label={`Delete analysis ${it.id}`} disabled={del.isPending}
                        onClick={(e) => { e.stopPropagation(); if (window.confirm(`Delete analysis #${it.id} and its image? This cannot be undone.`)) del.mutate(it.id); }}>
                        <Trash2 />
                      </Button>
                    </Td>
                  </tr>
                  {open === it.id && <tr><Td colSpan={7} className="bg-surface-2"><div className="py-3"><DetectionDetail id={it.id} /></div></Td></tr>}
                </Fragment>
              ))}
            </tbody>
          </Table>
          <Pager page={page} pages={pages} total={total} noun="analyses" setPage={setPage} />
        </Card>
      )}
    </>
  );
}

export default function History() {
  return (
    <>
      <PageHeader title="Detection History" description="Everything this device analysed, newest first (stored locally in SQLite). Click a row for details." />
      <Tabs defaultValue="detections">
        <TabsList className="mb-4">
          <TabsTrigger value="detections">Mixed-waste analyses</TabsTrigger>
          <TabsTrigger value="classifications">Single-item classifications</TabsTrigger>
        </TabsList>
        <TabsContent value="detections"><DetectionHistory /></TabsContent>
        <TabsContent value="classifications"><ClassificationHistory /></TabsContent>
      </Tabs>
    </>
  );
}

function ClassificationHistory() {
  const qc = useQueryClient();
  const del = useMutation({
    mutationFn: api.deleteClassification,
    onSuccess: () => { for (const k of ["history", "analytics"]) void qc.invalidateQueries({ queryKey: [k] }); },
  });
  const { settings } = useSettings();
  const model = useModelStatus();
  const [category, setCategory] = useState("");
  const [status, setStatus] = useState("");
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [page, setPage] = useState(0);
  const [open, setOpen] = useState<number | null>(null);
  const filters = { category: category || undefined, status: status || undefined, date_from: toIsoStart(from), date_to: toIsoEnd(to) };
  const q = useHistory({ ...filters, limit: PAGE, offset: page * PAGE });
  const total = q.data?.total ?? 0;
  const pages = Math.max(1, Math.ceil(total / PAGE));

  return (
    <>
      <div className="mb-3 flex justify-end">
        <Button asChild variant="outline" size="sm"><a href={api.historyCsvUrl(filters)} download><Download />Export CSV</a></Button>
      </div>
      <Card className="mb-4">
        <CardContent className="grid gap-3 p-4 sm:grid-cols-2 lg:grid-cols-4">
          <div><Label htmlFor="h-cat">Category</Label>
            <Select id="h-cat" value={category} onChange={(e) => { setCategory(e.target.value); setPage(0); }}>
              <option value="">All</option>
              {(model.data?.classes ?? ["Organic", "Recyclable"]).map((c) => <option key={c} value={c}>{c}</option>)}
            </Select></div>
          <div><Label htmlFor="h-status">EcoGuard</Label>
            <Select id="h-status" value={status} onChange={(e) => { setStatus(e.target.value); setPage(0); }}>
              <option value="">All</option><option value="ACCEPT">ACCEPT</option><option value="REVIEW">REVIEW</option>
            </Select></div>
          <div><Label htmlFor="h-from">From</Label><Input id="h-from" type="date" value={from} onChange={(e) => { setFrom(e.target.value); setPage(0); }} /></div>
          <div><Label htmlFor="h-to">To</Label><Input id="h-to" type="date" value={to} onChange={(e) => { setTo(e.target.value); setPage(0); }} /></div>
        </CardContent>
      </Card>

      {q.isError ? <ErrorState error={q.error} /> : q.isLoading ? <Skeleton className="h-64" /> : !q.data?.items.length ? (
        <EmptyState title="No classifications match">Classify an image on the Single-item Classification page, or clear the filters.</EmptyState>
      ) : (
        <Card>
          <Table>
            <thead><tr>
              <Th className="w-14"><span className="sr-only">Image</span></Th><Th>#</Th><Th>Time</Th><Th>Category</Th><Th className="text-right">Score</Th>
              <Th>Decision</Th><Th className="hidden md:table-cell">Action</Th><Th className="hidden text-right md:table-cell">Inference</Th><Th className="hidden lg:table-cell">Source</Th>
            </tr></thead>
            <tbody>
              {q.data.items.map((it) => (
                <Fragment key={it.id}>
                  <tr className="cursor-pointer hover:bg-surface-2" onClick={() => setOpen(open === it.id ? null : it.id)}
                    onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), setOpen(open === it.id ? null : it.id))}
                    tabIndex={0} aria-expanded={open === it.id} data-testid="history-row">
                    <Td>{it.image_url ? <img src={apiUrl(it.image_url)} alt="" loading="lazy" className="size-10 rounded object-cover" />
                      : <ImageOff className="size-4 text-muted-foreground" aria-label="no image stored" />}</Td>
                    <Td className="font-mono text-xs">{it.id}</Td>
                    <Td className="whitespace-nowrap text-xs">{fmtTime(it.created_at)}</Td>
                    <Td><CategoryBadge category={it.predicted_category} lang={settings.lang} /></Td>
                    <Td className="text-right font-mono text-xs tabular">{fmtPct(it.confidence, 2)}</Td>
                    <Td><DecisionBadge status={it.ecoguard_status} lang={settings.lang} /></Td>
                    <Td className="hidden font-mono text-xs md:table-cell">{it.disposal_action}</Td>
                    <Td className="hidden whitespace-nowrap text-right font-mono text-xs tabular md:table-cell">{fmtMs(it.inference_ms, 2)}</Td>
                    <Td className="hidden text-xs text-muted-foreground lg:table-cell">{it.source}</Td>
                  </tr>
                  {open === it.id && (
                    <tr><Td colSpan={9} className="bg-surface-2">
                      <div className="grid gap-4 py-2 md:grid-cols-2">
                        <ScoreBars scores={it.class_scores} predicted={it.predicted_category} />
                        <dl className="grid grid-cols-2 gap-2 text-xs">
                          {[
                            ["File", it.filename], ["Review reason", it.review_reason || "—"], ["Sorter", it.sort_message ?? "—"],
                            ["Ground truth", it.ground_truth ? `${it.ground_truth} (${it.correct ? "correct" : "incorrect"})` : "—"],
                            ["GreenScore event", it.greenscore_event_id ?? "—"], ["Model", `${it.model_profile} · ${it.model_sha256?.slice(0, 12) ?? ""}`],
                            ["Preprocess / total", `${fmtMs(it.preprocess_ms)} / ${fmtMs(it.total_ms)}`], ["Batch", it.batch_id ?? "—"],
                          ].map(([k, v]) => <div key={String(k)} className="min-w-0"><dt className="text-muted-foreground">{k}</dt><dd className="truncate font-mono" title={String(v)}>{v}</dd></div>)}
                        </dl>
                      </div>
                      <Button size="sm" variant="outline" disabled={del.isPending}
                        onClick={() => { if (window.confirm(`Delete classification #${it.id}? This cannot be undone.`)) del.mutate(it.id); }}>
                        <Trash2 />Delete
                      </Button>
                    </Td></tr>
                  )}
                </Fragment>
              ))}
            </tbody>
          </Table>
          <Pager page={page} pages={pages} total={total} noun="classification(s)" setPage={setPage} />
        </Card>
      )}
    </>
  );
}
