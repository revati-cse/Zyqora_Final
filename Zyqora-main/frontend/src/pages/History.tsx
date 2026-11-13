import { ChevronLeft, ChevronRight, Download, ImageOff } from "lucide-react";
import { Fragment, useState } from "react";
import { CategoryBadge, DecisionBadge, EmptyState, ErrorState, PageHeader } from "@/components/common";
import { ScoreBars } from "@/components/inference/ResultCard";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input, Label, Select } from "@/components/ui/input";
import { Skeleton, Table, Td, Th } from "@/components/ui/misc";
import { useHistory, useModelStatus } from "@/hooks/useApi";
import { api, apiUrl } from "@/lib/api";
import { useSettings } from "@/lib/settings";
import { fmtMs, fmtPct, fmtTime, toIsoEnd, toIsoStart } from "@/lib/utils";

const PAGE = 20;

export default function History() {
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
      <PageHeader title="History" description="Every classification served by this device, newest first (stored locally in SQLite)."
        actions={<Button asChild variant="outline" size="sm"><a href={api.historyCsvUrl(filters)} download><Download />Export CSV</a></Button>} />
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
        <EmptyState title="No classifications match">Classify an image on the Live Classification page, or clear the filters.</EmptyState>
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
                    <Td className="hidden text-right font-mono text-xs tabular md:table-cell">{fmtMs(it.inference_ms, 2)}</Td>
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
                    </Td></tr>
                  )}
                </Fragment>
              ))}
            </tbody>
          </Table>
          <div className="flex items-center justify-between border-t border-border px-4 py-3 text-xs text-muted-foreground">
            <span>{total} classification(s)</span>
            <div className="flex items-center gap-2">
              <Button size="sm" variant="outline" disabled={page === 0} onClick={() => setPage((p) => p - 1)} aria-label="Previous page"><ChevronLeft /></Button>
              <span className="font-mono">{page + 1} / {pages}</span>
              <Button size="sm" variant="outline" disabled={page + 1 >= pages} onClick={() => setPage((p) => p + 1)} aria-label="Next page"><ChevronRight /></Button>
            </div>
          </div>
        </Card>
      )}
    </>
  );
}
