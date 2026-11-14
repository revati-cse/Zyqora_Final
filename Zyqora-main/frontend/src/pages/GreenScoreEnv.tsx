import { useMutation } from "@tanstack/react-query";
import { Calculator } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router";
import { CountBars, Lines } from "@/components/charts/Charts";
import { EmptyState, ErrorState, MetricCard, PageHeader, ScopeTag } from "@/components/common";
import { AnnotatedImage, GreenGauge, ScoreBreakdown } from "@/components/detection";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input, Label, Select } from "@/components/ui/input";
import { Alert, Skeleton } from "@/components/ui/misc";
import { useAnalyticsAll, useDetection, useDetections, useGreenScoreRules } from "@/hooks/useApi";
import { api, apiUrl } from "@/lib/api";
import { bandTone, NOT_SCORED_TEXT, objectTitle, wasteObjects } from "@/lib/detection";
import { fmtTime } from "@/lib/utils";

const COMPONENTS = [
  ["recycling_potential", "Recycling / composting potential", 0.4],
  ["environmental_impact", "Environmental impact rating", 0.35],
  ["disposal_suitability", "Disposal suitability", 0.25],
] as const;

function CalculatorCard() {
  const rules = useGreenScoreRules();
  const [mode, setMode] = useState<"material" | "custom">("material");
  const [material, setMaterial] = useState("Metal");
  const [vals, setVals] = useState({ recycling_potential: 90, environmental_impact: 80, disposal_suitability: 100 });
  const m = useMutation({
    mutationFn: () => api.greenscoreCompute(mode === "material" ? { material } : { components: vals }),
  });
  const materials = Object.keys(rules.data?.materials ?? {});
  return (
    <Card>
      <CardHeader><CardTitle>Calculator</CardTitle>
        <CardDescription>Computed by the backend GreenScoreEngine (same code as the pipeline). Default custom values reproduce the 90 / 80 / 100 → 89 example.</CardDescription></CardHeader>
      <CardContent className="space-y-4">
        <div className="flex gap-2" role="radiogroup" aria-label="Calculator input">
          {(["material", "custom"] as const).map((k) => (
            <Button key={k} size="sm" variant={mode === k ? "default" : "outline"} role="radio" aria-checked={mode === k} onClick={() => { setMode(k); m.reset(); }}>
              {k === "material" ? "By material rule" : "Custom ratings"}
            </Button>
          ))}
        </div>
        {mode === "material" ? (
          <div>
            <Label htmlFor="gs-material">Material</Label>
            <Select id="gs-material" value={material} onChange={(e) => setMaterial(e.target.value)}>
              {materials.map((x) => <option key={x}>{x}</option>)}
            </Select>
          </div>
        ) : (
          <div className="grid gap-3 sm:grid-cols-3">
            {COMPONENTS.map(([k, label, w]) => (
              <div key={k}>
                <Label htmlFor={`gs-${k}`}>{label} <span className="font-mono text-muted-foreground">× {w}</span></Label>
                <Input id={`gs-${k}`} type="number" min={0} max={100} value={vals[k]}
                  onChange={(e) => setVals((v) => ({ ...v, [k]: Math.max(0, Math.min(100, Number(e.target.value))) }))} />
              </div>
            ))}
          </div>
        )}
        <Button onClick={() => m.mutate()} disabled={m.isPending}><Calculator />Calculate</Button>
        {m.isError && <ErrorState error={m.error} />}
        {m.data && (
          <div className="grid gap-4 sm:grid-cols-[auto_minmax(0,1fr)]">
            <GreenGauge score={m.data.score} band={m.data.band} size={128} />
            <ScoreBreakdown gs={m.data} />
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function RulesCard() {
  const rules = useGreenScoreRules();
  if (rules.isLoading) return <Skeleton className="h-64" />;
  if (rules.isError) return <ErrorState error={rules.error} />;
  const r = rules.data!;
  return (
    <Card className="min-w-0">
      <CardHeader><CardTitle>Rules <ScopeTag>{r.version}</ScopeTag></CardTitle><CardDescription>{r.disclaimer}</CardDescription></CardHeader>
      <CardContent className="space-y-4">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[480px] text-sm">
            <thead><tr className="text-left text-xs text-muted-foreground"><th className="py-1.5 pr-3 font-medium">Basis</th>
              {COMPONENTS.map(([k, l, w]) => <th key={k} className="py-1.5 pr-3 text-right font-medium">{l.split(" ")[0]} ×{w}</th>)}
              <th className="py-1.5 text-right font-medium">Score</th></tr></thead>
            <tbody className="font-mono tabular">
              {[...Object.entries(r.materials).map(([k, v]) => [k, v] as const), ...Object.entries(r.model_classes).map(([k, v]) => [`${k} (material unknown)`, v] as const)].map(([name, v]) => {
                const score = COMPONENTS.reduce((s, [k, , w]) => s + v[k].value * w, 0);
                return (
                  <tr key={name} className="border-t border-border">
                    <td className="py-1.5 pr-3 font-sans">{name}</td>
                    {COMPONENTS.map(([k]) => <td key={k} className="py-1.5 pr-3 text-right" title={v[k].why}>{v[k].value}</td>)}
                    <td className="py-1.5 text-right font-semibold" style={{ color: bandTone(score) }}>{score.toFixed(1)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="flex flex-wrap gap-2 text-xs">
          {r.bands.slice().reverse().map((b) => <span key={b.min} className="rounded border border-border px-2 py-1"><span className="font-mono">≥ {b.min}</span> {b.label}</span>)}
        </div>
        <p className="text-xs text-muted-foreground">Hover a rating to read its rationale. Items are scored only when reliably identified (EcoGuard ACCEPT and no detector/classifier conflict).</p>
      </CardContent>
    </Card>
  );
}

export default function GreenScoreEnv() {
  const list = useDetections({ limit: 30 });
  const [chosen, setChosen] = useState<number | null>(null);
  const id = chosen ?? list.data?.items[0]?.id ?? null;
  const det = useDetection(id);
  const an = useAnalyticsAll();
  const [sel, setSel] = useState<number | null>(null);
  const r = det.data;
  const waste = r ? wasteObjects(r) : [];
  const selected = r?.objects.find((o) => o.index === sel) ?? waste.find((o) => o.green_score.scored) ?? waste[0];
  const g = an.data?.detections?.green_score;
  const imgSrc = r?.image_url ? apiUrl(r.image_url) : null;

  return (
    <>
      <PageHeader title="Green Score"
        description="A rule-based, project-defined sustainability indicator (0–100): recycling potential × 0.40 + environmental impact × 0.35 + disposal suitability × 0.25. It is not a certified lifecycle assessment." />
      <Alert variant="neutral" className="mb-6">
        This is different from <Link to="/rewards" className="underline">Green Points</Link>, which reward people for supervisor-verified disposals.
      </Alert>
      <div className="mb-6 grid grid-cols-2 gap-3 md:grid-cols-4">
        <MetricCard label="Mean image score" value={g?.mean_image_score != null ? g.mean_image_score.toFixed(1) : "—"} loading={an.isLoading} hint="stored analyses" />
        <MetricCard label="Mean item score" value={g?.mean_item_score != null ? g.mean_item_score.toFixed(1) : "—"} loading={an.isLoading} />
        <MetricCard label="Scored items" value={g?.scored_items ?? "—"} loading={an.isLoading} hint={`of ${an.data?.detections?.waste_items ?? 0} waste items`} />
        <MetricCard label="Analyses" value={an.data?.detections?.images ?? "—"} loading={an.isLoading} />
      </div>

      {list.isError && <ErrorState error={list.error} />}
      {list.data && !list.data.items.length && (
        <EmptyState title="No analyses yet"><Link to="/detect" className="underline">Analyse a mixed-waste image</Link> to see its Green Score.</EmptyState>
      )}
      {r && (
        <div className="mb-6 grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
          <Card className="min-w-0">
            <CardHeader>
              <CardTitle>Analysis #{r.id}</CardTitle>
              <div className="pt-1">
                <Label htmlFor="gs-pick" className="sr-only">Choose analysis</Label>
                <Select id="gs-pick" value={id ?? ""} onChange={(e) => { setChosen(Number(e.target.value)); setSel(null); }}>
                  {list.data?.items.map((d) => <option key={d.id} value={d.id}>#{d.id} · {fmtTime(d.created_at)} · {d.filename} · GS {d.image_green_score?.toFixed(1) ?? "—"}</option>)}
                </Select>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-center"><GreenGauge score={r.image_green_score.score} band={r.image_green_score.band} label="Overall image score" size={190} /></div>
              <p className="text-center text-xs text-muted-foreground">{r.image_green_score.explanation}</p>
              {imgSrc && <AnnotatedImage src={imgSrc} size={r.image_size} objects={r.objects} selected={selected?.index} onSelect={setSel} />}
            </CardContent>
          </Card>
          <Card className="min-w-0">
            <CardHeader><CardTitle>Items</CardTitle><CardDescription>Click an item for its weighted calculation and recommended disposal.</CardDescription></CardHeader>
            <CardContent className="space-y-4">
              <ul className="divide-y divide-border rounded-lg border border-border">
                {waste.map((o) => (
                  <li key={o.index}>
                    <button type="button" onClick={() => setSel(o.index)}
                      className={`flex w-full items-center gap-3 px-3 py-2 text-left text-sm ${selected?.index === o.index ? "bg-surface-2" : "hover:bg-surface-2"}`}>
                      <span className="font-mono text-xs text-muted-foreground">#{o.index}</span>
                      <span className="min-w-0 flex-1 truncate">{objectTitle(o)} · {o.organiser_category} · {o.material.category}</span>
                      <span className="whitespace-nowrap font-mono" style={{ color: bandTone(o.green_score.score) }}>
                        {o.green_score.score != null ? o.green_score.score.toFixed(1) : NOT_SCORED_TEXT[o.green_score.not_scored_reason ?? ""] ?? "—"}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
              {selected && (
                <>
                  {selected.recommendation && (
                    <div className="rounded-lg bg-surface-2 p-3 text-sm">
                      <p className="text-xs text-muted-foreground">Recommended disposal</p>
                      <p className="font-medium">{selected.recommendation.stream_title.en}</p>
                      <p>{selected.recommendation.guidance.en}</p>
                    </div>
                  )}
                  <ScoreBreakdown gs={selected.green_score} />
                </>
              )}
            </CardContent>
          </Card>
        </div>
      )}

      <div className="mb-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card className="min-w-0">
          <CardHeader><CardTitle>Historical trend <ScopeTag>stored analyses</ScopeTag></CardTitle></CardHeader>
          <CardContent>
            {g?.series.length ? <Lines data={g.series.map((s) => ({ ...s, label: `#${s.id}` }))} x="label" lines={[{ key: "score", name: "image Green Score" }]} domain={[0, 100]} />
              : <EmptyState title="No scored analyses yet" />}
          </CardContent>
        </Card>
        <Card className="min-w-0">
          <CardHeader><CardTitle>Items by band</CardTitle></CardHeader>
          <CardContent>{g && Object.keys(g.by_band).length ? <CountBars data={g.by_band} /> : <EmptyState title="No scored items yet" />}</CardContent>
        </Card>
      </div>
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <CalculatorCard />
        <RulesCard />
      </div>
    </>
  );
}
