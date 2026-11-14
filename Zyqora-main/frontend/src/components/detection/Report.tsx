import { useState } from "react";
import { MetricCard, ScopeTag } from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert } from "@/components/ui/misc";
import { wasteObjects } from "@/lib/detection";
import { useSettings } from "@/lib/settings";
import { fmtMs } from "@/lib/utils";
import type { DetectionResult } from "@/types/mixed";
import { AnnotatedImage, GreenGauge, ObjectCard, ScoreBreakdown } from "./index";

function Counts({ title, data }: { title: string; data: Record<string, number> }) {
  const entries = Object.entries(data);
  return (
    <div>
      <p className="mb-1.5 text-xs text-muted-foreground">{title}</p>
      {entries.length ? (
        <div className="flex flex-wrap gap-1.5">
          {entries.map(([k, v]) => <Badge key={k} variant="outline"><span className="text-foreground">{k}</span><span className="font-mono">{v}</span></Badge>)}
        </div>
      ) : <p className="text-xs text-muted-foreground">—</p>}
    </div>
  );
}

export function DetectionReport({ r, imageSrc }: { r: DetectionResult; imageSrc: string }) {
  const { settings } = useSettings();
  const [sel, setSel] = useState<number | null>(null);
  const waste = wasteObjects(r);
  const selected = sel != null ? r.objects.find((o) => o.index === sel) : waste[0];
  const s = r.summary;
  const t = r.timing;
  return (
    <div className="space-y-6" data-testid="detection-report">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <MetricCard label="Detected objects" value={s.detected_objects} hint={`${s.ignored_non_waste} not waste · ${s.skipped} skipped`} />
        <MetricCard label="Waste items" value={s.waste_items} hint={`${s.accepted} reliable · ${s.review} review`} />
        <MetricCard label="Image Green Score" value={r.image_green_score.score != null ? r.image_green_score.score.toFixed(1) : "—"}
          hint={r.image_green_score.valid ? `${r.image_green_score.scored_items}/${r.image_green_score.waste_items} items scored` : "no reliable item"} />
        <MetricCard label="Pipeline on device" value={fmtMs(t.pipeline_ms, 0)}
          hint={`YOLO ${fmtMs(t.detector_inference_ms, 0)} · ${t.classified_crops}× MobileNetV2 ${fmtMs(t.classification_ms, 0)}`} />
      </div>
      {r.notes.map((n) => <Alert key={n} variant="neutral">{n}</Alert>)}
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
        <Card className="min-w-0">
          <CardHeader>
            <CardTitle>Detections <ScopeTag>{r.detector.profile}</ScopeTag></CardTitle>
            <CardDescription>Solid boxes are classified waste items (green = reliable, amber = review); dashed boxes are not waste. Click a box.</CardDescription>
          </CardHeader>
          <CardContent>
            <AnnotatedImage src={imageSrc} size={r.image_size} objects={r.objects} selected={selected?.index} onSelect={setSel} />
            <div className="mt-4 grid gap-4 sm:grid-cols-3">
              <Counts title={`Organiser labels (${r.label_profile.title})`} data={s.by_organiser_category} />
              <Counts title="Material (from detector label)" data={s.by_material} />
              <Counts title="Disposal stream" data={s.by_stream} />
            </div>
          </CardContent>
        </Card>
        <Card className="min-w-0">
          <CardHeader><CardTitle>Objects</CardTitle>
            <CardDescription>Model label = MobileNetV2 output on the crop. Organiser label = profile mapping; REVIEW items show as Unknown.</CardDescription></CardHeader>
          <CardContent className="max-h-[640px] space-y-2 overflow-y-auto">
            {r.objects.length === 0 && <p className="text-sm text-muted-foreground">Nothing detected and whole-image fallback is off.</p>}
            {r.objects.map((o) => <ObjectCard key={o.index} o={o} lang={settings.lang} active={selected?.index === o.index} onSelect={() => setSel(o.index)} />)}
          </CardContent>
        </Card>
      </div>
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[260px_minmax(0,1fr)]">
        <Card>
          <CardContent className="flex flex-col items-center gap-2 p-6">
            <GreenGauge score={r.image_green_score.score} band={r.image_green_score.band} label="Image Green Score" />
            <p className="text-center text-xs text-muted-foreground">{r.image_green_score.explanation}</p>
          </CardContent>
        </Card>
        <Card className="min-w-0">
          <CardHeader><CardTitle>{selected ? `Item #${selected.index}: recommendation and Green Score` : "Item details"}</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            {selected?.recommendation ? (
              <div className="rounded-lg bg-surface-2 p-3 text-sm">
                <p className="font-medium">{selected.recommendation.stream_title[settings.lang] || selected.recommendation.stream_title.en}</p>
                <p className={settings.lang === "ta" ? "font-tamil" : undefined}>{selected.recommendation.guidance[settings.lang] || selected.recommendation.guidance.en}</p>
                {selected.recommendation.warnings.map((w) => <p key={w} className="mt-1 text-xs text-review">⚠ {w}</p>)}
              </div>
            ) : <p className="text-sm text-muted-foreground">Select a waste item.</p>}
            {selected && <ScoreBreakdown gs={selected.green_score} />}
          </CardContent>
        </Card>
      </div>
      {r.mode_note && <p className="text-xs text-muted-foreground">{r.mode_note}</p>}
    </div>
  );
}

