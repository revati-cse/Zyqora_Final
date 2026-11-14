import { AlertTriangle, Recycle } from "lucide-react";
import { useId } from "react";
import { Badge } from "@/components/ui/badge";
import { bandTone, DECISION_TEXT, DECISION_VAR, NOT_SCORED_TEXT, objectTitle } from "@/lib/detection";
import type { Lang } from "@/lib/i18n";
import { cn, fmtPct } from "@/lib/utils";
import type { DetectedObject, GreenScore } from "@/types/mixed";

/** Uploaded image with the detector's boxes drawn in image-pixel coordinates (SVG viewBox = image size). */
export function AnnotatedImage({ src, size, objects, selected, onSelect, className }: {
  src: string; size: [number, number]; objects: DetectedObject[]; selected?: number | null;
  onSelect?: (i: number) => void; className?: string;
}) {
  const [w, h] = size;
  const fs = Math.max(12, Math.round(Math.max(w, h) / 42));
  const lw = Math.max(2, Math.round(Math.max(w, h) / 320));
  return (
    <div className={cn("relative overflow-hidden rounded-lg border border-border bg-surface-2", className)}>
      <img src={src} alt="Analysed image" className="block h-auto w-full" />
      <svg viewBox={`0 0 ${w} ${h}`} className="absolute inset-0 size-full" preserveAspectRatio="none" role="group"
        aria-label={`${objects.filter((o) => o.source === "detector").length} detected objects`}>
        {objects.filter((o) => o.source === "detector").map((o) => {
          const [x1, y1, x2, y2] = o.box;
          const color = DECISION_VAR[o.decision];
          const active = selected === o.index;
          const label = o.decision === "ACCEPT" || o.decision === "REVIEW"
            ? `#${o.index} ${o.organiser_category ?? "?"}${o.green_score.score != null ? ` · ${o.green_score.score.toFixed(0)}` : ""}`
            : `#${o.index} ${o.detection_label}`;
          const tw = label.length * fs * 0.56 + 10;
          const ty = y1 - fs - 6 >= 0 ? y1 - fs - 6 : y1;
          return (
            <g key={o.index} onClick={() => onSelect?.(o.index)} className={onSelect ? "cursor-pointer" : undefined}
              role={onSelect ? "button" : undefined} tabIndex={onSelect ? 0 : undefined}
              aria-label={`Object ${o.index}: ${objectTitle(o)}, ${DECISION_TEXT[o.decision]}`}
              onKeyDown={(e) => { if (onSelect && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); onSelect(o.index); } }}>
              <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} fill={active ? color : "transparent"} fillOpacity={active ? 0.12 : 0}
                stroke={color} strokeWidth={active ? lw * 2 : lw} strokeDasharray={o.decision === "IGNORED" || o.decision === "SKIPPED" ? `${lw * 4} ${lw * 3}` : undefined} />
              <rect x={x1} y={ty} width={tw} height={fs + 6} fill={color} rx={3} />
              <text x={x1 + 5} y={ty + fs} fontSize={fs} fill="var(--surface)" fontFamily="ui-monospace, monospace">{label}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/** Circular 0-100 gauge. null = not scored (never drawn as zero). */
export function GreenGauge({ score, band, size = 168, label = "Green Score" }: { score: number | null; band: string | null; size?: number; label?: string }) {
  const id = useId();
  const r = 42;
  const c = 2 * Math.PI * r;
  const frac = score == null ? 0 : Math.max(0, Math.min(100, score)) / 100;
  return (
    <figure className="flex flex-col items-center" aria-labelledby={id}>
      <svg viewBox="0 0 100 100" width={size} height={size} role="img" aria-label={score == null ? `${label}: not scored` : `${label} ${score.toFixed(1)} of 100, ${band}`}>
        <circle cx="50" cy="50" r={r} fill="none" stroke="var(--border)" strokeWidth="8" />
        {score != null && (
          <circle cx="50" cy="50" r={r} fill="none" stroke={bandTone(score)} strokeWidth="8" strokeLinecap="round"
            strokeDasharray={`${c * frac} ${c}`} transform="rotate(-90 50 50)" />
        )}
        <text x="50" y={score == null ? 52 : 50} textAnchor="middle" fontSize={score == null ? 9 : 22} fontWeight="600"
          fill="var(--fg)" fontFamily="ui-monospace, monospace">{score == null ? "not scored" : score.toFixed(score % 1 ? 1 : 0)}</text>
        {score != null && <text x="50" y="64" textAnchor="middle" fontSize="7" fill="var(--muted-fg)">/ 100</text>}
      </svg>
      <figcaption id={id} className="mt-1 text-center">
        <p className="text-xs text-muted-foreground">{label}</p>
        {band && <p className="text-sm font-medium" style={{ color: bandTone(score) }}>{band}</p>}
      </figcaption>
    </figure>
  );
}

export function ScoreBreakdown({ gs }: { gs: GreenScore }) {
  if (!gs.scored) return <p className="text-sm text-muted-foreground">{gs.explanation}</p>;
  return (
    <div className="space-y-3">
      <div className="overflow-x-auto">
        <table className="w-full text-xs sm:text-sm">
          <thead><tr className="text-left text-xs text-muted-foreground">
            <th className="py-1.5 pr-2 font-medium">Component</th><th className="py-1.5 pr-2 text-right font-medium">Rating</th>
            <th className="whitespace-nowrap py-1.5 pr-2 text-right font-medium">× Weight</th><th className="py-1.5 text-right font-medium">Points</th></tr></thead>
          <tbody className="font-mono tabular">
            {gs.components.map((c) => (
              <tr key={c.key} className="border-t border-border">
                <td className="py-1.5 pr-2 font-sans">{c.title}</td><td className="py-1.5 pr-2 text-right">{c.value}</td>
                <td className="py-1.5 pr-2 text-right">{c.weight.toFixed(2)}</td><td className="py-1.5 text-right">{c.weighted.toFixed(2)}</td>
              </tr>
            ))}
            <tr className="border-t border-border font-semibold"><td className="py-1.5 font-sans" colSpan={3}>Green Score</td>
              <td className="py-1.5 text-right">{gs.score?.toFixed(2)}</td></tr>
          </tbody>
        </table>
      </div>
      <p className="font-mono text-xs text-muted-foreground">{gs.explanation}</p>
      <ul className="space-y-1 text-xs text-muted-foreground">
        {gs.components.map((c) => <li key={c.key}><span className="font-medium text-foreground">{c.title}:</span> {c.rationale}</li>)}
      </ul>
      <p className="text-xs text-muted-foreground">Basis: <span className="font-mono">{gs.basis}</span> · rules {gs.scoring_version}.
        Ratings are project-defined assumptions, not measurements; the model's confidence is not used.</p>
    </div>
  );
}

export function ObjectCard({ o, lang, active, onSelect }: { o: DetectedObject; lang: Lang; active?: boolean; onSelect?: () => void }) {
  const c = o.classification;
  const waste = o.decision === "ACCEPT" || o.decision === "REVIEW";
  return (
    <button type="button" onClick={onSelect} data-testid="object-card"
      className={cn("w-full rounded-lg border p-3 text-left transition-colors", active ? "border-primary bg-surface-2" : "border-border hover:bg-surface-2")}>
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-xs text-muted-foreground">#{o.index}</span>
        <span className="text-sm font-medium">{objectTitle(o)}</span>
        {o.detection_confidence != null && <span className="font-mono text-xs text-muted-foreground">det {fmtPct(o.detection_confidence, 0)}</span>}
        <Badge variant={o.decision === "ACCEPT" ? "accept" : o.decision === "REVIEW" ? "review" : "neutral"} className="ml-auto">{o.decision}</Badge>
      </div>
      {waste ? (
        <dl className="mt-2 grid grid-cols-[6.5rem_minmax(0,1fr)] gap-x-3 gap-y-1 text-xs">
          <dt className="text-muted-foreground">Organiser label</dt><dd className="font-medium" data-testid="organiser-category">{o.organiser_category}</dd>
          <dt className="text-muted-foreground">Model label</dt>
          <dd className="font-mono">{c?.model_label ?? "—"} {c?.confidence != null && <span className="text-muted-foreground">{fmtPct(c.confidence, 1)}</span>}</dd>
          <dt className="text-muted-foreground">Material</dt><dd>{o.material.category}{o.material.category === "Unknown" && o.material.note ? <span className="text-muted-foreground"> · {o.material.note}</span> : null}</dd>
          <dt className="text-muted-foreground">Disposal</dt>
          <dd className="flex items-center gap-1"><Recycle className="size-3" aria-hidden />{o.recommendation?.stream_title[lang] || o.recommendation?.stream_title.en}</dd>
          <dt className="text-muted-foreground">Green Score</dt>
          <dd className="font-mono" style={{ color: bandTone(o.green_score.score) }}>
            {o.green_score.score != null ? `${o.green_score.score.toFixed(1)} · ${o.green_score.band}` : NOT_SCORED_TEXT[o.green_score.not_scored_reason ?? ""] ?? "not scored"}
          </dd>
        </dl>
      ) : (
        <p className="mt-1 text-xs text-muted-foreground">{DECISION_TEXT[o.decision]} — drawn for context, not classified or scored.</p>
      )}
      {o.conflict && <p className="mt-2 flex items-start gap-1 text-xs text-review"><AlertTriangle className="mt-0.5 size-3 shrink-0" aria-hidden />{o.conflict}</p>}
    </button>
  );
}
