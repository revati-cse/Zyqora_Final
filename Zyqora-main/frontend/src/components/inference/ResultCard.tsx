import { Cpu, Info, Volume2 } from "lucide-react";
import { motion } from "motion/react";
import { useState } from "react";
import { CategoryBadge, DecisionBadge } from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { api, apiUrl } from "@/lib/api";
import { categoryLabel, t, type Lang } from "@/lib/i18n";
import { usePrefersReducedMotion, useSettings } from "@/lib/settings";
import { cn, fmtMs, fmtPct } from "@/lib/utils";
import type { Classification } from "@/types/api";

export function ScoreBars({ scores, threshold, predicted }: { scores: Record<string, number>; threshold?: number; predicted?: string | null }) {
  const entries = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  return (
    <div className="space-y-2.5">
      {entries.map(([name, p]) => (
        <div key={name}>
          <div className="mb-1 flex items-baseline justify-between text-xs">
            <span className={cn(name === predicted && "font-medium")}>{name}</span>
            <span className="font-mono tabular">{fmtPct(p, 2)}</span>
          </div>
          <div className="relative h-2 rounded-full bg-surface-2" role="img" aria-label={`${name} score ${fmtPct(p, 1)}`}>
            <div className={cn("h-full rounded-full", name === predicted ? "bg-primary" : "bg-chart-4")} style={{ width: `${p * 100}%` }} />
            {threshold !== undefined && (
              <div className="absolute -top-1 h-4 w-px bg-foreground/50" style={{ left: `${threshold * 100}%` }} title={`EcoGuard threshold ${fmtPct(threshold, 0)}`} />
            )}
          </div>
        </div>
      ))}
      {threshold !== undefined && <p className="text-[11px] text-muted-foreground">Vertical tick = EcoGuard threshold ({fmtPct(threshold, 0)}).</p>}
    </div>
  );
}

function VoiceButton({ action, lang }: { action: string; lang: Lang }) {
  const { settings } = useSettings();
  const [state, setState] = useState<string | null>(null);
  if (settings.voice === "off") return null;
  const play = async () => {
    setState(null);
    try {
      if (settings.voice === "device") {
        const r = await api.playOnDevice(lang, action);
        setState(r.played ? "Played on device speaker" : `Not played on device: ${r.detail}`);
      } else {
        await new Audio(api.voiceClipUrl(lang, action)).play();
      }
    } catch (e) {
      setState(`Audio unavailable: ${(e as Error).message}`);
    }
  };
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button size="sm" variant="outline" onClick={play}><Volume2 />{t("play", lang)}</Button>
      {state && <span className="text-xs text-muted-foreground">{state}</span>}
    </div>
  );
}

export function ResultCard({ result, previewUrl }: { result: Classification; previewUrl?: string }) {
  const { settings } = useSettings();
  const lang = settings.lang;
  const reduced = usePrefersReducedMotion();
  const review = result.ecoguard.status === "REVIEW";
  const imgSrc = previewUrl ?? (result.image_url ? apiUrl(result.image_url) : undefined);

  return (
    <motion.div initial={reduced ? false : { opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25, ease: "easeOut" }}>
      <Card className={cn("overflow-hidden", review ? "border-review/40" : "border-accept/40")} data-testid="result-card">
        <div className={cn("flex flex-wrap items-center justify-between gap-2 px-5 py-3", review ? "bg-review-bg" : "bg-accept-bg")}>
          <DecisionBadge status={result.ecoguard.status} lang={lang} />
          <span className="font-mono text-xs text-muted-foreground">#{result.id ?? "—"} · {result.source}</span>
        </div>
        <CardContent className="grid gap-6 pt-5 md:grid-cols-[minmax(0,220px)_1fr]">
          {imgSrc && <img src={imgSrc} alt={`Classified image ${result.filename}`} className="aspect-square w-full rounded-md border border-border object-cover" />}
          <div className="min-w-0 space-y-5">
            <div>
              <p className="text-xs text-muted-foreground">{t("predicted", lang)}</p>
              <p className={cn("text-3xl font-semibold tracking-tight", lang === "ta" && "font-tamil text-2xl")} data-testid="predicted-category">
                {categoryLabel(result.predicted_category, lang)}
              </p>
              <p className="mt-1 font-mono text-sm tabular">
                score {fmtPct(result.confidence, 2)} <span className="text-xs text-muted-foreground">(softmax, not calibrated)</span>
              </p>
            </div>
            <ScoreBars scores={result.class_scores} threshold={result.ecoguard.threshold} predicted={result.predicted_category} />
            <div className={cn("rounded-md p-3 text-sm", review ? "bg-review-bg text-review" : "bg-surface-2")}>
              <p className="text-xs font-medium opacity-80">{t("guidance", lang)}</p>
              <p className={cn("mt-0.5", lang === "ta" && "font-tamil")}>{result.guidance[lang] || result.guidance.en}</p>
              {review && (
                <p className={cn("mt-2 text-xs", lang === "ta" && "font-tamil")}>
                  <span className="font-medium">{t("reviewWhy", lang)}</span> {result.ecoguard.explanation[lang] || result.ecoguard.explanation.en}.{" "}
                  {t("noBin", lang)}
                </p>
              )}
            </div>
            <VoiceButton action={result.disposal_action} lang={lang} />
          </div>
        </CardContent>
        <div className="grid grid-cols-2 gap-px border-t border-border bg-border text-xs sm:grid-cols-4">
          {[
            ["Inference", fmtMs(result.timing.inference_ms, 2)],
            ["Preprocess", fmtMs(result.timing.preprocess_ms, 1)],
            ["Total on device", fmtMs(result.timing.total_ms, 1)],
            ["Disposal action", result.disposal_action],
          ].map(([k, v]) => (
            <div key={k} className="bg-surface px-4 py-3">
              <p className="text-muted-foreground">{k}</p>
              <p className="mt-0.5 font-mono tabular">{v}</p>
            </div>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-2 border-t border-border px-5 py-3 text-xs text-muted-foreground">
          <Cpu className="size-3.5" aria-hidden />
          <span>{result.model.profile}</span>·<span>{result.model.runtime} + {result.model.backend}</span>·
          <span>{result.model.device.device_model ?? result.model.device.machine}</span>
          {result.model.fallback_used && <Badge variant="danger">FALLBACK MODEL</Badge>}
          <span className="ml-auto inline-flex items-center gap-1">
            <Info className="size-3.5" aria-hidden />
            {result.sort.message}
          </span>
        </div>
        {(result.greenscore_event_id || result.ground_truth) && (
          <div className="flex flex-wrap gap-2 border-t border-border px-5 py-3 text-xs">
            {result.greenscore_event_id && <Badge variant="outline">GreenScore event #{result.greenscore_event_id}: PENDING verification</Badge>}
            {result.ground_truth && (
              <Badge variant={result.correct ? "accept" : "danger"}>
                truth <CategoryBadge category={result.ground_truth} /> · {result.correct ? "correct" : "incorrect"}
              </Badge>
            )}
          </div>
        )}
      </Card>
    </motion.div>
  );
}
