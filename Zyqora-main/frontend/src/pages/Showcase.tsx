import { ArrowRight, Cpu, Leaf, ScanSearch, ShieldCheck, Tags, Users } from "lucide-react";
import type { ReactNode } from "react";
import { Link } from "react-router";
import { PageHeader, ScopeTag } from "@/components/common";
import { DetectorEvalCard, DetectorLatencyCard, ModelComparison } from "@/components/detection/Comparison";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useAnalyticsAll, useModelMetrics, useModelsStatus, useSystemMetrics } from "@/hooks/useApi";
import { fmtMs, fmtPct } from "@/lib/utils";

function Block({ title, icon, children }: { title: string; icon?: ReactNode; children: ReactNode }) {
  return (
    <Card className="min-w-0 rounded-md">
      <CardHeader className="pb-2"><CardTitle className="flex items-center gap-2 text-base">{icon}{title}</CardTitle></CardHeader>
      <CardContent className="space-y-2 text-sm text-muted-foreground">{children}</CardContent>
    </Card>
  );
}

const ARCH = [
  ["Image / camera", "upload, browser camera or Pi camera"],
  ["YOLO11n", "multiple boxes, NMS (ExecuTorch XNNPACK or ONNX Runtime)"],
  ["Crop", "padded box per object"],
  ["MobileNetV2", "Organic / Recyclable per crop + EcoGuard"],
  ["Organiser labels", "configurable profile mapping"],
  ["Rules", "disposal stream + Green Score"],
  ["SQLite + dashboard", "history, analytics, monitoring"],
];

export default function Showcase() {
  const st = useModelsStatus();
  const mm = useModelMetrics();
  const an = useAnalyticsAll();
  const sys = useSystemMetrics();
  const pi = mm.data?.validation?.pi_reproduction;
  const d = an.data?.detections;
  return (
    <>
      <PageHeader title="Hackathon Showcase" description="Zyqora AI 4.0 — offline mixed-waste detection, segregation guidance and Green Score on a Raspberry Pi 5."
        actions={<Button asChild><Link to="/detect"><ScanSearch />Live demo</Link></Button>} />

      <div className="mb-6 grid grid-cols-1 gap-4 md:grid-cols-3">
        <Block title="Problem statement">
          <p>Mixed household waste is rarely separated at source, so recyclables are contaminated and organic waste goes to landfill.</p>
        </Block>
        <Block title="Existing limitation">
          <p>A single-image classifier gives one label for the whole photo. A photo with a bottle, a peel and a phone gets one answer, and a 2-class model forces every item into Organic or Recyclable.</p>
        </Block>
        <Block title="Our solution">
          <p>Detect every object, classify each crop on the device, refuse to guess when unsure (EcoGuard → Unknown), route electronics to e-waste, and score each reliable item with a transparent Green Score.</p>
        </Block>
      </div>

      <Card className="mb-6 rounded-md">
        <CardHeader><CardTitle>YOLO + MobileNetV2 architecture</CardTitle></CardHeader>
        <CardContent>
          <ol className="flex flex-col gap-2 lg:flex-row lg:items-stretch">
            {ARCH.map(([t, s], i) => (
              <li key={t} className="flex items-center gap-2 lg:flex-1">
                <div className="flex-1 rounded-md border border-border bg-surface-2 p-3">
                  <p className="font-mono text-[10px] text-muted-foreground">{String(i + 1).padStart(2, "0")}</p>
                  <p className="text-sm font-medium">{t}</p>
                  <p className="text-xs text-muted-foreground">{s}</p>
                </div>
                {i < ARCH.length - 1 && <ArrowRight className="hidden size-4 shrink-0 text-muted-foreground lg:block" aria-hidden />}
              </li>
            ))}
          </ol>
          <p className="mt-3 text-xs text-muted-foreground">Why two models: the detector localises objects but its COCO labels are not waste categories; the classifier knows the organiser's waste classes but sees one item at a time. Trade-off: latency grows with the number of objects (one MobileNetV2 run per crop).</p>
        </CardContent>
      </Card>

      <div className="mb-6 grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        <Block title="Raspberry Pi 5 utilisation" icon={<Cpu className="size-4" aria-hidden />}>
          <p>Both models are loaded once and run on the Pi's CPU (XNNPACK); no cloud, no GPU.</p>
          <ul className="space-y-0.5 font-mono text-xs">
            <li>device: {sys.data?.device_model ?? "—"}</li>
            <li>CPU now: {sys.data?.cpu_percent != null ? `${sys.data.cpu_percent.toFixed(0)}%` : "—"} · RAM {sys.data?.memory.used_percent != null ? `${sys.data.memory.used_percent.toFixed(0)}%` : "—"}</li>
            <li>classifier on Pi: {pi ? `${fmtMs(pi.inference_median_ms, 1)} median` : "—"} <ScopeTag>Pi</ScopeTag></li>
          </ul>
        </Block>
        <Block title="Green Score" icon={<Leaf className="size-4" aria-hidden />}>
          <p className="font-mono text-xs">RP × 0.40 + EI × 0.35 + DS × 0.25</p>
          <p>Project-defined ratings with written rationale; never the model's confidence; unknown items are not scored.</p>
          <p className="font-mono text-xs">mean image score: {d?.green_score.mean_image_score != null ? d.green_score.mean_image_score.toFixed(1) : "—"} ({d?.images ?? 0} analyses)</p>
        </Block>
        <Block title="Organiser-defined categories" icon={<Tags className="size-4" aria-hidden />}>
          <p>The model's own classes are the organiser dataset labels (Organic / Recyclable). Profiles rename them for display:</p>
          <ul className="text-xs">{st.data?.label_profiles.profiles.map((p) => <li key={p.name}>• {p.title}</li>)}</ul>
        </Block>
        <Block title="Safety by design" icon={<ShieldCheck className="size-4" aria-hidden />}>
          <p>Low confidence or detector/classifier disagreement → REVIEW, shown as Unknown. Mixed-waste mode never moves a bin.</p>
          <p className="font-mono text-xs">validation accepted-only accuracy: {pi ? fmtPct(pi.accepted_only_accuracy, 2) : "—"} <ScopeTag>validation</ScopeTag></p>
        </Block>
      </div>

      <h2 className="mb-3 text-lg font-semibold tracking-tight">Model comparison and performance evidence</h2>
      <div className="mb-6"><ModelComparison /></div>
      <div className="mb-6 grid grid-cols-1 gap-6 xl:grid-cols-2">
        <DetectorEvalCard />
        <DetectorLatencyCard />
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Block title="Target users" icon={<Users className="size-4" aria-hidden />}>
          <ul className="list-disc space-y-1 pl-4">
            <li>Schools, campuses and offices running segregation stations</li>
            <li>Municipal dry-waste collection centres (supervisor review queue)</li>
            <li>Households and residential associations (Tamil / English guidance)</li>
          </ul>
        </Block>
        <Block title="Future scope">
          <ul className="list-disc space-y-1 pl-4">
            <li>Fine-tune the detector on box-annotated waste (TACO pipeline is ready: scripts/prepare_taco.py, train_detector.py)</li>
            <li>Independent TEST evaluation and labelled mixed-waste test images for end-to-end accuracy</li>
            <li>Residual / e-waste classes in the classifier; calibrated confidence</li>
            <li>Measured power with an inline USB-C power meter</li>
          </ul>
        </Block>
      </div>
    </>
  );
}
