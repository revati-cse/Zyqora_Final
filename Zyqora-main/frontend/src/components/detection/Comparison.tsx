import { GroupedBars } from "@/components/charts/Charts";
import { EmptyState, ScopeTag } from "@/components/common";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton, Table, Td, Th } from "@/components/ui/misc";
import { useAnalyticsAll, useBenchmarks, useModelMetrics, useModelsStatus } from "@/hooks/useApi";
import { fmtBytes, fmtMs, fmtPct } from "@/lib/utils";
import type { DetectorBench, DetectorEval } from "@/types/mixed";

const where = (p: Record<string, unknown> | null | undefined) =>
  p ? (p.is_raspberry_pi_5 ? "Raspberry Pi 5" : `${p.device_model ?? p.machine ?? "host"} (not the Pi)`) : "unknown device";

/** Detection evaluations on held-out VOC2007 test (generic objects, not waste). */
export function DetectorEvalCard() {
  const b = useBenchmarks();
  if (b.isLoading) return <Skeleton className="h-60" />;
  const evals = (b.data?.detector_evaluations ?? []).map((e) => e.data as unknown as DetectorEval);
  return (
    <Card className="min-w-0">
      <CardHeader>
        <CardTitle>Detector accuracy <ScopeTag>detection · held-out</ScopeTag></CardTitle>
        <CardDescription>mAP of the pretrained COCO detectors on PASCAL VOC2007 test (4,952 images never used for training). These are generic objects (bottle, tv, …), NOT waste categories; no waste-detection mAP exists yet because no box-annotated waste test set was available.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {!evals.length ? <EmptyState title="No detector evaluation recorded">Run scripts/evaluate_detector.py --voc …</EmptyState> : (
          <>
            <GroupedBars data={evals.map((e) => ({ name: `${e.profile.replace("_onnx", "")}`, "mAP@0.5": +e.map50.toFixed(3), "mAP@0.5:0.95": +e.map50_95.toFixed(3) }))}
              keys={["mAP@0.5", "mAP@0.5:0.95"]} />
            <Table>
              <thead><tr><Th>Profile</Th><Th className="text-right">mAP@0.5</Th><Th className="text-right">mAP@.5:.95</Th><Th className="text-right">bottle AP50</Th>
                <Th className="text-right">P / R</Th><Th className="hidden 2xl:table-cell">Runtime</Th></tr></thead>
              <tbody>
                {evals.map((e) => (
                  <tr key={e.profile}>
                    <Td className="font-mono text-xs">{e.profile}</Td>
                    <Td className="text-right font-mono tabular">{e.map50.toFixed(3)}</Td>
                    <Td className="text-right font-mono tabular">{e.map50_95.toFixed(3)}</Td>
                    <Td className="text-right font-mono tabular">{e.per_class.find((c) => c.label === "bottle")?.ap50?.toFixed(3) ?? "—"}</Td>
                    <Td className="whitespace-nowrap text-right font-mono text-xs tabular">{e.precision_at_conf.toFixed(2)} / {e.recall_at_conf.toFixed(2)}</Td>
                    <Td className="hidden text-xs 2xl:table-cell">{e.runtime} · {e.imgsz}px</Td>
                  </tr>
                ))}
              </tbody>
            </Table>
            <p className="text-xs text-muted-foreground">P / R at confidence {evals[0].operating_conf}. {evals[0].protocol}. Evaluated on {where(evals[0].platform)}; accuracy does not depend on the device.</p>
          </>
        )}
      </CardContent>
    </Card>
  );
}

/** Side-by-side: classifier alone vs detector vs combined pipeline, each metric labelled with its kind and source. */
export function ModelComparison() {
  const st = useModelsStatus();
  const mm = useModelMetrics();
  const b = useBenchmarks();
  const an = useAnalyticsAll();
  if (st.isLoading || mm.isLoading || b.isLoading) return <Skeleton className="h-72" />;
  const v = mm.data?.validation;
  const pi = v?.pi_reproduction;
  const clfBench = mm.data?.benchmark;
  const clfBest = clfBench?.results.slice().sort((a, b) => a.median_ms - b.median_ms)[0];
  const activeDet = st.data?.detector.profile;
  const evals = (b.data?.detector_evaluations ?? []).map((e) => e.data as unknown as DetectorEval);
  const ev = evals.find((e) => e.profile === activeDet) ?? evals.find((e) => e.profile === `${activeDet}_onnx`);
  const dbs = (b.data?.detector_benchmarks ?? []).map((e) => e.data as unknown as DetectorBench);
  const piDb = dbs.find((d) => d.profile === activeDet && d.platform.is_raspberry_pi_5) ?? null;
  const anyDb = piDb ?? dbs.find((d) => d.profile === activeDet) ?? null;
  const dr = anyDb?.results[0];
  const live = an.data?.detections;
  const derived = v ? Object.values(v.derived_from_confusion_matrix) : [];
  const macro = derived.length ? { p: derived.reduce((a, x) => a + x.precision, 0) / derived.length,
    r: derived.reduce((a, x) => a + x.recall, 0) / derived.length, f1: derived.reduce((a, x) => a + x.f1, 0) / derived.length } : null;
  const rows: Array<[string, string, string, string, string]> = [
    // metric, MobileNetV2, YOLO, combined, note
    ["Task", "classify one item (2 classes)", "find objects (80 COCO classes)", "per-object waste label + Green Score", ""],
    ["Accuracy (classification)", pi ? `${fmtPct(pi.accuracy, 2)} (validation, Pi)` : v ? `${fmtPct(v.accuracy, 2)} (validation)` : "—", "n/a", "not measured — needs labelled mixed-waste images", "classification metric"],
    ["Precision / Recall / F1 (macro)", macro ? `${macro.p.toFixed(3)} / ${macro.r.toFixed(3)} / ${macro.f1.toFixed(3)} (validation)` : "—", "n/a", "not measured", "classification metric"],
    ["Detection mAP@0.5", "n/a", ev ? `${ev.map50.toFixed(3)} (VOC07 test, generic)` : "not measured", "n/a", "detection metric"],
    ["Model size", st.data?.classifier.pte_size_bytes ? fmtBytes(st.data.classifier.pte_size_bytes) : "—",
      st.data?.detector.file_size_bytes ? fmtBytes(st.data.detector.file_size_bytes) : "—",
      st.data?.classifier.pte_size_bytes && st.data.detector.file_size_bytes ? fmtBytes(st.data.classifier.pte_size_bytes + st.data.detector.file_size_bytes) : "—", ""],
    ["Inference latency (median)", clfBest ? `${fmtMs(clfBest.median_ms, 2)} (${where(clfBench?.platform)})` : pi ? `${fmtMs(pi.inference_median_ms, 2)} (Pi)` : "—",
      dr ? `${fmtMs(dr.model_only.median_ms, 1)} (${where(anyDb?.platform)})` : "not benchmarked",
      dr?.pipeline ? `${fmtMs(dr.pipeline.pipeline_latency.median_ms, 0)} (${where(anyDb?.platform)})` : live?.latency.pipeline.median_ms != null ? `${fmtMs(live.latency.pipeline.median_ms, 0)} (live, this device)` : "—",
      "pipeline = YOLO + one MobileNetV2 run per crop"],
    ["CPU time / inference", clfBest?.cpu_ms_per_inference != null ? fmtMs(clfBest.cpu_ms_per_inference, 1) : "—", dr ? fmtMs(dr.cpu_ms_per_inference, 1) : "—",
      dr?.pipeline ? `${fmtMs(dr.pipeline.cpu_ms_per_image, 0)} / image` : "—", "process CPU, all threads"],
    ["Peak RAM (process)", clfBest?.peak_rss_mb != null ? `${clfBest.peak_rss_mb.toFixed(0)} MB` : "—", dr?.memory_mb.end.vm_hwm_mb != null ? `${dr.memory_mb.end.vm_hwm_mb.toFixed(0)} MB` : "—",
      dr?.pipeline?.memory_mb_end.vm_hwm_mb != null ? `${dr.pipeline.memory_mb_end.vm_hwm_mb.toFixed(0)} MB` : "—", "includes Python + torch runtime"],
    ["Throughput", clfBest?.throughput != null ? `${clfBest.throughput.toFixed(1)} img/s` : "—", dr ? `${dr.throughput_inferences_per_second.toFixed(1)} img/s` : "—",
      dr?.pipeline ? `${dr.pipeline.images_per_second.toFixed(2)} img/s` : "—", "single stream"],
  ];
  return (
    <Card className="min-w-0">
      <CardHeader>
        <CardTitle>Model comparison</CardTitle>
        <CardDescription>Classification and detection metrics are different measurements and are never compared directly. Every latency names the device it was measured on; “not the Pi” figures are development-machine numbers.</CardDescription>
      </CardHeader>
      <CardContent>
        <Table>
          <thead><tr><Th>Metric</Th><Th>MobileNetV2 (classifier)</Th><Th>YOLO ({activeDet ?? "—"})</Th><Th>YOLO + MobileNetV2</Th><Th className="hidden lg:table-cell">Note</Th></tr></thead>
          <tbody>
            {rows.map(([m, a, b2, c, n]) => (
              <tr key={m}><Td className="font-medium">{m}</Td><Td className="text-xs">{a}</Td><Td className="text-xs">{b2}</Td><Td className="text-xs">{c}</Td>
                <Td className="hidden text-xs text-muted-foreground lg:table-cell">{n}</Td></tr>
            ))}
          </tbody>
        </Table>
        {!piDb && <p className="mt-3 text-xs text-review">No Raspberry Pi 5 detector benchmark recorded yet: run <span className="font-mono">python scripts/benchmark_detector.py --images data/validation --pipeline</span> on the Pi.</p>}
      </CardContent>
    </Card>
  );
}

/** Detector latency per profile from recorded benchmarks. */
export function DetectorLatencyCard() {
  const b = useBenchmarks();
  const dbs = (b.data?.detector_benchmarks ?? []).map((e) => e.data as unknown as DetectorBench);
  return (
    <Card className="min-w-0">
      <CardHeader><CardTitle>Detector latency <ScopeTag>benchmark</ScopeTag></CardTitle>
        <CardDescription>scripts/benchmark_detector.py: model-only forward() median per profile and thread count.</CardDescription></CardHeader>
      <CardContent>
        {!dbs.length ? <EmptyState title="No detector benchmark recorded" /> : (
          <>
            <GroupedBars data={dbs.flatMap((d) => d.results.map((r) => ({ name: `${d.profile}·${r.threads_requested ?? "def"}t`, "median ms": +r.model_only.median_ms!.toFixed(1),
              "p95 ms": +r.model_only.p95_ms!.toFixed(1) })))} keys={["median ms", "p95 ms"]} />
            <p className="mt-2 text-xs text-muted-foreground">{[...new Set(dbs.map((d) => where(d.platform)))].join(", ")}</p>
          </>
        )}
      </CardContent>
    </Card>
  );
}
