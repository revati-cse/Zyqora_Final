import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Camera, Download, FileJson, ImageDown, Loader2, ScanSearch, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { ErrorState, PageHeader } from "@/components/common";
import { DetectionReport } from "@/components/detection/Report";
import { BrowserCamera, Dropzone, type PickedImage } from "@/components/inference/ImageInput";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, Skeleton } from "@/components/ui/misc";
import { useModelsStatus } from "@/hooks/useApi";
import { api, apiUrl } from "@/lib/api";
import { detectionRows } from "@/lib/detection";
import { useSettings } from "@/lib/settings";
import { downloadText, toCsv } from "@/lib/utils";

export default function Detect() {
  const { settings } = useSettings();
  const status = useModelsStatus();
  const qc = useQueryClient();
  const [img, setImg] = useState<PickedImage | null>(null);
  const [camera, setCamera] = useState(false);
  const preview = useMemo(() => (img ? URL.createObjectURL(img.blob) : null), [img]);
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  const m = useMutation({
    mutationFn: (p: PickedImage) => api.detect(p.blob, p.name, { userId: settings.userId || undefined, source: p.source }),
    onSuccess: () => { for (const k of ["analytics-all", "detections", "system-metrics"]) void qc.invalidateQueries({ queryKey: [k] }); },
  });
  const ready = status.data?.pipeline.ready;
  const r = m.data;

  return (
    <>
      <PageHeader title="AI Detection"
        description="Mixed-waste analysis on the device: YOLO finds each object, MobileNetV2 classifies every crop, then organiser labels, disposal rules and the Green Score are applied."
        actions={r && (
          <>
            <Button size="sm" variant="outline" onClick={() => downloadText(`zyqora_detection_${r.id ?? "unsaved"}.json`, JSON.stringify(r, null, 2), "application/json")}><FileJson />JSON</Button>
            <Button size="sm" variant="outline" onClick={() => downloadText(`zyqora_detection_${r.id ?? "unsaved"}.csv`, toCsv(detectionRows(r)), "text/csv")}><Download />CSV</Button>
            {r.annotated_url && <Button size="sm" variant="outline" asChild><a href={apiUrl(r.annotated_url)} download={`zyqora_detection_${r.id}.jpg`}><ImageDown />Annotated</a></Button>}
          </>
        )} />
      {status.isError && <div className="mb-4"><ErrorState error={status.error} title="DEVICE DISCONNECTED" /></div>}
      {status.data && !status.data.detector.ready && (
        <Alert variant="danger" className="mb-4" title="DETECTOR UNAVAILABLE">{status.data.detector.code}: {status.data.detector.message}</Alert>
      )}
      {status.data && !status.data.classifier.ready && (
        <Alert variant="danger" className="mb-4" title="CLASSIFIER UNAVAILABLE">{status.data.classifier.code}: {status.data.classifier.message}</Alert>
      )}
      {status.data?.detector.ready && (
        <p className="mb-4 text-xs text-muted-foreground">
          Detector <span className="font-mono">{status.data.detector.profile}</span> ({status.data.detector.trained_on}). Its class names are used only as material hints;
          waste categories come from the MobileNetV2 waste model (<span className="font-mono">{status.data.classifier.profile}</span>).
        </p>
      )}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,340px)_minmax(0,1fr)]">
        <Card className="self-start">
          <CardHeader><CardTitle>Image</CardTitle><CardDescription>Upload, drop or capture one photo with one or more waste items.</CardDescription></CardHeader>
          <CardContent className="space-y-3">
            {camera ? <BrowserCamera onCapture={(p) => { setImg(p); setCamera(false); m.reset(); }} onClose={() => setCamera(false)} /> : (
              <Dropzone disabled={m.isPending} onPick={(p) => { setImg(p); m.reset(); }} />
            )}
            {!camera && <Button variant="outline" className="w-full" onClick={() => setCamera(true)} disabled={m.isPending}><Camera />Use browser camera</Button>}
            {preview && !r && (
              <div className="relative">
                <img src={preview} alt="Selected image preview" className="w-full rounded-lg border border-border" />
                <Button size="icon" variant="ghost" className="absolute right-1 top-1 bg-surface/80" aria-label="Remove image" onClick={() => { setImg(null); m.reset(); }}><X /></Button>
              </div>
            )}
            <Button className="w-full" size="lg" disabled={!img || !ready || m.isPending} onClick={() => img && m.mutate(img)}>
              {m.isPending ? <><Loader2 className="animate-spin" />Analysing on device…</> : <><ScanSearch />Analyse waste</>}
            </Button>
            {img && <p className="truncate font-mono text-xs text-muted-foreground" title={img.name}>{img.name}</p>}
          </CardContent>
        </Card>
        <div className="min-w-0">
          {m.isError && <ErrorState error={m.error} title="Analysis failed — no result was produced" />}
          {m.isPending && <div className="space-y-3" aria-live="polite"><p className="text-sm font-medium">INFERENCE IN PROGRESS</p><Skeleton className="h-24" /><Skeleton className="h-80" /></div>}
          {!m.isPending && !r && !m.isError && (
            <div className="flex h-full min-h-60 items-center justify-center rounded-lg border border-dashed border-border p-8 text-center text-sm text-muted-foreground">
              Results appear here: bounding boxes, per-object labels, disposal guidance and the Green Score.
            </div>
          )}
          {r && preview && <DetectionReport r={r} imageSrc={preview} />}
        </div>
      </div>
    </>
  );
}
