import { Camera as CameraIcon, CircleStop, Loader2, Play, ScanSearch } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { ErrorState, MetricCard, PageHeader, ScopeTag } from "@/components/common";
import { AnnotatedImage, ObjectCard } from "@/components/detection";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label, Select } from "@/components/ui/input";
import { Alert, Switch } from "@/components/ui/misc";
import { useModelsStatus } from "@/hooks/useApi";
import { api } from "@/lib/api";
import { useSettings } from "@/lib/settings";
import { fmtMs } from "@/lib/utils";
import type { DetectionResult } from "@/types/mixed";

const FRAME_MAX_SIDE = 960;

/** Grab the current video frame as a JPEG (downscaled; the detector letterboxes to 320/640 anyway). */
async function grab(video: HTMLVideoElement): Promise<Blob | null> {
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (!w || !h) return null;
  const s = Math.min(1, FRAME_MAX_SIDE / Math.max(w, h));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(w * s);
  canvas.height = Math.round(h * s);
  canvas.getContext("2d")!.drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise((res) => canvas.toBlob((b) => res(b), "image/jpeg", 0.85));
}

interface Frame { result: DetectionResult; url: string; roundTripMs: number; at: number }

export default function Camera() {
  const { settings } = useSettings();
  const status = useModelsStatus();
  const video = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [camError, setCamError] = useState<string | null>(null);
  const [frame, setFrame] = useState<Frame | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const [running, setRunning] = useState(false);
  const [intervalMs, setIntervalMs] = useState(1000);
  const [save, setSave] = useState(false);
  const [times, setTimes] = useState<number[]>([]);
  const stop = useRef(false);
  const secure = window.isSecureContext;

  useEffect(() => () => stream?.getTracks().forEach((t) => t.stop()), [stream]);
  useEffect(() => () => { if (frame) URL.revokeObjectURL(frame.url); }, [frame]);

  const start = async () => {
    setCamError(null);
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1280 } }, audio: false });
      setStream(s);
      if (video.current) { video.current.srcObject = s; await video.current.play(); }
    } catch (e) {
      setCamError((e as Error).message || "Camera permission denied");
    }
  };

  const analyse = useCallback(async (persist: boolean) => {
    if (!video.current) return;
    const blob = await grab(video.current);
    if (!blob) return;
    setBusy(true);
    const t0 = performance.now();
    try {
      const result = await api.detect(blob, `camera-${Date.now()}.jpg`, { source: "camera-frame", persist, userId: settings.userId || undefined });
      const now = performance.now();
      setFrame({ result, url: URL.createObjectURL(blob), roundTripMs: now - t0, at: now });
      setTimes((ts) => [...ts.slice(-9), now]);
      setError(null);
    } catch (e) {
      setError(e);
      stop.current = true;
      setRunning(false);
    } finally {
      setBusy(false);
    }
  }, [settings.userId]);

  // Continuous mode: one request at a time, then wait `intervalMs`, so the device is never queued up.
  useEffect(() => {
    if (!running) return;
    stop.current = false;
    let timer: number | undefined;
    const loop = async () => {
      if (stop.current) return;
      await analyse(save);
      if (!stop.current) timer = window.setTimeout(loop, intervalMs);
    };
    void loop();
    return () => { stop.current = true; window.clearTimeout(timer); };
  }, [running, intervalMs, save, analyse]);

  const fps = times.length > 1 ? ((times.length - 1) * 1000) / (times[times.length - 1] - times[0]) : null;
  const ready = status.data?.pipeline.ready;
  const r = frame?.result;

  return (
    <>
      <PageHeader title="Live Camera"
        description={<>Frames from this browser's camera are sent to the device and analysed with the same YOLO + MobileNetV2 pipeline. FPS and latency below are measured, not estimated. <ScopeTag>live</ScopeTag></>} />
      {!secure && <Alert variant="review" className="mb-4" title="CAMERA NEEDS A SECURE PAGE">Browsers only allow the camera on https:// or http://localhost. Open the dashboard on the Pi itself, or use AI Detection with a photo.</Alert>}
      {status.data && !ready && <Alert variant="danger" className="mb-4" title="PIPELINE UNAVAILABLE">{status.data.detector.code ?? status.data.classifier.code}</Alert>}
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <Card className="min-w-0">
          <CardHeader><CardTitle>Camera</CardTitle><CardDescription>Continuous frames are not saved to history unless you switch saving on.</CardDescription></CardHeader>
          <CardContent className="space-y-3">
            <video ref={video} playsInline muted className="aspect-video w-full rounded-lg border border-border bg-surface-2 object-contain" aria-label="Camera preview" />
            {camError && <Alert variant="danger">{camError}</Alert>}
            <div className="flex flex-wrap gap-2">
              {!stream ? <Button onClick={start} disabled={!secure}><CameraIcon />Start camera</Button> : (
                <>
                  <Button onClick={() => analyse(true)} disabled={!ready || busy || running}>{busy && !running ? <Loader2 className="animate-spin" /> : <ScanSearch />}Capture &amp; analyse</Button>
                  {!running
                    ? <Button variant="outline" onClick={() => { setTimes([]); setRunning(true); }} disabled={!ready}><Play />Start continuous</Button>
                    : <Button variant="outline" onClick={() => { stop.current = true; setRunning(false); }}><CircleStop />Stop</Button>}
                </>
              )}
            </div>
            <div className="flex flex-wrap items-end gap-4">
              <div>
                <Label htmlFor="interval">Pause between frames</Label>
                <Select id="interval" value={intervalMs} onChange={(e) => setIntervalMs(Number(e.target.value))} disabled={running}>
                  {[0, 250, 500, 1000, 2000, 5000].map((v) => <option key={v} value={v}>{v === 0 ? "none (as fast as possible)" : `${v} ms`}</option>)}
                </Select>
              </div>
              <label className="flex items-center gap-2 text-sm"><Switch checked={save} onCheckedChange={setSave} disabled={running} aria-label="Save continuous frames to history" />Save frames to history</label>
            </div>
          </CardContent>
        </Card>
        <div className="min-w-0 space-y-4">
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <MetricCard label="Measured FPS" value={fps != null && running ? fps.toFixed(2) : "—"} hint="last 10 frames, incl. pause" />
            <MetricCard label="Round trip" value={frame ? fmtMs(frame.roundTripMs, 0) : "—"} hint="browser → device → browser" />
            <MetricCard label="Pipeline on device" value={r ? fmtMs(r.timing.pipeline_ms, 0) : "—"} hint={r ? `YOLO ${fmtMs(r.timing.detector_inference_ms, 0)}` : undefined} />
            <MetricCard label="Waste items" value={r ? r.summary.waste_items : "—"} hint={r ? `${r.summary.detected_objects} objects` : undefined} />
          </div>
          {error != null && <ErrorState error={error} title="Frame analysis failed — continuous mode stopped" />}
          {frame && r ? (
            <Card className="min-w-0">
              <CardContent className="space-y-3 p-4">
                <AnnotatedImage src={frame.url} size={r.image_size} objects={r.objects} />
                <div className="grid gap-2 sm:grid-cols-2">
                  {r.objects.filter((o) => o.decision === "ACCEPT" || o.decision === "REVIEW").map((o) => <ObjectCard key={o.index} o={o} lang={settings.lang} />)}
                </div>
                {r.id && <p className="text-xs text-muted-foreground">Saved as analysis #{r.id}.</p>}
              </CardContent>
            </Card>
          ) : <div className="rounded-lg border border-dashed border-border p-8 text-center text-sm text-muted-foreground">Start the camera and capture a frame.</div>}
        </div>
      </div>
    </>
  );
}
