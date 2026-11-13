import { Camera, ImageUp, RotateCcw, SwitchCamera, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState, type DragEvent } from "react";
import { Button } from "@/components/ui/button";
import { Alert } from "@/components/ui/misc";
import { cn } from "@/lib/utils";
import { ACCEPTED_TYPES, validateImageFile } from "@/lib/validation";

export interface PickedImage {
  blob: Blob;
  name: string;
  source: "upload" | "browser-camera";
}

/** Drag-and-drop / file picker with validation. */
export function Dropzone({ onPick, disabled, multiple = false, onMany }: {
  onPick?: (img: PickedImage) => void; onMany?: (files: File[], rejected: string[]) => void; disabled?: boolean; multiple?: boolean;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handle = useCallback((files: FileList | null) => {
    if (!files?.length) return;
    const list = Array.from(files);
    const rejected = list.map(validateImageFile).filter((e): e is string => !!e);
    const ok = list.filter((f) => !validateImageFile(f));
    if (multiple) {
      setError(rejected.length ? rejected.join(" ") : null);
      onMany?.(ok, rejected);
      return;
    }
    if (rejected.length) return setError(rejected[0]);
    setError(null);
    onPick?.({ blob: ok[0], name: ok[0].name, source: "upload" });
  }, [multiple, onMany, onPick]);

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setOver(false);
    if (!disabled) handle(e.dataTransfer.files);
  };

  return (
    <div>
      <button type="button" disabled={disabled} onClick={() => input.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)} onDrop={onDrop}
        className={cn("flex w-full flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed border-border bg-surface px-6 py-10 text-center transition-colors",
          over ? "border-primary bg-surface-2" : "hover:bg-surface-2", disabled && "cursor-not-allowed opacity-60")}
        aria-describedby="dropzone-help">
        <ImageUp className="size-7 text-muted-foreground" aria-hidden />
        <span className="text-sm font-medium">{multiple ? "Drop waste images here or choose files" : "Drop a waste image here or choose a file"}</span>
        <span id="dropzone-help" className="text-xs text-muted-foreground">JPEG, PNG or WebP · up to 10 MB{multiple ? " each" : ""}</span>
      </button>
      <input ref={input} type="file" accept={ACCEPTED_TYPES.join(",")} multiple={multiple} className="sr-only"
        aria-label={multiple ? "Choose image files" : "Choose an image file"} data-testid="file-input"
        onChange={(e) => { handle(e.target.files); e.target.value = ""; }} />
      {error && <Alert variant="danger" className="mt-3">{error}</Alert>}
    </div>
  );
}

/** Browser camera (getUserMedia). Needs a secure context: https, or http://localhost. */
export function BrowserCamera({ onCapture, onClose }: { onCapture: (img: PickedImage) => void; onClose: () => void }) {
  const video = useRef<HTMLVideoElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [facing, setFacing] = useState<"environment" | "user">("environment");
  const secure = window.isSecureContext;

  useEffect(() => {
    if (!secure || !navigator.mediaDevices?.getUserMedia) return;
    let stream: MediaStream | null = null;
    let cancelled = false;
    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: facing, width: { ideal: 1280 } }, audio: false })
      .then((s) => {
        if (cancelled) return s.getTracks().forEach((t) => t.stop());
        stream = s;
        if (video.current) {
          video.current.srcObject = s;
          void video.current.play();
        }
      })
      .catch((e: Error) => setError(e.name === "NotAllowedError" ? "Camera permission was denied." : `Camera unavailable: ${e.message}`));
    return () => {
      cancelled = true;
      stream?.getTracks().forEach((t) => t.stop()); // release the camera when leaving camera mode
    };
  }, [facing, secure]);

  if (!secure || !navigator.mediaDevices?.getUserMedia) {
    return (
      <Alert variant="review" title="Browser camera needs a secure connection">
        Browsers only allow camera access on https or http://localhost. On a laptop connected to the Pi over plain
        http, use “Capture with device camera” (the Pi's camera) or upload a photo instead.
        <div className="mt-2"><Button size="sm" variant="outline" onClick={onClose}>Close</Button></div>
      </Alert>
    );
  }

  const capture = () => {
    const v = video.current;
    if (!v || !v.videoWidth) return;
    const c = document.createElement("canvas");
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext("2d")?.drawImage(v, 0, 0);
    c.toBlob((b) => b && onCapture({ blob: b, name: `camera-${Date.now()}.jpg`, source: "browser-camera" }), "image/jpeg", 0.9);
  };

  return (
    <div className="space-y-3">
      {error ? <Alert variant="danger">{error}</Alert> : (
        <video ref={video} playsInline muted className="aspect-video w-full rounded-lg bg-black object-cover" aria-label="Camera preview" />
      )}
      <div className="flex flex-wrap gap-2">
        <Button onClick={capture} disabled={!!error}><Camera />Capture frame</Button>
        <Button variant="outline" onClick={() => setFacing((f) => (f === "user" ? "environment" : "user"))}><SwitchCamera />Switch camera</Button>
        <Button variant="ghost" onClick={onClose}><X />Close camera</Button>
      </div>
    </div>
  );
}

/** Preview of the picked image with object-URL lifecycle management. */
export function ImagePreview({ image, onReset }: { image: PickedImage; onReset: () => void }) {
  const [url, setUrl] = useState<string>("");
  useEffect(() => {
    const u = URL.createObjectURL(image.blob);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [image.blob]);
  return (
    <div className="space-y-2">
      <div className="overflow-hidden rounded-lg border border-border bg-surface-2">
        {url && <img src={url} alt={`Selected image ${image.name}`} className="mx-auto max-h-80 w-full object-contain" />}
      </div>
      <div className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
        <span className="truncate font-mono" title={image.name}>{image.name} · {(image.blob.size / 1024).toFixed(0)} KB</span>
        <Button size="sm" variant="ghost" onClick={onReset}><RotateCcw />Choose another</Button>
      </div>
    </div>
  );
}
