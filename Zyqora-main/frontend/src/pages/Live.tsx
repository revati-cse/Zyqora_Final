import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Camera, Loader2, ScanLine, Video } from "lucide-react";
import { useEffect, useState } from "react";
import { ErrorState, PageHeader } from "@/components/common";
import { BrowserCamera, Dropzone, ImagePreview, type PickedImage } from "@/components/inference/ImageInput";
import { ResultCard } from "@/components/inference/ResultCard";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert } from "@/components/ui/misc";
import { useModelStatus } from "@/hooks/useApi";
import { api } from "@/lib/api";
import { t } from "@/lib/i18n";
import { useSettings } from "@/lib/settings";
import type { Classification } from "@/types/api";

export default function Live() {
  const { settings } = useSettings();
  const lang = settings.lang;
  const qc = useQueryClient();
  const model = useModelStatus();
  const [image, setImage] = useState<PickedImage | null>(null);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | undefined>();

  useEffect(() => {
    if (!image) return setPreviewUrl(undefined);
    const u = URL.createObjectURL(image.blob);
    setPreviewUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [image]);

  const onDone = () => {
    for (const k of ["analytics", "history", "greenscore", "model-metrics", "device"]) void qc.invalidateQueries({ queryKey: [k] });
  };
  const classify = useMutation<Classification, Error, PickedImage>({
    mutationFn: (img) => api.classify(img.blob, img.name, { source: img.source, userId: settings.userId || undefined }),
    onSuccess: onDone,
  });
  const piCamera = useMutation<Classification, Error, void>({
    mutationFn: () => api.cameraCapture(settings.userId || undefined),
    onSuccess: () => { setImage(null); onDone(); },
  });

  const ready = model.data?.ready === true;
  const busy = classify.isPending || piCamera.isPending;
  const result = piCamera.data && !image ? piCamera.data : classify.data;
  const error = classify.error ?? piCamera.error;

  const reset = () => {
    setImage(null);
    classify.reset();
    piCamera.reset();
  };

  return (
    <>
      <PageHeader
        title="Live Classification"
        description="Upload or capture one waste item. The image is sent to the Zyqora device, preprocessed exactly like the training data (224×224, ImageNet normalisation) and classified by the MobileNetV2 ExecuTorch model."
      />
      {model.data && !ready && (
        <Alert variant="danger" title="MODEL UNAVAILABLE" className="mb-5">
          <span className="font-mono text-xs">{model.data.code}</span> · {model.data.message}
          {model.data.expected_pte && <> Expected model file: <span className="font-mono text-xs">{model.data.expected_pte}</span>.</>}
          {" "}Classification is disabled; no substitute model or mock result is used.
        </Alert>
      )}
      {model.isError && <div className="mb-5"><ErrorState error={model.error} title="DEVICE DISCONNECTED" /></div>}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)]">
        <Card>
          <CardHeader>
            <CardTitle>Image</CardTitle>
            <CardDescription>
              {ready ? <>Model <span className="font-mono">{model.data?.profile}</span> · classes {model.data?.classes?.join(" / ")}</> : "Waiting for the model…"}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {cameraOpen ? (
              <BrowserCamera onCapture={(img) => { setImage(img); setCameraOpen(false); classify.reset(); }} onClose={() => setCameraOpen(false)} />
            ) : image ? (
              <ImagePreview image={image} onReset={reset} />
            ) : (
              <Dropzone disabled={busy} onPick={(img) => { setImage(img); classify.reset(); piCamera.reset(); }} />
            )}
            <div className="flex flex-wrap gap-2">
              <Button size="lg" className="flex-1 sm:flex-none" disabled={!image || !ready || busy} onClick={() => image && classify.mutate(image)}>
                {classify.isPending ? <Loader2 className="animate-spin" /> : <ScanLine />}
                <span className={lang === "ta" ? "font-tamil" : undefined}>{classify.isPending ? t("classifying", lang) : t("classify", lang)}</span>
              </Button>
              {!cameraOpen && (
                <Button size="lg" variant="outline" disabled={busy} onClick={() => { setCameraOpen(true); setImage(null); }}>
                  <Video />Browser camera
                </Button>
              )}
              <Button size="lg" variant="outline" disabled={!ready || busy} onClick={() => piCamera.mutate()}
                title="Capture one frame with the camera attached to the Zyqora device">
                {piCamera.isPending ? <Loader2 className="animate-spin" /> : <Camera />}Capture with device camera
              </Button>
            </div>
            {settings.userId && (
              <p className="text-xs text-muted-foreground">
                GreenScore participant <span className="font-mono">{settings.userId}</span>: a PENDING event is recorded; points only after supervisor verification.
              </p>
            )}
          </CardContent>
        </Card>

        <div aria-live="polite" className="min-w-0">
          {busy && (
            <Card className="flex min-h-64 items-center justify-center" role="status">
              <div className="flex flex-col items-center gap-3 text-sm text-muted-foreground">
                <Loader2 className="size-6 animate-spin" aria-hidden />
                <span className="font-mono text-xs tracking-wide">INFERENCE IN PROGRESS</span>
              </div>
            </Card>
          )}
          {!busy && error && <ErrorState error={error} title="Classification failed" />}
          {!busy && !error && result && <ResultCard result={result} previewUrl={result === classify.data ? previewUrl : undefined} />}
          {!busy && !error && !result && (
            <Card className="flex min-h-64 items-center justify-center p-6 text-center">
              <div className="max-w-sm space-y-2">
                <ScanLine className="mx-auto size-6 text-muted-foreground" aria-hidden />
                <p className="text-sm font-medium">No classification yet</p>
                <p className="text-xs text-muted-foreground">
                  The result will show the predicted category (Organic or Recyclable), the softmax score, the EcoGuard decision
                  and the inference time measured on the device.
                </p>
              </div>
            </Card>
          )}
        </div>
      </div>
    </>
  );
}
