import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { RotateCcw, Save } from "lucide-react";
import { useEffect } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { ErrorState } from "@/components/common";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input, Label, Select } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/misc";
import { useDetectionSettings } from "@/hooks/useApi";
import { api } from "@/lib/api";

const schema = z.object({
  detector_profile: z.string().min(1),
  label_profile: z.string().min(1),
  conf_threshold: z.coerce.number().gt(0, "> 0").lt(1, "< 1"),
  iou_threshold: z.coerce.number().gt(0, "> 0").lt(1, "< 1"),
  agnostic_nms: z.boolean(),
  max_detections: z.coerce.number().int().min(1).max(300),
  max_objects_classified: z.coerce.number().int().min(1).max(300),
  crop_padding: z.coerce.number().min(0).max(0.5),
  min_box_area_fraction: z.coerce.number().min(0).lt(0.5),
  whole_image_fallback: z.boolean(),
  retention_days: z.union([z.literal(""), z.coerce.number().int().min(1).max(3650)]),
  metrics_sample_seconds: z.coerce.number().int().min(5).max(3600),
});
type In = z.input<typeof schema>;
type Out = z.output<typeof schema>;

const NUM: Array<[keyof In, string, string]> = [
  ["conf_threshold", "Detector confidence threshold", "0.01"], ["iou_threshold", "NMS IoU threshold", "0.01"],
  ["max_detections", "Max boxes per image", "1"], ["max_objects_classified", "Max crops classified per image", "1"],
  ["crop_padding", "Crop padding (fraction)", "0.01"], ["min_box_area_fraction", "Min box area (fraction of image)", "0.001"],
  ["metrics_sample_seconds", "Store a system sample at most every (s)", "1"],
];

export function DetectionSettingsCard() {
  const q = useDetectionSettings();
  const qc = useQueryClient();
  const form = useForm<In, unknown, Out>({ resolver: zodResolver(schema) });
  const err = form.formState.errors;
  useEffect(() => {
    if (q.data) {
      const e = q.data.effective;
      form.reset({ ...e, retention_days: e.retention_days ?? "" } as In);
    }
  }, [q.data, form]);
  const save = useMutation({
    mutationFn: (v: Out) => api.putSettings({ ...v, retention_days: v.retention_days === "" ? null : v.retention_days }),
    onSuccess: (d) => { qc.setQueryData(["det-settings"], d); void qc.invalidateQueries({ queryKey: ["models-status"] }); },
  });
  const reset = useMutation({
    mutationFn: () => api.putSettings(Object.fromEntries(Object.keys(q.data?.overrides ?? {}).map((k) => [k, null]))),
    onSuccess: (d) => { qc.setQueryData(["det-settings"], d); void qc.invalidateQueries({ queryKey: ["models-status"] }); },
  });
  if (q.isLoading) return <Skeleton className="h-96" />;
  if (q.isError) return <ErrorState error={q.error} title="Detection settings unavailable" />;
  const o = q.data!.options;
  return (
    <Card>
      <CardHeader><CardTitle>Mixed-waste detection</CardTitle>
        <CardDescription>Stored on the device (SQLite settings table) and validated by the API. Defaults come from config/detector.yaml and config/label_profiles.yaml.</CardDescription></CardHeader>
      <CardContent>
        <form className="space-y-4" onSubmit={form.handleSubmit((v) => save.mutate(v))} noValidate>
          <div className="grid gap-4 sm:grid-cols-2">
            <div><Label htmlFor="ds-label">Organiser label profile</Label>
              <Select id="ds-label" {...form.register("label_profile")}>{o.label_profiles.map((p) => <option key={p.name} value={p.name}>{p.title}</option>)}</Select></div>
            <div><Label htmlFor="ds-det">Detector model</Label>
              <Select id="ds-det" {...form.register("detector_profile")}>
                {o.detector_profiles.map((p) => <option key={p.name} value={p.name} disabled={!p.available}>{p.name} · {p.imgsz}px · {p.runtime}{p.available ? "" : " (file missing)"}</option>)}
              </Select></div>
            {NUM.map(([k, label, step]) => (
              <div key={k}><Label htmlFor={`ds-${k}`}>{label}</Label>
                <Input id={`ds-${k}`} type="number" step={step} {...form.register(k)} aria-invalid={!!err[k]} />
                {err[k] && <p className="mt-1 text-xs text-danger">{String(err[k]?.message)}</p>}</div>
            ))}
            <div><Label htmlFor="ds-ret">Keep analyses for (days)</Label>
              <Input id="ds-ret" type="number" placeholder="empty = keep everything" {...form.register("retention_days")} aria-invalid={!!err.retention_days} />
              {err.retention_days && <p className="mt-1 text-xs text-danger">1–3650 days, or empty</p>}</div>
          </div>
          <label className="flex items-center gap-2 text-sm"><input type="checkbox" className="size-4 accent-[var(--primary)]" {...form.register("agnostic_nms")} />
            One box per physical object (class-agnostic NMS)</label>
          <label className="flex items-center gap-2 text-sm"><input type="checkbox" className="size-4 accent-[var(--primary)]" {...form.register("whole_image_fallback")} />
            Classify the whole image when no waste object is detected</label>
          <div className="flex flex-wrap items-center gap-3">
            <Button type="submit" disabled={save.isPending}><Save />Save on device</Button>
            <Button type="button" variant="outline" disabled={reset.isPending || !Object.keys(q.data!.overrides).length} onClick={() => reset.mutate()}><RotateCcw />Reset to defaults</Button>
            {save.isSuccess && <span role="status" className="text-xs text-accept">Saved</span>}
          </div>
          {(save.isError || reset.isError) && <ErrorState error={save.error ?? reset.error} title="Not saved" />}
          <p className="text-xs text-muted-foreground">Changing the detector model reloads it on the device. Thresholds apply to the next analysis; they were not tuned on waste data.</p>
        </form>
      </CardContent>
    </Card>
  );
}
