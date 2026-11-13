import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { RefreshCw, Save } from "lucide-react";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { PageHeader } from "@/components/common";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input, Label, Select } from "@/components/ui/input";
import { Alert } from "@/components/ui/misc";
import { useModelStatus } from "@/hooks/useApi";
import { api, ApiError } from "@/lib/api";
import { useSettings, type Settings as S } from "@/lib/settings";

const schema = z.object({
  theme: z.enum(["light", "dark", "system"]),
  lang: z.enum(["en", "ta"]),
  apiBase: z.string().trim().refine((v) => v === "" || /^https?:\/\/[^\s/]+(:\d+)?\/?$/.test(v), "Use http://<host>:<port>, or leave empty"),
  pollSeconds: z.coerce.number().int().min(2, "minimum 2 s").max(30, "maximum 30 s"),
  userId: z.string().trim().regex(/^[A-Za-z0-9_-]{0,32}$/, "letters, digits, _ or -"),
  voice: z.enum(["off", "browser", "device"]),
  visual3d: z.boolean(),
});

export default function Settings() {
  const { settings, update } = useSettings();
  const qc = useQueryClient();
  const model = useModelStatus();
  const [saved, setSaved] = useState(false);
  const reload = useMutation({ mutationFn: api.reloadModel, onSuccess: () => void qc.invalidateQueries() });
  const form = useForm<z.input<typeof schema>, unknown, z.output<typeof schema>>({ resolver: zodResolver(schema), defaultValues: settings });
  const err = form.formState.errors;

  const onSubmit = (v: z.output<typeof schema>) => {
    update(v as S);
    setSaved(true);
    void qc.invalidateQueries();
    setTimeout(() => setSaved(false), 2500);
  };

  return (
    <>
      <PageHeader title="Settings" description="Preferences are stored in this browser. Model and EcoGuard settings live on the device (config/models.yaml) and are shown read-only." />
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader><CardTitle>Dashboard</CardTitle></CardHeader>
          <CardContent>
            <form className="space-y-4" onSubmit={form.handleSubmit(onSubmit)} noValidate>
              <div className="grid gap-4 sm:grid-cols-2">
                <div><Label htmlFor="s-theme">Theme</Label>
                  <Select id="s-theme" {...form.register("theme")}><option value="system">System</option><option value="light">Light</option><option value="dark">Dark</option></Select></div>
                <div><Label htmlFor="s-lang">Language</Label>
                  <Select id="s-lang" {...form.register("lang")}><option value="en">English</option><option value="ta">தமிழ் (Tamil)</option></Select></div>
                <div><Label htmlFor="s-voice">Voice instructions</Label>
                  <Select id="s-voice" {...form.register("voice")}>
                    <option value="browser">Play in this browser (offline clips)</option><option value="device">Play on the device speaker</option><option value="off">Off</option>
                  </Select></div>
                <div><Label htmlFor="s-poll">Telemetry interval (seconds)</Label>
                  <Input id="s-poll" type="number" min={2} max={30} {...form.register("pollSeconds")} aria-invalid={!!err.pollSeconds} />
                  {err.pollSeconds && <p className="mt-1 text-xs text-danger">{err.pollSeconds.message}</p>}</div>
              </div>
              <label className="flex items-center gap-2 text-sm">
                <input type="checkbox" className="size-4 accent-[var(--primary)]" {...form.register("visual3d")} />
                3D illustration on Overview (automatically off in the Pi's own browser and with reduced motion)
              </label>
              <div><Label htmlFor="s-user">GreenScore participant for new classifications</Label>
                <Input id="s-user" placeholder="empty = do not record events" {...form.register("userId")} aria-invalid={!!err.userId} />
                {err.userId && <p className="mt-1 text-xs text-danger">{err.userId.message}</p>}</div>
              <div><Label htmlFor="s-api">API address</Label>
                <Input id="s-api" placeholder="empty = this server (recommended)" {...form.register("apiBase")} aria-invalid={!!err.apiBase} />
                {err.apiBase && <p className="mt-1 text-xs text-danger">{err.apiBase.message}</p>}
                <p className="mt-1 text-xs text-muted-foreground">Leave empty when the dashboard is opened from the Pi (http://&lt;pi-ip&gt;:8000). A different address needs the Pi started with --cors-origin.</p></div>
              <div className="flex items-center gap-3">
                <Button type="submit"><Save />Save</Button>
                {saved && <span role="status" className="text-xs text-accept">Saved</span>}
              </div>
            </form>
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>Device configuration</CardTitle><CardDescription>Read-only, from the running API</CardDescription></CardHeader>
          <CardContent className="space-y-3 text-sm">
            {model.data ? (
              <dl className="grid grid-cols-2 gap-3">
                {[
                  ["Model profile", model.data.profile], ["Active in config", model.data.active_profile_in_config],
                  ["Model file", model.data.pte ?? model.data.expected_pte], ["Status", model.data.ready ? "ready" : model.data.code],
                  ["EcoGuard threshold", model.data.ecoguard?.threshold], ["Threshold calibrated", model.data.ecoguard ? String(model.data.ecoguard.calibrated) : "—"],
                  ["Threads", model.data.threads ?? "default"], ["Profiles", model.data.available_profiles.join(", ")],
                ].map(([k, v]) => <div key={String(k)} className="min-w-0"><dt className="text-xs text-muted-foreground">{k}</dt><dd className="truncate font-mono text-xs" title={String(v ?? "")}>{v ?? "—"}</dd></div>)}
              </dl>
            ) : <p className="text-muted-foreground">Not connected.</p>}
            <div className="flex flex-wrap items-center gap-2">
              <Button variant="outline" size="sm" disabled={reload.isPending} onClick={() => reload.mutate()}>
                <RefreshCw className={reload.isPending ? "animate-spin" : undefined} />Reload model on device
              </Button>
              {reload.data && <span role="status" className="text-xs">{reload.data.ready ? "Model ready" : `${reload.data.code}: ${reload.data.message}`}</span>}
              {reload.error && <span role="alert" className="text-xs text-danger">{(reload.error as ApiError).message}</span>}
            </div>
            <Alert>Reload after copying a new .pte into models/. To change the model profile or threshold, edit config/models.yaml on the device and restart the server.</Alert>
          </CardContent>
        </Card>
      </div>
    </>
  );
}
