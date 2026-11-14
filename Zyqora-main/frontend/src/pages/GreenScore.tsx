import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Check, UserPlus } from "lucide-react";
import { Fragment, useState } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { GroupedBars } from "@/components/charts/Charts";
import { DecisionBadge, EmptyState, ErrorState, MetricCard, PageHeader } from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input, Label, Select } from "@/components/ui/input";
import { Alert, Skeleton, Table, Td, Th } from "@/components/ui/misc";
import { useGreenScore, useGreenScoreUser } from "@/hooks/useApi";
import { api, ApiError } from "@/lib/api";
import { useSettings } from "@/lib/settings";
import { fmtTime } from "@/lib/utils";
import type { GreenScoreEvent } from "@/types/api";

const userSchema = z.object({
  user_id: z.string().regex(/^[A-Za-z0-9_-]{1,32}$/, "1–32 letters, digits, _ or -"),
  display_name: z.string().max(64).optional(),
});

function AddUser() {
  const qc = useQueryClient();
  const form = useForm<z.infer<typeof userSchema>>({ resolver: zodResolver(userSchema), defaultValues: { user_id: "", display_name: "" } });
  const m = useMutation({
    mutationFn: (v: z.infer<typeof userSchema>) => api.addUser(v.user_id, v.display_name),
    onSuccess: () => { form.reset(); void qc.invalidateQueries({ queryKey: ["greenscore"] }); },
  });
  return (
    <form className="grid gap-2 sm:grid-cols-[1fr_1fr_auto] sm:items-end" onSubmit={form.handleSubmit((v) => m.mutate(v))}>
      <div><Label htmlFor="gs-uid">Participant id</Label><Input id="gs-uid" placeholder="e.g. TEAM01" {...form.register("user_id")} aria-invalid={!!form.formState.errors.user_id} />
        {form.formState.errors.user_id && <p className="mt-1 text-xs text-danger">{form.formState.errors.user_id.message}</p>}</div>
      <div><Label htmlFor="gs-name">Display name (optional)</Label><Input id="gs-name" {...form.register("display_name")} /></div>
      <Button type="submit" disabled={m.isPending}><UserPlus />Add</Button>
      {m.error && <p className="text-xs text-danger sm:col-span-3">{(m.error as ApiError).message}</p>}
    </form>
  );
}

const verifySchema = z.object({
  observed_action: z.string().min(2, "Choose the bin the item was placed in"),
  correct_action: z.string().optional(),
  supervisor_id: z.string().regex(/^[A-Za-z0-9_-]{0,32}$/, "letters, digits, _ or -").optional(),
});

function VerifyForm({ ev, bins, onDone }: { ev: GreenScoreEvent; bins: string[]; onDone: () => void }) {
  const review = ev.ecoguard_status === "REVIEW";
  const form = useForm<z.infer<typeof verifySchema>>({
    resolver: zodResolver(verifySchema),
    defaultValues: { observed_action: "", correct_action: review ? "" : ev.recommended_action, supervisor_id: "" },
  });
  const m = useMutation({
    mutationFn: (v: z.infer<typeof verifySchema>) =>
      api.verify({ event_id: ev.event_id, observed_action: v.observed_action, correct_action: v.correct_action || undefined, supervisor_id: v.supervisor_id || undefined }),
    onSuccess: onDone,
  });
  return (
    <form className="grid gap-2 rounded-md bg-surface-2 p-3 sm:grid-cols-4 sm:items-end"
      onSubmit={form.handleSubmit((v) => {
        if (review && !v.correct_action) return form.setError("correct_action", { message: "EcoGuard asked for review: state the correct bin" });
        m.mutate(v);
      })}>
      <div><Label htmlFor={`obs-${ev.event_id}`}>Bin item was placed in</Label>
        <Select id={`obs-${ev.event_id}`} {...form.register("observed_action")}><option value="">Choose…</option>{bins.map((b) => <option key={b}>{b}</option>)}</Select>
        {form.formState.errors.observed_action && <p className="mt-1 text-xs text-danger">{form.formState.errors.observed_action.message}</p>}</div>
      <div><Label htmlFor={`cor-${ev.event_id}`}>Correct bin {review ? "(required)" : "(AI recommendation)"}</Label>
        <Select id={`cor-${ev.event_id}`} {...form.register("correct_action")}><option value="">Choose…</option>{bins.map((b) => <option key={b}>{b}</option>)}</Select>
        {form.formState.errors.correct_action && <p className="mt-1 text-xs text-danger">{form.formState.errors.correct_action.message}</p>}</div>
      <div><Label htmlFor={`sup-${ev.event_id}`}>Supervisor id</Label><Input id={`sup-${ev.event_id}`} {...form.register("supervisor_id")} /></div>
      <Button type="submit" disabled={m.isPending}><Check />Verify</Button>
      {m.error && <p className="text-xs text-danger sm:col-span-4">{(m.error as ApiError).code}: {(m.error as ApiError).message}</p>}
    </form>
  );
}

export default function GreenScore() {
  const { settings } = useSettings();
  const qc = useQueryClient();
  const overview = useGreenScore();
  const [selected, setSelected] = useState("");
  const userId = selected || overview.data?.users[0]?.user_id || "";
  const user = useGreenScoreUser(userId);
  const [verifying, setVerifying] = useState<number | null>(null);
  const bins = overview.data?.bins ?? [];

  return (
    <>
      <PageHeader title="Green Points"
        description="Rewards for people (not the environmental Green Score). Rewards verified sorting, not AI predictions. A classification only creates a PENDING event; a supervisor checks the bin the item actually went into, and points are awarded once, only if it was the correct bin." />
      {overview.isError && <ErrorState error={overview.error} />}
      {overview.data && (
        <Alert className="mb-6">{overview.data.rule} Rule in this deployment: +{overview.data.points_verified_correct} for a verified correct bin, +{overview.data.points_verified_incorrect} otherwise.</Alert>
      )}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,320px)_1fr]">
        <Card>
          <CardHeader><CardTitle>Participants</CardTitle><CardDescription>Select one to see events</CardDescription></CardHeader>
          <CardContent className="space-y-4">
            <AddUser />
            {overview.isLoading ? <Skeleton className="h-32" /> : !overview.data?.users.length ? (
              <EmptyState title="No participants">Add one, then set it as the participant in Settings so new classifications create events.</EmptyState>
            ) : (
              <ul className="divide-y divide-border rounded-md border border-border">
                {overview.data.users.map((u) => (
                  <li key={u.user_id}>
                    <button type="button" onClick={() => setSelected(u.user_id)}
                      className={`flex w-full items-center justify-between px-3 py-2 text-left text-sm ${u.user_id === userId ? "bg-surface-2 font-medium" : "hover:bg-surface-2"}`}>
                      <span>{u.display_name ?? u.user_id}<span className="ml-2 font-mono text-xs text-muted-foreground">{u.user_id}</span></span>
                      <span className="font-mono tabular">{u.total_points}</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
            {settings.userId ? <p className="text-xs text-muted-foreground">Active participant for new classifications: <span className="font-mono">{settings.userId}</span></p>
              : <p className="text-xs text-muted-foreground">No active participant set (Settings).</p>}
          </CardContent>
        </Card>

        <div className="min-w-0 space-y-6">
          {!userId ? null : user.isError ? <ErrorState error={user.error} /> : !user.data ? <Skeleton className="h-64" /> : (
            <>
              <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                <MetricCard label="Verified points" value={user.data.total_points} />
                <MetricCard label="Pending" value={user.data.counts.PENDING ?? 0} />
                <MetricCard label="Verified correct" value={user.data.counts.VERIFIED_CORRECT ?? 0} />
                <MetricCard label="Verified incorrect" value={user.data.counts.VERIFIED_INCORRECT ?? 0} />
              </div>
              {Object.keys(user.data.contribution_by_bin).length > 0 && (
                <Card><CardHeader><CardTitle>Points by bin</CardTitle><CardDescription>From verified events only</CardDescription></CardHeader>
                  <CardContent><GroupedBars data={Object.entries(user.data.contribution_by_bin).map(([name, points]) => ({ name, points }))} keys={["points"]} /></CardContent></Card>
              )}
              <Card>
                <CardHeader><CardTitle>Events · {userId}</CardTitle></CardHeader>
                {!user.data.events.length ? <CardContent><EmptyState title="No events yet" /></CardContent> : (
                  <Table>
                    <thead><tr><Th>#</Th><Th>Item</Th><Th>EcoGuard</Th><Th>Recommended</Th><Th>Status</Th><Th className="text-right">Points</Th><Th><span className="sr-only">Actions</span></Th></tr></thead>
                    <tbody>
                      {user.data.events.map((e) => (
                        <Fragment key={e.event_id}>
                          <tr>
                            <Td className="font-mono text-xs">{e.event_id}</Td>
                            <Td className="max-w-40 truncate text-xs" title={e.image_id}>{e.image_id}<p className="text-[11px] text-muted-foreground">{fmtTime(e.created_at)}</p></Td>
                            <Td><DecisionBadge status={e.ecoguard_status} lang={settings.lang} /></Td>
                            <Td className="font-mono text-xs">{e.recommended_action}</Td>
                            <Td><Badge variant={e.verification_status === "PENDING" ? "neutral" : e.verification_status === "VERIFIED_CORRECT" ? "accept" : "danger"}>{e.verification_status}</Badge>
                              {e.observed_action && <p className="mt-0.5 font-mono text-[11px] text-muted-foreground">placed {e.observed_action} · correct {e.verified_action}</p>}</Td>
                            <Td className="text-right font-mono">{e.awarded_points}</Td>
                            <Td>{e.verification_status === "PENDING" && (
                              <Button size="sm" variant="outline" onClick={() => setVerifying(verifying === e.event_id ? null : e.event_id)}>Verify</Button>
                            )}</Td>
                          </tr>
                          {verifying === e.event_id && (
                            <tr><Td colSpan={7}>
                              <VerifyForm ev={e} bins={bins} onDone={() => { setVerifying(null); void qc.invalidateQueries({ queryKey: ["greenscore"] }); }} />
                            </Td></tr>
                          )}
                        </Fragment>
                      ))}
                    </tbody>
                  </Table>
                )}
              </Card>
            </>
          )}
        </div>
      </div>
    </>
  );
}
