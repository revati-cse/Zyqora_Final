import { AlertTriangle, CircleCheck, CircleHelp, Inbox, ShieldAlert } from "lucide-react";
import type { ReactNode } from "react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/misc";
import { ApiError } from "@/lib/api";
import { categoryLabel, t, type Lang } from "@/lib/i18n";
import { cn } from "@/lib/utils";

export function PageHeader({ title, description, actions }: { title: string; description?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
      <div className="min-w-0">
        <h1 className="text-xl font-semibold tracking-tight sm:text-2xl">{title}</h1>
        {description && <p className="mt-1 max-w-2xl text-sm text-muted-foreground">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap gap-2">{actions}</div>}
    </div>
  );
}

export function MetricCard({ label, value, hint, loading, mono = true }: {
  label: string; value: ReactNode; hint?: ReactNode; loading?: boolean; mono?: boolean;
}) {
  return (
    <Card>
      <CardContent className="p-4">
        <p className="text-xs text-muted-foreground">{label}</p>
        {loading ? <Skeleton className="mt-2 h-7 w-20" /> : (
          <p className={cn("mt-1 whitespace-nowrap text-xl font-semibold tracking-tight sm:text-2xl", mono && "font-mono tabular")}>{value}</p>
        )}
        {hint && <p className="mt-1 text-xs text-muted-foreground">{hint}</p>}
      </CardContent>
    </Card>
  );
}

export function EmptyState({ title, children, icon }: { title: string; children?: ReactNode; icon?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-border px-6 py-10 text-center">
      <div className="text-muted-foreground">{icon ?? <Inbox className="size-6" aria-hidden />}</div>
      <p className="text-sm font-medium">{title}</p>
      {children && <div className="max-w-md text-xs text-muted-foreground">{children}</div>}
    </div>
  );
}

export function ErrorState({ error, title = "Could not load data" }: { error: unknown; title?: string }) {
  const e = error instanceof ApiError ? error : null;
  return (
    <div role="alert" className="flex items-start gap-3 rounded-lg bg-danger-bg px-4 py-3 text-sm text-danger">
      <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden />
      <div className="min-w-0">
        <p className="font-medium">{title}</p>
        <p className="break-words opacity-90">
          {e ? <><span className="font-mono text-xs">{e.code}</span> · {e.message}</> : String((error as Error)?.message ?? error)}
        </p>
      </div>
    </div>
  );
}

export function DecisionBadge({ status, lang = "en" }: { status: string; lang?: Lang }) {
  return status === "ACCEPT" ? (
    <Badge variant="accept"><CircleCheck className="size-3.5" aria-hidden />{t("ACCEPT", lang)}</Badge>
  ) : (
    <Badge variant="review"><ShieldAlert className="size-3.5" aria-hidden />{t("REVIEW", lang)}</Badge>
  );
}

export function CategoryBadge({ category, lang = "en" }: { category: string | null; lang?: Lang }) {
  if (!category) return <Badge variant="outline"><CircleHelp className="size-3.5" aria-hidden />none</Badge>;
  return <Badge variant="outline" className={cn(lang === "ta" && "font-tamil")}>{categoryLabel(category, lang)}</Badge>;
}

/** Explicit provenance tag for every metric: VALIDATION / TEST / LIVE / HOST. */
export function ScopeTag({ children }: { children: ReactNode }) {
  return <span className="rounded border border-border px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wide text-muted-foreground">{children}</span>;
}
