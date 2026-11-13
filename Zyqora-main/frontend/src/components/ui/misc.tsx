import * as SwitchPrimitive from "@radix-ui/react-switch";
import * as TabsPrimitive from "@radix-ui/react-tabs";
import * as React from "react";
import { cn } from "@/lib/utils";

export const Skeleton = ({ className, ...p }: React.HTMLAttributes<HTMLDivElement>) => (
  <div aria-hidden className={cn("animate-pulse rounded-md bg-surface-2", className)} {...p} />
);

export const Separator = ({ className }: { className?: string }) => <div role="separator" className={cn("h-px w-full bg-border", className)} />;

export function Progress({ value, className, label }: { value: number; className?: string; label?: string }) {
  const v = Math.max(0, Math.min(100, value));
  return (
    <div role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(v)} aria-label={label}
      className={cn("h-2 w-full overflow-hidden rounded-full bg-surface-2", className)}>
      <div className="h-full rounded-full bg-primary transition-[width] duration-300" style={{ width: `${v}%` }} />
    </div>
  );
}

export function Alert({ variant = "neutral", title, children, className }: {
  variant?: "neutral" | "danger" | "review" | "accept"; title?: string; children?: React.ReactNode; className?: string;
}) {
  const styles = { neutral: "bg-info-bg", danger: "bg-danger-bg text-danger", review: "bg-review-bg text-review", accept: "bg-accept-bg text-accept" };
  return (
    <div role={variant === "danger" ? "alert" : "status"} className={cn("rounded-md px-4 py-3 text-sm", styles[variant], className)}>
      {title && <p className="font-medium">{title}</p>}
      {children && <div className={cn(title && "mt-1", "opacity-90")}>{children}</div>}
    </div>
  );
}

export const Switch = React.forwardRef<React.ElementRef<typeof SwitchPrimitive.Root>, React.ComponentPropsWithoutRef<typeof SwitchPrimitive.Root>>(
  ({ className, ...p }, ref) => (
    <SwitchPrimitive.Root ref={ref}
      className={cn("inline-flex h-5 w-9 shrink-0 cursor-pointer items-center rounded-full border border-border bg-surface-2 transition-colors data-[state=checked]:bg-primary", className)}
      {...p}>
      <SwitchPrimitive.Thumb className="block size-4 translate-x-0.5 rounded-full bg-surface shadow-sm transition-transform data-[state=checked]:translate-x-4" />
    </SwitchPrimitive.Root>
  ),
);
Switch.displayName = "Switch";

export const Tabs = TabsPrimitive.Root;
export const TabsList = ({ className, ...p }: React.ComponentPropsWithoutRef<typeof TabsPrimitive.List>) => (
  <TabsPrimitive.List className={cn("inline-flex rounded-md bg-surface-2 p-1", className)} {...p} />
);
export const TabsTrigger = ({ className, ...p }: React.ComponentPropsWithoutRef<typeof TabsPrimitive.Trigger>) => (
  <TabsPrimitive.Trigger
    className={cn("rounded px-3 py-1.5 text-xs font-medium text-muted-foreground data-[state=active]:bg-surface data-[state=active]:text-foreground data-[state=active]:shadow-sm", className)}
    {...p}
  />
);
export const TabsContent = TabsPrimitive.Content;

export function Table({ className, ...p }: React.TableHTMLAttributes<HTMLTableElement>) {
  return (
    <div className="relative w-full overflow-x-auto">
      <table className={cn("w-full text-sm", className)} {...p} />
    </div>
  );
}
export const Th = ({ className, ...p }: React.ThHTMLAttributes<HTMLTableCellElement>) => (
  <th className={cn("border-b border-border px-3 py-2 text-left text-xs font-medium text-muted-foreground", className)} {...p} />
);
export const Td = ({ className, ...p }: React.TdHTMLAttributes<HTMLTableCellElement>) => (
  <td className={cn("border-b border-border px-3 py-2 align-middle", className)} {...p} />
);
