import { cn } from "@/lib/utils";

/** rows = actual, columns = predicted. `labels` order is preserved; extra columns (e.g. "<none>") allowed. */
export function ConfusionMatrix({ rows, cols, value, caption }: {
  rows: string[]; cols: string[]; value: (r: string, c: string) => number; caption?: string;
}) {
  const max = Math.max(1, ...rows.flatMap((r) => cols.map((c) => value(r, c))));
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[320px] border-separate border-spacing-1 text-sm">
        {caption && <caption className="mb-2 text-left text-xs text-muted-foreground">{caption}</caption>}
        <thead>
          <tr>
            <th className="text-left text-[11px] font-normal text-muted-foreground">actual ↓ / predicted →</th>
            {cols.map((c) => <th key={c} scope="col" className="px-2 text-center text-xs font-medium">{c}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r}>
              <th scope="row" className="pr-2 text-left text-xs font-medium">{r}</th>
              {cols.map((c) => {
                const v = value(r, c);
                const diag = r === c;
                return (
                  <td key={c} className={cn("rounded-md px-3 py-3 text-center font-mono tabular", diag ? "text-accept" : v ? "text-review" : "text-muted-foreground")}
                    style={{ background: v ? `color-mix(in oklab, var(${diag ? "--accept" : "--review"}) ${8 + (v / max) * 30}%, transparent)` : undefined }}>
                    {v}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
