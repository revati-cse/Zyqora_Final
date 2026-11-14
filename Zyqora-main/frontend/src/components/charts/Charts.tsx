import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { categoryColor, type Palette, usePalette } from "./palette";

const axis = (p: Palette) => ({ stroke: p["--muted-fg"], fontSize: 11, tickLine: false, axisLine: false });
const tooltip = (p: Palette) => ({
  contentStyle: { background: p["--surface"], border: `1px solid ${p["--border"]}`, borderRadius: 8, fontSize: 12, color: p["--fg"] },
  labelStyle: { color: p["--muted-fg"] },
});

export function CategoryDonut({ data, height = 220 }: { data: Record<string, number>; height?: number }) {
  const p = usePalette();
  const rows = Object.entries(data).map(([name, value]) => ({ name, value }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <PieChart>
        <Pie data={rows} dataKey="value" nameKey="name" innerRadius="58%" outerRadius="85%" paddingAngle={2} stroke="none" isAnimationActive={false}>
          {rows.map((r, i) => <Cell key={r.name} fill={categoryColor(p, r.name, i)} />)}
        </Pie>
        <Tooltip {...tooltip(p)} />
        <Legend verticalAlign="bottom" iconType="circle" wrapperStyle={{ fontSize: 12 }} />
      </PieChart>
    </ResponsiveContainer>
  );
}

export function VolumeChart({ data, categories, height = 240 }: { data: Array<Record<string, string | number>>; categories: string[]; height?: number }) {
  const p = usePalette();
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ left: -16, right: 8, top: 8 }}>
        <CartesianGrid vertical={false} stroke={p["--border"]} />
        <XAxis dataKey="date" {...axis(p)} />
        <YAxis allowDecimals={false} {...axis(p)} />
        <Tooltip {...tooltip(p)} cursor={{ fill: p["--border"], opacity: 0.4 }} />
        <Legend iconType="circle" wrapperStyle={{ fontSize: 12 }} />
        {categories.map((c, i) => <Bar key={c} dataKey={c} stackId="a" fill={categoryColor(p, c, i)} isAnimationActive={false} />)}
      </BarChart>
    </ResponsiveContainer>
  );
}

export function DecisionChart({ data, height = 240 }: { data: Array<{ date: string; accepted: number; review: number }>; height?: number }) {
  const p = usePalette();
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ left: -16, right: 8, top: 8 }}>
        <CartesianGrid vertical={false} stroke={p["--border"]} />
        <XAxis dataKey="date" {...axis(p)} />
        <YAxis allowDecimals={false} {...axis(p)} />
        <Tooltip {...tooltip(p)} cursor={{ fill: p["--border"], opacity: 0.4 }} />
        <Legend iconType="circle" wrapperStyle={{ fontSize: 12 }} />
        <Bar dataKey="accepted" name="ACCEPT" stackId="d" fill={p["--accept"]} isAnimationActive={false} />
        <Bar dataKey="review" name="REVIEW" stackId="d" fill={p["--review"]} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function LatencyChart({ data, height = 240 }: { data: Array<{ id: number; inference_ms: number; total_ms: number }>; height?: number }) {
  const p = usePalette();
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ left: -8, right: 8, top: 8 }}>
        <CartesianGrid vertical={false} stroke={p["--border"]} />
        <XAxis dataKey="id" {...axis(p)} label={{ value: "classification #", position: "insideBottomRight", offset: -2, fontSize: 11, fill: p["--muted-fg"] }} />
        <YAxis unit=" ms" {...axis(p)} width={64} />
        <Tooltip {...tooltip(p)} formatter={(v) => `${Number(v).toFixed(2)} ms`} />
        <Legend iconType="circle" wrapperStyle={{ fontSize: 12 }} />
        <Line type="monotone" dataKey="inference_ms" name="model inference" stroke={p["--chart-1"]} dot={false} strokeWidth={2} isAnimationActive={false} />
        <Line type="monotone" dataKey="total_ms" name="total on device" stroke={p["--chart-3"]} dot={false} strokeWidth={1.5} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function Sparkline({ data, dataKey, unit, height = 64, domain }: {
  data: Array<Record<string, number | null>>; dataKey: string; unit?: string; height?: number; domain?: [number, number];
}) {
  const p = usePalette();
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ left: 0, right: 0, top: 4, bottom: 0 }}>
        <YAxis hide domain={domain ?? ["auto", "auto"]} />
        <Tooltip {...tooltip(p)} formatter={(v) => `${Number(v).toFixed(1)}${unit ?? ""}`} labelFormatter={() => ""} />
        <Area type="monotone" dataKey={dataKey} stroke={p["--chart-1"]} fill={p["--chart-1"]} fillOpacity={0.12} strokeWidth={1.5} isAnimationActive={false} connectNulls />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function GroupedBars({ data, keys, height = 220 }: { data: Array<Record<string, string | number>>; keys: string[]; height?: number }) {
  const p = usePalette();
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ left: -16, right: 8, top: 8 }}>
        <CartesianGrid vertical={false} stroke={p["--border"]} />
        <XAxis dataKey="name" {...axis(p)} />
        <YAxis {...axis(p)} />
        <Tooltip {...tooltip(p)} cursor={{ fill: p["--border"], opacity: 0.4 }} />
        <Legend iconType="circle" wrapperStyle={{ fontSize: 12 }} />
        {keys.map((k, i) => <Bar key={k} dataKey={k} fill={[p["--chart-1"], p["--chart-2"], p["--chart-3"]][i % 3]} isAnimationActive={false} />)}
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Generic multi-line chart over an x key (time series of real records only). */
export function Lines({ data, x, lines, unit, height = 220, domain }: {
  data: Array<Record<string, string | number | null>>; x: string; lines: Array<{ key: string; name: string }>; unit?: string;
  height?: number; domain?: [number, number];
}) {
  const p = usePalette();
  const colors = [p["--chart-1"], p["--chart-3"], p["--chart-2"], p["--chart-4"]];
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ left: -8, right: 8, top: 8 }}>
        <CartesianGrid vertical={false} stroke={p["--border"]} />
        <XAxis dataKey={x} {...axis(p)} minTickGap={24} />
        <YAxis unit={unit} {...axis(p)} width={56} domain={domain ?? ["auto", "auto"]} />
        <Tooltip {...tooltip(p)} formatter={(v) => (v == null ? "—" : `${Number(v).toFixed(1)}${unit ?? ""}`)} />
        {lines.length > 1 && <Legend iconType="circle" wrapperStyle={{ fontSize: 12 }} />}
        {lines.map((l, i) => <Line key={l.key} type="monotone" dataKey={l.key} name={l.name} stroke={colors[i % 4]} dot={data.length < 30}
          strokeWidth={i ? 1.5 : 2} isAnimationActive={false} connectNulls />)}
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Horizontal bars for a {label: count} mapping. */
export function CountBars({ data, height }: { data: Record<string, number>; height?: number }) {
  const p = usePalette();
  const rows = Object.entries(data).sort((a, b) => b[1] - a[1]).map(([name, value]) => ({ name, value }));
  return (
    <ResponsiveContainer width="100%" height={height ?? Math.max(120, rows.length * 32 + 24)}>
      <BarChart data={rows} layout="vertical" margin={{ left: 8, right: 16, top: 4, bottom: 4 }}>
        <CartesianGrid horizontal={false} stroke={p["--border"]} />
        <XAxis type="number" allowDecimals={false} {...axis(p)} />
        <YAxis type="category" dataKey="name" {...axis(p)} width={110} />
        <Tooltip {...tooltip(p)} cursor={{ fill: p["--border"], opacity: 0.4 }} />
        <Bar dataKey="value" name="count" isAnimationActive={false} radius={[0, 3, 3, 0]}>
          {rows.map((r, i) => <Cell key={r.name} fill={r.name === "Unknown" ? p["--muted-fg"] : categoryColor(p, r.name, i)} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
