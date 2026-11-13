import { useEffect, useState } from "react";

const VARS = ["--chart-1", "--chart-2", "--chart-3", "--chart-4", "--accept", "--review", "--border", "--muted-fg", "--surface", "--fg"] as const;
export type Palette = Record<(typeof VARS)[number], string>;

function readPalette(): Palette {
  const cs = getComputedStyle(document.documentElement);
  return Object.fromEntries(VARS.map((v) => [v, cs.getPropertyValue(v).trim() || "#888"])) as Palette;
}

/** Chart colours follow the light/dark theme tokens. */
export function usePalette(): Palette {
  const [p, setP] = useState<Palette>(readPalette);
  useEffect(() => {
    const obs = new MutationObserver(() => setP(readPalette()));
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => obs.disconnect();
  }, []);
  return p;
}

export function categoryColor(p: Palette, name: string, i: number) {
  if (name === "Organic") return p["--chart-2"];
  if (name === "Recyclable") return p["--chart-1"];
  return [p["--chart-3"], p["--chart-4"]][i % 2];
}

