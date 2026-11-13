/** Static SVG used when WebGL is unavailable or motion is reduced on low-end devices. */
export function BinIllustration() {
  return (
    <svg viewBox="0 0 240 160" className="h-full w-full" aria-hidden>
      <rect x="20" y="132" width="200" height="6" rx="3" className="fill-border" />
      <path d="M48 52h56l-6 80H54z" className="fill-chart-2" />
      <rect x="42" y="42" width="68" height="10" rx="3" className="fill-chart-2" opacity="0.8" />
      <path d="M136 52h56l-6 80h-44z" className="fill-chart-1" />
      <rect x="130" y="42" width="68" height="10" rx="3" className="fill-chart-1" opacity="0.8" />
    </svg>
  );
}
