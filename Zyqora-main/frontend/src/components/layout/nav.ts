import { BarChart3, Cpu, Gauge, History, Images, LayoutDashboard, Leaf, ScanLine, Settings } from "lucide-react";

export const NAV = [
  { to: "/", label: "Overview", icon: LayoutDashboard },
  { to: "/classify", label: "Live Classification", icon: ScanLine },
  { to: "/batch", label: "Batch Analysis", icon: Images },
  { to: "/analytics", label: "Analytics", icon: BarChart3 },
  { to: "/greenscore", label: "GreenScore", icon: Leaf },
  { to: "/model", label: "Model Performance", icon: Gauge },
  { to: "/device", label: "Device Monitoring", icon: Cpu },
  { to: "/history", label: "History", icon: History },
  { to: "/settings", label: "Settings", icon: Settings },
] as const;
