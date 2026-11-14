import { Award, BarChart3, Camera, Cpu, Gauge, History, Images, LayoutDashboard, Leaf, Presentation, ScanLine, ScanSearch, Settings } from "lucide-react";

export const NAV = [
  { to: "/", label: "Overview", icon: LayoutDashboard },
  { to: "/detect", label: "AI Detection", icon: ScanSearch },
  { to: "/camera", label: "Live Camera", icon: Camera },
  { to: "/classify", label: "Single-item Classification", icon: ScanLine },
  { to: "/batch", label: "Batch Analysis", icon: Images },
  { to: "/green-score", label: "Green Score", icon: Leaf },
  { to: "/analytics", label: "Analytics", icon: BarChart3 },
  { to: "/model", label: "Model Comparison", icon: Gauge },
  { to: "/history", label: "Detection History", icon: History },
  { to: "/device", label: "System Monitoring", icon: Cpu },
  { to: "/rewards", label: "Green Points", icon: Award },
  { to: "/showcase", label: "Hackathon Showcase", icon: Presentation },
  { to: "/settings", label: "Settings", icon: Settings },
] as const;
