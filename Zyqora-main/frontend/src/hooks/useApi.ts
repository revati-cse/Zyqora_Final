import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useSettings } from "@/lib/settings";

// Status polling is deliberately slow; device telemetry uses the user-configured interval (2–30 s).
export const useHealth = () => useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 10_000, retry: 1 });
export const useModelStatus = () =>
  useQuery({ queryKey: ["model-status"], queryFn: api.modelStatus, refetchInterval: 30_000, retry: 1 });
export const useModelMetrics = () => useQuery({ queryKey: ["model-metrics"], queryFn: api.modelMetrics });
export const useAnalytics = (p: { date_from?: string; date_to?: string; category?: string } = {}) =>
  useQuery({ queryKey: ["analytics", p], queryFn: () => api.analytics(p) });
export const useHistory = (p: Parameters<typeof api.history>[0]) =>
  useQuery({ queryKey: ["history", p], queryFn: () => api.history(p), placeholderData: (prev) => prev });
export const useGreenScore = () => useQuery({ queryKey: ["greenscore"], queryFn: api.greenscore });
export const useGreenScoreUser = (userId: string) =>
  useQuery({ queryKey: ["greenscore", userId], queryFn: () => api.greenscoreUser(userId), enabled: !!userId });
export const useVoiceMessages = () => useQuery({ queryKey: ["voice"], queryFn: api.voiceMessages, staleTime: Infinity });

export function useDevice(enabled = true) {
  const { settings } = useSettings();
  return useQuery({
    queryKey: ["device"],
    queryFn: api.device,
    enabled,
    refetchInterval: settings.pollSeconds * 1000,
    refetchIntervalInBackground: false, // no polling while the tab is hidden
  });
}

// ---- Zyqora AI 4.0
export const useModelsStatus = () =>
  useQuery({ queryKey: ["models-status"], queryFn: api.modelsStatus, refetchInterval: 30_000, retry: 1 });
export const useAnalyticsAll = (days?: number) => useQuery({ queryKey: ["analytics-all", days], queryFn: () => api.analyticsAll(days) });
export const useDetections = (p: Parameters<typeof api.detections>[0]) =>
  useQuery({ queryKey: ["detections", p], queryFn: () => api.detections(p), placeholderData: (prev) => prev });
export const useDetection = (id: number | null) =>
  useQuery({ queryKey: ["detection", id], queryFn: () => api.detection(id!), enabled: id != null });
export const useGreenScoreRules = () => useQuery({ queryKey: ["gs-rules"], queryFn: api.greenscoreRules, staleTime: 300_000 });
export const useBenchmarks = () => useQuery({ queryKey: ["benchmarks"], queryFn: api.benchmarks, staleTime: 60_000 });
export const useDetectionSettings = () => useQuery({ queryKey: ["det-settings"], queryFn: api.getSettings });

export function useSystemMetrics(enabled = true) {
  const { settings } = useSettings();
  return useQuery({
    queryKey: ["system-metrics"],
    queryFn: api.systemMetrics,
    enabled,
    refetchInterval: settings.pollSeconds * 1000,
    refetchIntervalInBackground: false,
  });
}
