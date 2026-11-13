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
