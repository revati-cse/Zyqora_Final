import type {
  AnalyticsSummary,
  BatchResponse,
  Classification,
  DeviceMetrics,
  GreenScoreOverview,
  GreenScoreUser,
  Health,
  HistoryPage,
  ModelMetrics,
  ModelStatus,
  VerifyResult,
  VoiceMessages,
} from "@/types/api";

/** Error carrying the backend's stable error code (e.g. MODEL_NOT_FOUND). */
export class ApiError extends Error {
  readonly code: string;
  readonly status: number;
  readonly details: Record<string, unknown>;
  constructor(code: string, message: string, status: number, details: Record<string, unknown> = {}) {
    super(message);
    this.code = code;
    this.status = status;
    this.details = details;
  }
}

let apiBase = "";
/** "" = same origin (production build served by the Pi). Set to http://<pi>:8000 for a remote API. */
export function setApiBase(base: string) {
  apiBase = base.replace(/\/+$/, "");
}
export function apiUrl(path: string) {
  return `${apiBase}${path}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(apiUrl(path), init);
  } catch (err) {
    throw new ApiError("DEVICE_UNREACHABLE", `Cannot reach the Zyqora device (${(err as Error).message})`, 0);
  }
  const text = await res.text();
  let body: unknown = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = null;
  }
  if (!res.ok) {
    const e = (body as { error?: { code?: string; message?: string; details?: Record<string, unknown> } } | null)?.error;
    throw new ApiError(e?.code ?? `HTTP_${res.status}`, e?.message ?? (res.statusText || "Request failed"), res.status, e?.details);
  }
  return body as T;
}

function qs(params: Record<string, string | number | undefined | null>) {
  const s = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== "") s.set(k, String(v));
  const q = s.toString();
  return q ? `?${q}` : "";
}

export interface ClassifyOptions {
  userId?: string;
  batchId?: string;
  groundTruth?: string;
  source?: "upload" | "browser-camera" | "batch";
  signal?: AbortSignal;
}

export const api = {
  health: () => request<Health>("/api/health"),
  modelStatus: () => request<ModelStatus>("/api/model/status"),
  reloadModel: () => request<ModelStatus>("/api/model/reload", { method: "POST" }),
  modelMetrics: () => request<ModelMetrics>("/api/model/metrics"),
  classify: (file: Blob, filename: string, o: ClassifyOptions = {}) => {
    const fd = new FormData();
    fd.append("file", file, filename);
    if (o.userId) fd.append("user_id", o.userId);
    if (o.batchId) fd.append("batch_id", o.batchId);
    if (o.groundTruth) fd.append("ground_truth", o.groundTruth);
    fd.append("source", o.source ?? "upload");
    return request<Classification>("/api/inference", { method: "POST", body: fd, signal: o.signal });
  },
  cameraCapture: (userId?: string) => {
    const fd = new FormData();
    if (userId) fd.append("user_id", userId);
    return request<Classification>("/api/camera/capture", { method: "POST", body: fd });
  },
  batch: (batchId: string) => request<BatchResponse>(`/api/batches/${encodeURIComponent(batchId)}`),
  history: (p: { limit?: number; offset?: number; category?: string; status?: string; date_from?: string; date_to?: string; batch_id?: string }) =>
    request<HistoryPage>(`/api/history${qs(p)}`),
  analytics: (p: { date_from?: string; date_to?: string; category?: string } = {}) =>
    request<AnalyticsSummary>(`/api/analytics/summary${qs(p)}`),
  historyCsvUrl: (p: { date_from?: string; date_to?: string; category?: string; status?: string } = {}) =>
    apiUrl(`/api/history/export.csv${qs(p)}`),
  device: () => request<DeviceMetrics>("/api/device/metrics"),
  greenscore: () => request<GreenScoreOverview>("/api/greenscore"),
  greenscoreUser: (userId: string) => request<GreenScoreUser>(`/api/greenscore/${encodeURIComponent(userId)}`),
  addUser: (user_id: string, display_name?: string) =>
    request<{ user_id: string }>("/api/greenscore/users", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id, display_name: display_name || null }),
    }),
  verify: (body: { event_id: number; observed_action: string; correct_action?: string; supervisor_id?: string }) =>
    request<VerifyResult>("/api/greenscore/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  voiceMessages: () => request<VoiceMessages>("/api/voice/messages"),
  voiceClipUrl: (lang: "en" | "ta", key: string) => apiUrl(`/api/voice/${lang}/${encodeURIComponent(key)}.wav`),
  playOnDevice: (lang: "en" | "ta", key: string) =>
    request<{ played: boolean; detail: string; text: string; player: string | null }>("/api/voice/play", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lang, key }),
    }),
};
