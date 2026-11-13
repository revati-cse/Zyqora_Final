// UI test fixtures. These mock the HTTP API; they are NOT model results.
import { vi } from "vitest";
import type { Classification, HistoryPage, ModelStatus } from "@/types/api";

export const readyModel: ModelStatus = {
  ready: true, code: null, message: null, profile: "waste_mobilenetv2", expected_pte: "models/mobilenetv2_waste_fp32_xnnpack.pte",
  active_profile_in_config: "waste_mobilenetv2", available_profiles: ["imagenet_mobilenetv2", "waste_mobilenetv2"],
  description: "MobileNetV2 fine-tuned", decoder: "direct_classes", pte: "mobilenetv2_waste_fp32_xnnpack.pte", pte_size_bytes: 8880392,
  sha256: "a".repeat(64), sha256_pinned: null, precision: "fp32", input_shape: [1, 3, 224, 224], output_shape: [1, 2],
  preprocessing: { resize: [224, 224], center_crop: null, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] },
  classes: ["Organic", "Recyclable"], actions: { Organic: "COMPOST", Recyclable: "RECYCLE" }, unsupported_ground_truth: ["Residual"],
  runtime: "ExecuTorch", executorch_version: "1.3.1", registered_backends: ["XnnpackBackend"], xnnpack: true, threads: 4,
  fallback_used: false, failed_profiles: [], ecoguard: { threshold: 0.95, min_margin: 0, calibrated: false, frozen: false },
};

export const missingModel: ModelStatus = {
  ready: false, code: "MODEL_NOT_FOUND", message: "Model file not found: models/mobilenetv2_waste_fp32_xnnpack.pte",
  profile: "waste_mobilenetv2", expected_pte: "models/mobilenetv2_waste_fp32_xnnpack.pte",
  active_profile_in_config: "waste_mobilenetv2", available_profiles: ["waste_mobilenetv2"],
};

export function classification(over: Partial<Classification> = {}): Classification {
  return {
    id: 7, created_at: "2026-11-13T10:00:00.000+00:00", filename: "bottle.jpg", source: "upload", batch_id: null,
    predicted_category: "Recyclable", confidence: 0.991, class_scores: { Organic: 0.009, Recyclable: 0.991 },
    score_note: "softmax score of the model output; not a calibrated probability",
    ecoguard: { status: "ACCEPT", threshold: 0.95, reasons: [], explanation: { en: "", ta: "" } },
    disposal_action: "RECYCLE", guidance: { en: "Place this item in the recycling bin.", ta: "இந்தப் பொருளை மறுசுழற்சித் தொட்டியில் போடவும்." },
    sort: { action: "RECYCLE", actuate: true, message: "[SIMULATION] would open the RECYCLE bin", hardware: false },
    timing: { preprocess_ms: 4.2, inference_ms: 3.21, total_ms: 8.4 },
    model: { profile: "waste_mobilenetv2", description: "MobileNetV2", decoder: "direct_classes", pte: "x.pte", sha256: null,
      runtime: "ExecuTorch", backend: "XNNPACK", fallback_used: false, device: { device_model: "Raspberry Pi 5 Model B Rev 1.0", machine: "aarch64", is_raspberry_pi: true } },
    ground_truth: null, correct: null, greenscore_event_id: null, image_url: null, ...over,
  };
}

export const historyPage: HistoryPage = {
  total: 2, limit: 20, offset: 0,
  items: [
    { id: 2, created_at: "2026-11-13T10:01:00Z", source: "upload", filename: "leaf.jpg", batch_id: null, user_id: null, predicted_category: "Organic",
      confidence: 0.7, class_scores: { Organic: 0.7, Recyclable: 0.3 }, ecoguard_status: "REVIEW", review_reason: "LOW_CONFIDENCE",
      disposal_action: "MANUAL_REVIEW", preprocess_ms: 4, inference_ms: 3, total_ms: 8, model_profile: "waste_mobilenetv2", model_sha256: null,
      ground_truth: null, correct: null, greenscore_event_id: null, sort_message: "[SIMULATION] no bin opened", has_thumbnail: false, image_url: null },
    { id: 1, created_at: "2026-11-13T10:00:00Z", source: "upload", filename: "bottle.jpg", batch_id: null, user_id: null, predicted_category: "Recyclable",
      confidence: 0.99, class_scores: { Organic: 0.01, Recyclable: 0.99 }, ecoguard_status: "ACCEPT", review_reason: "",
      disposal_action: "RECYCLE", preprocess_ms: 4, inference_ms: 3, total_ms: 8, model_profile: "waste_mobilenetv2", model_sha256: null,
      ground_truth: null, correct: null, greenscore_event_id: null, sort_message: "[SIMULATION] would open", has_thumbnail: false, image_url: null },
  ],
};

type Handler = (url: URL, init?: RequestInit) => { status?: number; body: unknown } | Promise<{ status?: number; body: unknown }>;

/** Route fetch() calls to handlers keyed by "METHOD /path" (exact) or "/path" prefix. */
export function mockApi(routes: Record<string, Handler>) {
  const calls: Array<{ method: string; path: string; init?: RequestInit }> = [];
  const spy = vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(typeof input === "string" ? input : input.toString(), "http://localhost");
    const method = (init?.method ?? "GET").toUpperCase();
    calls.push({ method, path: url.pathname, init });
    const key = Object.keys(routes).find((k) => k === `${method} ${url.pathname}`) ?? Object.keys(routes).find((k) => url.pathname.startsWith(k));
    if (!key) return new Response(JSON.stringify({ error: { code: "NOT_FOUND", message: url.pathname } }), { status: 404 });
    const { status = 200, body } = await routes[key](url, init);
    return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
  });
  return { spy, calls };
}

export const baseRoutes = (model: ModelStatus = readyModel): Record<string, Handler> => ({
  "/api/health": () => ({ body: { status: "ok", version: "1.0.0", model_ready: model.ready, model_code: model.code, profile: model.profile,
    device_model: "Raspberry Pi 5 Model B Rev 1.0", machine: "aarch64", is_raspberry_pi: true, started_at: "", offline_note: "" } }),
  "/api/model/status": () => ({ body: model }),
});
