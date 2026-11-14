// Types mirror src/zyqora/api/schemas.py and service.py responses.

export interface ApiErrorBody {
  error: { code: string; message: string; details?: Record<string, unknown> };
}

export interface Health {
  status: string;
  version: string;
  model_ready: boolean;
  model_code: string | null;
  profile: string | null;
  device_model: string | null;
  machine: string;
  is_raspberry_pi: boolean;
  started_at: string;
  offline_note: string;
}

export interface ModelStatus {
  ready: boolean;
  code: string | null;
  message: string | null;
  profile: string | null;
  expected_pte: string | null;
  active_profile_in_config: string;
  available_profiles: string[];
  description?: string;
  decoder?: string;
  pte?: string;
  pte_size_bytes?: number;
  sha256?: string;
  sha256_pinned?: string | null;
  precision?: string;
  input_shape?: number[];
  output_shape?: number[] | null;
  preprocessing?: { resize: number | number[]; center_crop: number | null; mean: number[]; std: number[] };
  classes?: string[];
  actions?: Record<string, string>;
  unsupported_ground_truth?: string[];
  runtime?: string;
  executorch_version?: string;
  registered_backends?: string[];
  xnnpack?: boolean;
  threads?: number | null;
  thread_control?: string;
  fallback_used?: boolean;
  failed_profiles?: string[];
  ecoguard?: { threshold: number; min_margin: number; calibrated: boolean; frozen: boolean };
  loaded_at?: string;
  load_ms?: number;
}

export interface Classification {
  id: number | null;
  created_at: string;
  filename: string;
  source: string;
  batch_id: string | null;
  predicted_category: string | null;
  confidence: number | null;
  class_scores: Record<string, number>;
  score_note: string;
  ecoguard: { status: "ACCEPT" | "REVIEW"; threshold: number; reasons: string[]; explanation: { en: string; ta: string } };
  disposal_action: string;
  guidance: { en: string; ta: string };
  sort: { action: string; actuate: boolean; message: string; hardware: boolean };
  timing: { preprocess_ms: number | null; inference_ms: number | null; total_ms: number | null };
  model: {
    profile: string;
    description: string;
    decoder: string;
    pte: string;
    sha256: string | null;
    runtime: string;
    backend: string;
    fallback_used: boolean;
    device: { device_model: string | null; machine: string; is_raspberry_pi: boolean };
  };
  ground_truth: string | null;
  correct: boolean | null;
  greenscore_event_id: number | null;
  image_url: string | null;
  capture_ms?: number | null;
}

export interface HistoryItem {
  id: number;
  created_at: string;
  source: string;
  filename: string;
  batch_id: string | null;
  user_id: string | null;
  predicted_category: string | null;
  confidence: number | null;
  class_scores: Record<string, number>;
  ecoguard_status: "ACCEPT" | "REVIEW";
  review_reason: string | null;
  disposal_action: string;
  preprocess_ms: number | null;
  inference_ms: number | null;
  total_ms: number | null;
  model_profile: string;
  model_sha256: string | null;
  ground_truth: string | null;
  correct: boolean | null;
  greenscore_event_id: number | null;
  sort_message: string | null;
  has_thumbnail: boolean;
  image_url: string | null;
}

export interface HistoryPage {
  items: HistoryItem[];
  total: number;
  limit: number;
  offset: number;
}

export interface LatencyStats {
  mean_ms: number | null;
  median_ms: number | null;
  p95_ms: number | null;
  min_ms: number | null;
  max_ms: number | null;
  count: number;
}

export interface PerClassMetric {
  support: number;
  predicted: number;
  true_positive: number;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  predictable: boolean;
}

export interface ClassificationMetrics {
  confusion_matrix: { rows: string; columns: string; matrix: Record<string, Record<string, number>> };
  per_class: Record<string, PerClassMetric>;
  macro_avg_over_classes_with_support: { precision: number | null; recall: number | null; f1: number | null };
}

export interface LabelledAccuracy {
  scope: string;
  note: string;
  labelled_images: number;
  correct: number;
  accuracy: number;
  accepted: number;
  accepted_only_accuracy: number | null;
  metrics: ClassificationMetrics;
}

export interface AnalyticsSummary {
  total: number;
  accepted: number;
  review: number;
  review_rate: number | null;
  by_category: Record<string, number>;
  by_action: Record<string, number>;
  by_day: Array<{ date: string; total: number; accepted: number; review: number; [k: string]: number | string }>;
  latest: {
    id: number;
    created_at: string;
    predicted_category: string | null;
    confidence: number | null;
    ecoguard_status: string;
    disposal_action: string;
    inference_ms: number | null;
  } | null;
  inference_latency: LatencyStats;
  total_latency: LatencyStats;
  latency_series: Array<{ id: number; created_at: string; inference_ms: number; total_ms: number }>;
  batch_throughput: Array<{ batch_id: string; images: number; seconds: number; images_per_second: number }>;
  labelled_accuracy: LabelledAccuracy | null;
}

export interface DeviceMetrics {
  device_model: string | null;
  is_raspberry_pi: boolean;
  cpu_percent: number | null;
  cpu_count: number | null;
  load_average: number[] | null;
  memory: { total_mb: number | null; available_mb: number | null; used_mb: number | null; used_percent: number | null };
  temperature_c: number | null;
  uptime_s: number | null;
  process_rss_mb: number | null;
  process_peak_rss_mb: number | null;
  sampled_at: number;
  model_ready: boolean;
  model_profile: string | null;
  runtime: string | null;
  backend: string | null;
  threads: number | null;
  last_inference_ms: number | null;
  camera: { configured_backend: string; open: boolean; simulated: boolean; last_error: string | null };
  sorter: { name: string; hardware: boolean };
}

export interface GreenScoreOverview {
  users: Array<{ user_id: string; display_name: string | null; total_points: number; created_at: string }>;
  points_verified_correct: number;
  points_verified_incorrect: number;
  bins: string[];
  rule: string;
}

export interface GreenScoreEvent {
  event_id: number;
  user_id: string;
  image_id: string;
  run_id: string;
  predicted_category: string | null;
  recommended_action: string;
  ecoguard_status: "ACCEPT" | "REVIEW";
  verification_status: "PENDING" | "VERIFIED_CORRECT" | "VERIFIED_INCORRECT";
  observed_action: string | null;
  verified_action: string | null;
  supervisor_id: string | null;
  awarded_points: number;
  created_at: string;
  verified_at: string | null;
}

export interface GreenScoreUser {
  user_id: string;
  total_points: number;
  counts: Record<string, number>;
  contribution_by_bin: Record<string, number>;
  events: GreenScoreEvent[];
}

export interface VerifyResult {
  event_id: number;
  status: string;
  observed_action: string;
  correct_action: string;
  awarded_points: number;
  total_points: number;
}

export interface BatchResponse {
  batch_id: string;
  items: HistoryItem[];
  summary: AnalyticsSummary;
}

export interface ModelMetrics {
  validation: null | {
    scope: string;
    source: string;
    dataset: { training_images: number; validation_images: number; validation_per_class: Record<string, number> };
    accuracy: number;
    per_class_accuracy: Record<string, number>;
    confusion_matrix: { labels: string[]; matrix: number[][] };
    derived_from_confusion_matrix: Record<string, { precision: number; recall: number; f1: number }>;
    ecoguard_threshold_table: Array<{ threshold: number; review_rate: number; accepted_accuracy: number }>;
    limitations: string[];
    pi_reproduction?: {
      scope: string;
      source: string;
      total: number;
      correct: number;
      accuracy: number;
      review_rate: number;
      accepted_only_accuracy: number;
      accepted_incorrect: number;
      inference_median_ms: number;
      inference_p95_ms: number;
      throughput_images_per_second: number;
      note: string;
    };
  };
  test: null | {
    scope: string;
    source: string;
    model_profile: string;
    measured_on: string | null;
    total_images: number;
    overall_accuracy: number;
    review_rate: number;
    accepted_only_accuracy: number | null;
    accepted_incorrect: number;
    metrics: ClassificationMetrics | null;
    inference_latency_median_ms: number | null;
    inference_latency_p95_ms: number | null;
  };
  benchmark: null | {
    source: string;
    platform: { device_model: string | null; machine: string; is_raspberry_pi_5: boolean };
    warning?: string;
    results: Array<{
      threads: number | null;
      median_ms: number;
      p95_ms: number;
      mean_ms: number;
      cpu_ms_per_inference: number | null;
      cpu_percent: number | null;
      throughput: number | null;
      peak_rss_mb: number | null;
    }>;
  };
  live: null | {
    scope: string;
    source: string;
    measured_on: string;
    total: number;
    review_rate: number | null;
    inference_latency: LatencyStats;
    total_latency: LatencyStats;
    labelled_accuracy: LabelledAccuracy | null;
  };
}

export type VoiceMessages = Record<string, { en: string; ta: string }>;
