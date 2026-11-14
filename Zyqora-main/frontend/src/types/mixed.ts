// Zyqora AI 4.0 mixed-waste types. Mirror src/zyqora/mixed.py, api/mixed_service.py and api/detection_store.py.

export type Decision = "ACCEPT" | "REVIEW" | "IGNORED" | "SKIPPED";

export interface GreenScoreComponent {
  key: "recycling_potential" | "environmental_impact" | "disposal_suitability";
  title: string;
  value: number;
  weight: number;
  weighted: number;
  rationale: string;
}

export interface GreenScore {
  scored: boolean;
  score: number | null;
  band: string | null;
  basis: string | null;
  components: GreenScoreComponent[];
  not_scored_reason: string | null;
  explanation: string;
  scoring_version?: string;
  formula?: string;
}

export interface Recommendation {
  stream: string;
  stream_title: { en: string; ta: string };
  guidance: { en: string; ta: string };
  warnings: string[];
  basis: string;
}

export interface DetectedObject {
  index: number;
  object_id?: number;
  source: "detector" | "whole_image";
  box: [number, number, number, number];
  crop_box: [number, number, number, number] | null;
  detection_label: string | null;
  detection_confidence: number | null;
  waste_item: boolean;
  decision: Decision;
  review_reasons?: string[];
  classification: null | {
    model_label: string | null;
    confidence: number | null;
    class_scores: Record<string, number>;
    ecoguard_status: string;
    review_reason: string | null;
    preprocess_ms: number | null;
    inference_ms: number | null;
    error: string | null;
    score_note: string;
  };
  material: { category: string; certainty: string; food: boolean | null; note: string; hint_material: string };
  conflict: string | null;
  organiser_category: string | null;
  organiser_labels: Record<string, string>;
  recommendation: Recommendation | null;
  green_score: GreenScore;
}

export interface LabelProfile {
  name: string;
  title: string;
  description: string;
  labels: Record<string, string>;
  unknown_label: string;
  categories: string[];
}

export interface ImageGreenScore {
  valid: boolean;
  score: number | null;
  band: string | null;
  scored_items: number;
  waste_items: number;
  coverage: number | null;
  partial?: boolean;
  min?: number;
  max?: number;
  explanation: string;
  scoring_version: string;
}

export interface DetectionResult {
  id: number | null;
  created_at?: string;
  filename: string;
  source?: string;
  image_size: [number, number];
  stored_image_size?: [number, number];
  label_profile: LabelProfile;
  objects: DetectedObject[];
  summary: {
    detected_objects: number;
    waste_items: number;
    accepted: number;
    review: number;
    ignored_non_waste: number;
    skipped: number;
    whole_image_fallback: boolean;
    by_organiser_category: Record<string, number>;
    by_profile: Record<string, Record<string, number>>;
    by_model_label: Record<string, number>;
    by_material: Record<string, number>;
    by_stream: Record<string, number>;
    food_items: number;
  };
  image_green_score: ImageGreenScore;
  timing: {
    detector_preprocess_ms: number;
    detector_inference_ms: number;
    detector_postprocess_ms: number;
    classification_ms: number;
    classified_crops: number;
    pipeline_ms: number;
  };
  detector: { profile: string; imgsz: number; runtime: string; trained_on: string; candidates_above_conf: number;
    suppressed_small: number; settings: Record<string, unknown> };
  notes: string[];
  versions: Record<string, string>;
  model?: { classifier_profile: string; classifier_pte: string; classifier_sha256: string | null; classifier_runtime: string;
    detector_profile: string; detector_file: string; detector_runtime: string;
    device: { device_model: string | null; machine: string; is_raspberry_pi: boolean } };
  mode_note?: string;
  image_url: string | null;
  annotated_url: string | null;
}

export interface DetectionListItem {
  id: number;
  created_at: string;
  source: string;
  filename: string;
  total_objects: number;
  waste_items: number;
  accepted_items: number;
  review_items: number;
  ignored_objects: number;
  whole_image_fallback: boolean;
  image_green_score: number | null;
  green_score_coverage: number | null;
  inference_time_ms: number | null;
  pipeline_ms: number | null;
  detector_profile: string;
  label_profile: string;
  image_url: string | null;
  objects: Array<{ obj_index: number; detection_label: string | null; classification_label: string | null; decision: Decision;
    material_category: string | null; organiser_category: string | null }>;
}

export interface DetectionPage { items: DetectionListItem[]; total: number; limit: number; offset: number }

export interface LatencyStats { mean_ms: number | null; median_ms: number | null; p95_ms: number | null; min_ms: number | null;
  max_ms: number | null; count: number }

export interface SystemSample {
  timestamp: string;
  source: string;
  cpu_percent: number | null;
  memory_percent: number | null;
  process_rss_mb: number | null;
  inference_time_ms: number | null;
  temperature_c: number | null;
  measured_power_watts: number | null;
}

export interface DetectionAnalytics {
  images: number;
  detected_objects: number;
  waste_items: number;
  accepted_items: number;
  review_items: number;
  by_profile: Record<string, { title: string; counts: Record<string, number> }>;
  by_model_label: Record<string, number>;
  by_material: Record<string, number>;
  by_stream: Record<string, number>;
  food_items: number;
  green_score: { scored_items: number; mean_item_score: number | null; mean_image_score: number | null;
    by_band: Record<string, number>; series: Array<{ id: number; created_at: string; score: number }> };
  by_day: Array<{ date: string; images: number; objects: number; waste_items: number; mean_green_score: number | null }>;
  latency: { pipeline: LatencyStats; detector: LatencyStats;
    series: Array<{ id: number; created_at: string; pipeline_ms: number; detector_ms: number; classification_ms: number }> };
  system_metrics: SystemSample[];
  latest: number | null;
}

export interface CombinedAnalytics {
  classifications: import("./api").AnalyticsSummary;
  detections: DetectionAnalytics | null;
  scope: string;
}

export interface DetectorStatus {
  ready: boolean;
  code: string | null;
  message: string | null;
  active_profile_in_config: string | null;
  available_profiles: string[];
  profile?: string;
  description?: string;
  file?: string;
  file_size_bytes?: number | null;
  runtime?: string;
  backend?: string;
  imgsz?: number;
  num_classes?: number;
  weights?: string;
  license?: string;
  trained_on?: string;
  sha256_pinned?: string | null;
  load_ms?: number;
  loaded_at?: string;
  settings?: Record<string, unknown>;
  class_names_are_waste_categories?: boolean;
  note?: string;
}

export interface ModelsStatus {
  classifier: import("./api").ModelStatus;
  detector: DetectorStatus;
  label_profiles: { active: string | null; profiles: LabelProfile[] };
  greenscore_rules: { version: string } | null;
  pipeline: { ready: boolean; stages: string[] };
}

export interface DetectionSettingsPayload {
  effective: Record<string, unknown> & { label_profile: string; detector_profile: string; conf_threshold: number;
    iou_threshold: number; agnostic_nms: boolean; max_detections: number; max_objects_classified: number; crop_padding: number;
    min_box_area_fraction: number; whole_image_fallback: boolean; retention_days: number | null; metrics_sample_seconds: number };
  overrides: Record<string, unknown>;
  defaults: Record<string, unknown>;
  descriptions: Record<string, string>;
  options: { detector_profiles: Array<{ name: string; description: string; imgsz: number; runtime: string; available: boolean }>;
    label_profiles: LabelProfile[] };
}

export interface GreenScoreRules {
  version: string;
  weights: Record<string, number>;
  bands: Array<{ min: number; label: string }>;
  materials: Record<string, Record<string, { value: number; why: string }>>;
  model_classes: Record<string, Record<string, { value: number; why: string }>>;
  disclaimer: string;
}

export interface BenchmarkEntry { file: string; committed_evidence: boolean; platform: Record<string, unknown> | null;
  data: Record<string, unknown> }
export interface Benchmarks {
  classifier_benchmarks: BenchmarkEntry[];
  detector_benchmarks: BenchmarkEntry[];
  pipeline_benchmarks: BenchmarkEntry[];
  detector_evaluations: BenchmarkEntry[];
}

export interface DetectorEval {
  dataset: string; profile: string; file: string; runtime: string; imgsz: number; images: number;
  map50: number; map50_95: number; precision_at_conf: number; recall_at_conf: number; operating_conf: number;
  inference_ms_median: number | null; protocol: string; platform: Record<string, unknown>;
  per_class: Array<{ label: string; n_gt: number; ap50: number | null; ap50_95: number | null }>;
}

export interface DetectorBench {
  kind: "detector_benchmark"; profile: string; platform: { device_model: string | null; machine: string; is_raspberry_pi_5: boolean };
  warning: string | null; measured_utc: string;
  results: Array<{ profile: string; runtime: string; backend: string; imgsz: number; threads_requested: number | null;
    file_size_bytes: number; load_ms: number; model_only: LatencyStats; detect_end_to_end: LatencyStats;
    cpu_ms_per_inference: number; throughput_inferences_per_second: number;
    memory_mb: { end: { vm_rss_mb: number | null; vm_hwm_mb: number | null } };
    pipeline?: { images: number; pipeline_latency: LatencyStats; classification_per_crop_ms: number; mean_crops_per_image: number;
      images_per_second: number; cpu_ms_per_image: number; memory_mb_end: { vm_rss_mb: number | null; vm_hwm_mb: number | null } } }>;
}

export interface SystemMetrics extends Omit<import("./api").DeviceMetrics, "model_ready"> {
  model_ready: boolean;
  model_load_ms: number | null;
  detector_ready: boolean;
  detector_profile: string | null;
  detector_load_ms: number | null;
  detector_runtime: string | null;
  backend_status: string;
  pmic: null | { rails_watts: number; rails: number; source: string };
  measured_power_watts: number | null;
  power_note: string;
}
