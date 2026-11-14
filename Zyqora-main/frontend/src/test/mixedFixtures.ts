// UI fixtures for Zyqora AI 4.0 pages. They mock the HTTP API; they are NOT model results.
import type { DetectionResult, ModelsStatus } from "@/types/mixed";
import { readyModel } from "./fixtures";

export const labelProfiles = [
  { name: "organiser", title: "Organic / Recyclable (organiser dataset labels)", description: "", labels: { Organic: "Organic", Recyclable: "Recyclable" }, unknown_label: "Unknown", categories: ["Organic", "Recyclable", "Unknown"] },
  { name: "recyclable", title: "Recyclable / Non-Recyclable", description: "", labels: { Organic: "Non-Recyclable", Recyclable: "Recyclable" }, unknown_label: "Unknown", categories: ["Non-Recyclable", "Recyclable", "Unknown"] },
];

export function modelsStatus(detectorReady = true): ModelsStatus {
  return {
    classifier: readyModel,
    detector: detectorReady
      ? { ready: true, code: null, message: null, active_profile_in_config: "yolo11n_640", available_profiles: ["yolo11n_640"], profile: "yolo11n_640",
        imgsz: 640, runtime: "executorch", backend: "ExecuTorch/XNNPACK", trained_on: "COCO 2017 (80 generic classes; not waste-specific)",
        file_size_bytes: 10683072, class_names_are_waste_categories: false }
      : { ready: false, code: "DETECTOR_NOT_FOUND", message: "Detector file not found: models/yolo11n_640_xnnpack.pte",
        active_profile_in_config: "yolo11n_640", available_profiles: ["yolo11n_640"] },
    label_profiles: { active: "organiser", profiles: labelProfiles },
    greenscore_rules: { version: "gs-rules-1" },
    pipeline: { ready: detectorReady, stages: [] },
  };
}

const gs = (score: number | null, reason: string | null = null) => ({
  scored: score != null, score, band: score != null ? "Moderate-to-high potential" : null, basis: score != null ? "model_class:Recyclable" : null,
  components: score != null ? [
    { key: "recycling_potential" as const, title: "Recycling / composting potential", value: 65, weight: 0.4, weighted: 26, rationale: "material unconfirmed" },
    { key: "environmental_impact" as const, title: "Environmental impact rating", value: 55, weight: 0.35, weighted: 19.25, rationale: "unknown material" },
    { key: "disposal_suitability" as const, title: "Disposal suitability", value: 60, weight: 0.25, weighted: 15, rationale: "dry stream" },
  ] : [],
  not_scored_reason: reason, explanation: score != null ? "65 x 0.40 + 55 x 0.35 + 60 x 0.25 = 60.25/100" : "not scored", scoring_version: "gs-rules-1",
});

export function detection(over: Partial<DetectionResult> = {}): DetectionResult {
  return {
    id: 11, filename: "mixed.jpg", image_size: [640, 480], label_profile: labelProfiles[0],
    objects: [
      { index: 0, source: "detector", box: [10, 20, 110, 220], crop_box: [2, 4, 118, 236], detection_label: "bottle", detection_confidence: 0.9, waste_item: true,
        decision: "ACCEPT", review_reasons: [], classification: { model_label: "Recyclable", confidence: 0.99, class_scores: { Organic: 0.01, Recyclable: 0.99 },
          ecoguard_status: "ACCEPT", review_reason: "", preprocess_ms: 1, inference_ms: 12, error: null, score_note: "" },
        material: { category: "Unknown", certainty: "unknown", food: false, note: "PET plastic or glass bottle", hint_material: "Unknown" }, conflict: null,
        organiser_category: "Recyclable", organiser_labels: { organiser: "Recyclable", recyclable: "Recyclable" },
        recommendation: { stream: "RECYCLE", stream_title: { en: "Dry recyclables", ta: "உலர் மறுசுழற்சி" }, guidance: { en: "Check local acceptance.", ta: "x" }, warnings: [], basis: "model_class:Recyclable" },
        green_score: gs(60.25) },
      { index: 1, source: "detector", box: [300, 100, 380, 180], crop_box: null, detection_label: "banana", detection_confidence: 0.8, waste_item: true,
        decision: "REVIEW", review_reasons: ["DETECTOR_CLASSIFIER_CONFLICT"], classification: { model_label: "Recyclable", confidence: 0.98, class_scores: {},
          ecoguard_status: "ACCEPT", review_reason: "", preprocess_ms: 1, inference_ms: 12, error: null, score_note: "" },
        material: { category: "Organic", certainty: "likely", food: true, note: "food / peel", hint_material: "Organic" },
        conflict: "detector saw 'banana' (normally Organic), classifier said Recyclable",
        organiser_category: "Unknown", organiser_labels: { organiser: "Unknown", recyclable: "Unknown" },
        recommendation: { stream: "MANUAL_REVIEW", stream_title: { en: "Manual review", ta: "x" }, guidance: { en: "Ask a supervisor.", ta: "x" }, warnings: [], basis: "review" },
        green_score: gs(null, "DETECTOR_CLASSIFIER_CONFLICT") },
      { index: 2, source: "detector", box: [400, 50, 600, 470], crop_box: null, detection_label: "person", detection_confidence: 0.95, waste_item: false,
        decision: "IGNORED", classification: null, material: { category: "Unknown", certainty: "unknown", food: null, note: "not a waste item", hint_material: "Unknown" },
        conflict: null, organiser_category: null, organiser_labels: {}, recommendation: null, green_score: gs(null, "NOT_A_WASTE_ITEM") },
    ],
    summary: { detected_objects: 3, waste_items: 2, accepted: 1, review: 1, ignored_non_waste: 1, skipped: 0, whole_image_fallback: false,
      by_organiser_category: { Recyclable: 1, Unknown: 1 }, by_profile: {}, by_model_label: { Recyclable: 2 }, by_material: { Unknown: 1, Organic: 1 },
      by_stream: { RECYCLE: 1, MANUAL_REVIEW: 1 }, food_items: 1 },
    image_green_score: { valid: true, score: 60.25, band: "Moderate-to-high potential", scored_items: 1, waste_items: 2, coverage: 0.5, partial: true,
      explanation: "Mean of 1 scored item(s) = 60.25/100. 1 item(s) were not scored.", scoring_version: "gs-rules-1" },
    timing: { detector_preprocess_ms: 10, detector_inference_ms: 87.5, detector_postprocess_ms: 2, classification_ms: 26, classified_crops: 2, pipeline_ms: 131 },
    detector: { profile: "yolo11n_640", imgsz: 640, runtime: "executorch", trained_on: "COCO", candidates_above_conf: 3, suppressed_small: 0, settings: {} },
    notes: [], versions: {}, mode_note: "Mixed-waste mode never actuates a bin.", image_url: "/api/detections/11/image", annotated_url: "/api/detections/11/annotated.jpg",
    ...over,
  };
}
