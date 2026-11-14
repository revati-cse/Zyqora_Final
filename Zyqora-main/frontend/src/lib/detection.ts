import type { Decision, DetectedObject, DetectionResult } from "@/types/mixed";

export const DECISION_VAR: Record<Decision, string> = {
  ACCEPT: "var(--accept)",
  REVIEW: "var(--review)",
  IGNORED: "var(--muted-fg)",
  SKIPPED: "var(--muted-fg)",
};

export const DECISION_TEXT: Record<Decision, string> = {
  ACCEPT: "Reliable",
  REVIEW: "Review — shown as Unknown",
  IGNORED: "Not a waste item",
  SKIPPED: "Not classified (object limit)",
};

export const NOT_SCORED_TEXT: Record<string, string> = {
  CLASSIFICATION_NOT_RELIABLE: "not scored: classification not reliable",
  DETECTOR_CLASSIFIER_CONFLICT: "not scored: detector and classifier disagree",
  NOT_A_WASTE_ITEM: "not a waste item",
  NO_RULE_FOR_CATEGORY: "not scored: no rule for this category",
  MISSING_INFORMATION: "not scored: missing information",
  NOT_CLASSIFIED_LIMIT: "not classified (object limit)",
};

export function objectTitle(o: DetectedObject): string {
  if (o.source === "whole_image") return "Whole image";
  return o.detection_label ?? "object";
}

export const wasteObjects = (r: DetectionResult) => r.objects.filter((o) => o.decision === "ACCEPT" || o.decision === "REVIEW");

/** Flat rows for CSV / JSON download of one analysis. */
export function detectionRows(r: DetectionResult) {
  return r.objects.map((o) => ({
    analysis_id: r.id ?? "",
    filename: r.filename,
    index: o.index,
    source: o.source,
    box_x1: o.box[0], box_y1: o.box[1], box_x2: o.box[2], box_y2: o.box[3],
    detection_label: o.detection_label ?? "",
    detection_confidence: o.detection_confidence ?? "",
    model_label: o.classification?.model_label ?? "",
    model_score: o.classification?.confidence ?? "",
    decision: o.decision,
    organiser_category: o.organiser_category ?? "",
    material: o.material.category,
    stream: o.recommendation?.stream ?? "",
    green_score: o.green_score.score ?? "",
    conflict: o.conflict ?? "",
  }));
}

export function bandTone(score: number | null | undefined): string {
  if (score == null) return "var(--muted-fg)";
  if (score >= 60) return "var(--accept)";
  if (score >= 40) return "var(--chart-3)";
  return "var(--review)";
}
