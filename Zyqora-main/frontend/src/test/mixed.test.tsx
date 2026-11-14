import { QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";
import { AppRoutes } from "@/App";
import { makeQueryClient } from "@/lib/queryClient";
import { SettingsProvider } from "@/lib/settings";
import { baseRoutes, mockApi } from "./fixtures";
import { detection, modelsStatus } from "./mixedFixtures";

function renderAt(path: string) {
  const qc = makeQueryClient();
  qc.setDefaultOptions({ queries: { retry: false, staleTime: 0 } });
  return render(
    <QueryClientProvider client={qc}><SettingsProvider><MemoryRouter initialEntries={[path]}><AppRoutes /></MemoryRouter></SettingsProvider></QueryClientProvider>,
  );
}
const jpg = () => new File([new Uint8Array([255, 216, 255])], "mixed.jpg", { type: "image/jpeg" });

describe("AI Detection page", () => {
  it("sends the image to /api/detect and renders boxes, per-object labels and the Green Score", async () => {
    const { calls } = mockApi({ ...baseRoutes(), "/api/models/status": () => ({ body: modelsStatus() }),
      "POST /api/detect": () => ({ body: detection() }) });
    const user = userEvent.setup();
    renderAt("/detect");
    await screen.findByText(/class names are used only as material hints/);
    await user.upload(screen.getByTestId("file-input"), jpg());
    const btn = screen.getByRole("button", { name: /Analyse waste/ });
    await waitFor(() => expect(btn).toBeEnabled());
    await user.click(btn);
    const report = await screen.findByTestId("detection-report");
    const cards = within(report).getAllByTestId("object-card");
    expect(cards).toHaveLength(3);
    expect(within(cards[0]).getByTestId("organiser-category")).toHaveTextContent("Recyclable");
    expect(within(cards[1]).getByTestId("organiser-category")).toHaveTextContent("Unknown");      // conflict -> Unknown
    expect(within(cards[1]).getByText(/classifier said Recyclable/)).toBeInTheDocument();
    expect(within(cards[2]).getByText(/not classified or scored/)).toBeInTheDocument();          // person
    expect(screen.getByRole("group", { name: "3 detected objects" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /Image Green Score 60.3 of 100/ })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Annotated/ })).toHaveAttribute("href", "/api/detections/11/annotated.jpg");
    const fd = calls.find((c) => c.method === "POST" && c.path === "/api/detect")!.init!.body as FormData;
    expect((fd.get("file") as File).name).toBe("mixed.jpg");
  });

  it("shows the backend error and no result", async () => {
    mockApi({ ...baseRoutes(), "/api/models/status": () => ({ body: modelsStatus() }),
      "POST /api/detect": () => ({ status: 422, body: { error: { code: "INVALID_IMAGE", message: "not a readable image" } } }) });
    const user = userEvent.setup();
    renderAt("/detect");
    await screen.findByText(/material hints/);
    await user.upload(screen.getByTestId("file-input"), jpg());
    await user.click(screen.getByRole("button", { name: /Analyse waste/ }));
    expect(await screen.findByRole("alert")).toHaveTextContent("INVALID_IMAGE");
    expect(screen.queryByTestId("detection-report")).not.toBeInTheDocument();
  });

  it("disables analysis when the detector is missing", async () => {
    mockApi({ ...baseRoutes(), "/api/models/status": () => ({ body: modelsStatus(false) }) });
    const user = userEvent.setup();
    renderAt("/detect");
    expect(await screen.findByText(/DETECTOR_NOT_FOUND/)).toBeInTheDocument();
    await user.upload(screen.getByTestId("file-input"), jpg());
    expect(screen.getByRole("button", { name: /Analyse waste/ })).toBeDisabled();
  });
});

describe("Green Score page", () => {
  it("computes a score through the backend calculator", async () => {
    const { calls } = mockApi({
      ...baseRoutes(),
      "/api/detections": () => ({ body: { items: [], total: 0, limit: 30, offset: 0 } }),
      "/api/analytics": () => ({ body: { classifications: null, detections: null, scope: "" } }),
      "/api/greenscore/rules": () => ({ body: { version: "gs-rules-1", weights: {}, bands: [{ min: 0, label: "Very low" }], materials: {}, model_classes: {}, disclaimer: "project-defined" } }),
      "POST /api/greenscore": () => ({ body: { scored: true, score: 89, band: "High sustainability potential", basis: "custom", not_scored_reason: null,
        explanation: "90 x 0.40 + 80 x 0.35 + 100 x 0.25 = 89/100", scoring_version: "gs-rules-1", components: [] } }),
    });
    const user = userEvent.setup();
    renderAt("/green-score");
    await screen.findByText("No analyses yet");
    await user.click(screen.getByRole("radio", { name: "Custom ratings" }));
    await user.click(screen.getByRole("button", { name: /Calculate/ }));
    expect(await screen.findByRole("img", { name: /Green Score 89.0 of 100/ })).toBeInTheDocument();
    const body = JSON.parse(calls.find((c) => c.method === "POST" && c.path === "/api/greenscore")!.init!.body as string);
    expect(body).toEqual({ components: { recycling_potential: 90, environmental_impact: 80, disposal_suitability: 100 } });
  });
});

describe("Detection history", () => {
  it("lists analyses and deletes one after confirmation", async () => {
    const item = { id: 11, created_at: "2026-11-14T08:00:00Z", source: "upload", filename: "mixed.jpg", total_objects: 3, waste_items: 2, accepted_items: 1,
      review_items: 1, ignored_objects: 1, whole_image_fallback: false, image_green_score: 60.25, green_score_coverage: 0.5, inference_time_ms: 110,
      pipeline_ms: 131, detector_profile: "yolo11n_640", label_profile: "organiser", image_url: null,
      objects: [{ obj_index: 0, detection_label: "bottle", classification_label: "Recyclable", decision: "ACCEPT", material_category: "Unknown", organiser_category: "Recyclable" }] };
    const { calls } = mockApi({ ...baseRoutes(), "DELETE /api/detections/11": () => ({ body: { deleted: 11 } }),
      "/api/detections": () => ({ body: { items: [item], total: 1, limit: 20, offset: 0 } }) });
    vi.spyOn(window, "confirm").mockReturnValue(true);
    const user = userEvent.setup();
    renderAt("/history");
    const rows = await screen.findAllByTestId("detection-row");
    expect(within(rows[0]).getByText(/bottle → Recyclable/)).toBeInTheDocument();
    await user.click(within(rows[0]).getByRole("button", { name: "Delete analysis 11" }));
    await waitFor(() => expect(calls.some((c) => c.method === "DELETE" && c.path === "/api/detections/11")).toBe(true));
  });
});
