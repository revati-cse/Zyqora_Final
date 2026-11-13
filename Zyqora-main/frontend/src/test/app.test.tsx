import { QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";
import { AppRoutes } from "@/App";
import { makeQueryClient } from "@/lib/queryClient";
import { SettingsProvider } from "@/lib/settings";
import { validateImageFile } from "@/lib/validation";
import { baseRoutes, classification, historyPage, missingModel, mockApi } from "./fixtures";

function renderAt(path: string) {
  const qc = makeQueryClient();
  qc.setDefaultOptions({ queries: { retry: false, staleTime: 0 } });
  return render(
    <QueryClientProvider client={qc}>
      <SettingsProvider>
        <MemoryRouter initialEntries={[path]}>
          <AppRoutes />
        </MemoryRouter>
      </SettingsProvider>
    </QueryClientProvider>,
  );
}

const png = () => new File([new Uint8Array([137, 80, 78, 71])], "bottle.png", { type: "image/png" });

describe("navigation and status", () => {
  it("navigates between pages from the sidebar", async () => {
    mockApi({ ...baseRoutes(), "/api/history": () => ({ body: historyPage }), "/api/analytics": () => ({ body: null }),
      "/api/greenscore": () => ({ body: { users: [], points_verified_correct: 10, points_verified_incorrect: 0, bins: [], rule: "" } }) });
    const user = userEvent.setup();
    renderAt("/classify");
    expect(await screen.findByRole("heading", { level: 1, name: "Live Classification" })).toBeInTheDocument();
    const nav = screen.getAllByRole("navigation", { name: "Main" })[0];
    await user.click(within(nav).getByRole("link", { name: "History" }));
    expect(await screen.findByRole("heading", { level: 1, name: "History" })).toBeInTheDocument();
  });

  it("shows real connection state, including DEVICE DISCONNECTED", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("Failed to fetch")); // device unreachable
    renderAt("/classify");
    // the health query retries once before reporting the failure
    expect((await screen.findAllByText("DEVICE DISCONNECTED", {}, { timeout: 5000 })).length).toBeGreaterThan(0);
  });
});

describe("live classification", () => {
  it("uploads an image, shows progress, then the backend's result", async () => {
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    const { calls } = mockApi({
      ...baseRoutes(),
      "POST /api/inference": async () => { await gate; return { body: classification() }; },
    });
    const user = userEvent.setup();
    renderAt("/classify");
    await screen.findByText(/classes Organic \/ Recyclable/);
    await user.upload(screen.getByTestId("file-input"), png());
    const button = screen.getByRole("button", { name: /Classify Waste/ });
    await waitFor(() => expect(button).toBeEnabled());
    await user.click(button);
    expect(await screen.findByText("INFERENCE IN PROGRESS")).toBeInTheDocument();
    release();
    expect(await screen.findByTestId("predicted-category")).toHaveTextContent("Recyclable");
    const card = screen.getByTestId("result-card");
    expect(within(card).getByText("ACCEPT")).toBeInTheDocument();
    expect(within(card).getByText("3.21 ms")).toBeInTheDocument();          // measured latency from the response
    expect(within(card).getByText(/would open the RECYCLE bin/)).toBeInTheDocument();
    const post = calls.find((c) => c.method === "POST" && c.path === "/api/inference")!;
    const fd = post.init!.body as FormData;
    expect((fd.get("file") as File).name).toBe("bottle.png");               // the actual file is sent
  });

  it("explains a REVIEW decision and says no bin opens", async () => {
    mockApi({ ...baseRoutes(), "POST /api/inference": () => ({ body: classification({
      predicted_category: "Organic", confidence: 0.7, class_scores: { Organic: 0.7, Recyclable: 0.3 }, disposal_action: "MANUAL_REVIEW",
      ecoguard: { status: "REVIEW", threshold: 0.95, reasons: ["LOW_CONFIDENCE"], explanation: { en: "the model's confidence is below the EcoGuard threshold", ta: "x" } },
      guidance: { en: "Please ask a supervisor to check this item.", ta: "y" } }) }) });
    const user = userEvent.setup();
    renderAt("/classify");
    await screen.findByText(/classes Organic/);
    await user.upload(screen.getByTestId("file-input"), png());
    await user.click(screen.getByRole("button", { name: /Classify Waste/ }));
    expect(await screen.findByText("REVIEW REQUIRED")).toBeInTheDocument();
    expect(screen.getByText(/No bin is opened automatically/)).toBeInTheDocument();
    expect(screen.getByText(/below the EcoGuard threshold/)).toBeInTheDocument();
  });

  it("shows the backend error instead of a fake result", async () => {
    mockApi({ ...baseRoutes(), "POST /api/inference": () => ({ status: 422, body: { error: { code: "INVALID_IMAGE", message: "not a readable image" } } }) });
    const user = userEvent.setup();
    renderAt("/classify");
    await screen.findByText(/classes Organic/);
    await user.upload(screen.getByTestId("file-input"), png());
    await user.click(screen.getByRole("button", { name: /Classify Waste/ }));
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("INVALID_IMAGE");
    expect(screen.queryByTestId("result-card")).not.toBeInTheDocument();
  });

  it("disables classification when the model is unavailable", async () => {
    mockApi(baseRoutes(missingModel));
    const user = userEvent.setup();
    renderAt("/classify");
    expect(await screen.findByText(/MODEL_NOT_FOUND/)).toBeInTheDocument();
    await user.upload(screen.getByTestId("file-input"), png());
    expect(screen.getByRole("button", { name: /Classify Waste/ })).toBeDisabled();
  });

  it("rejects unsupported and oversized files client-side", () => {
    expect(validateImageFile(new File(["x"], "a.gif", { type: "image/gif" }))).toMatch(/only JPEG, PNG or WebP/);
    expect(validateImageFile(new File([], "a.png", { type: "image/png" }))).toMatch(/empty/);
    expect(validateImageFile(new File([new Uint8Array(11 * 1024 * 1024)], "big.jpg", { type: "image/jpeg" }))).toMatch(/10 MB/);
    expect(validateImageFile(png())).toBeNull();
  });

  it("renders Tamil guidance when Tamil is selected", async () => {
    mockApi({ ...baseRoutes(), "POST /api/inference": () => ({ body: classification() }) });
    const user = userEvent.setup();
    renderAt("/classify");
    await screen.findByText(/classes Organic/);
    await user.click(screen.getByRole("button", { name: "Switch language" }));
    await user.upload(screen.getByTestId("file-input"), png());
    await user.click(screen.getByRole("button", { name: /கழிவை வகைப்படுத்து/ }));
    expect(await screen.findByText("இந்தப் பொருளை மறுசுழற்சித் தொட்டியில் போடவும்.")).toBeInTheDocument();
    expect(screen.getByTestId("predicted-category")).toHaveTextContent("மறுசுழற்சி செய்யக்கூடியது");
  });
});

describe("batch and history", () => {
  it("processes a queue sequentially and loads the server summary", async () => {
    let n = 0;
    const { calls } = mockApi({
      ...baseRoutes(),
      "POST /api/inference": () => (++n === 2
        ? { status: 422, body: { error: { code: "INVALID_IMAGE", message: "bad" } } }
        : { body: classification({ id: n }) }),
      "/api/batches/": () => ({ body: { batch_id: "b", items: [], summary: { total: 2, accepted: 2, review: 0, review_rate: 0, by_category: {}, by_action: {}, by_day: [],
        latest: null, inference_latency: { median_ms: 3, p95_ms: 4, mean_ms: 3, min_ms: 3, max_ms: 4, count: 2 },
        total_latency: { median_ms: 8, p95_ms: 9, mean_ms: 8, min_ms: 8, max_ms: 9, count: 2 }, latency_series: [], batch_throughput: [], labelled_accuracy: null } } }),
    });
    const user = userEvent.setup();
    renderAt("/batch");
    await screen.findByRole("heading", { level: 1, name: "Batch Analysis" });
    await user.upload(screen.getByTestId("file-input"), [png(), new File([new Uint8Array([1])], "b.jpg", { type: "image/jpeg" }), png()]);
    expect(screen.getAllByTestId("batch-row")).toHaveLength(3);
    await waitFor(() => expect(screen.getByRole("button", { name: /Classify queue/ })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: /Classify queue/ }));
    expect(await screen.findByText(/2 done · 1 failed · 3 total/)).toBeInTheDocument();
    expect(screen.getByText("needs labels")).toBeInTheDocument();                // no accuracy without labels
    expect(screen.getByRole("button", { name: /Retry failed/ })).toBeEnabled();
    expect(calls.filter((c) => c.method === "POST").length).toBe(3);
  });

  it("renders history rows with decisions", async () => {
    mockApi({ ...baseRoutes(), "/api/history": () => ({ body: historyPage }) });
    renderAt("/history");
    const rows = await screen.findAllByTestId("history-row");
    expect(rows).toHaveLength(2);
    expect(within(rows[0]).getByText("REVIEW REQUIRED")).toBeInTheDocument();
    expect(within(rows[1]).getByText("ACCEPT")).toBeInTheDocument();
  });
});

describe("accessibility basics", () => {
  it("labels controls and provides a skip link and main landmark", async () => {
    mockApi(baseRoutes());
    renderAt("/classify");
    await screen.findByRole("heading", { level: 1 });
    expect(screen.getByRole("main")).toBeInTheDocument();
    expect(screen.getByText("Skip to content")).toHaveAttribute("href", "#main");
    expect(screen.getByLabelText("Choose an image file")).toBeInTheDocument();
    for (const b of screen.getAllByRole("button")) expect(b).toHaveAccessibleName();
  });
});
