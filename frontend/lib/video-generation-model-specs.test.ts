import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  cheapestDroppedCellMessage,
  type ChunkedFastCell,
  resolveVideoGenerationOptions,
  type DroppedFastAspectCell,
  type DroppedFastCell,
  type VideoGenerationModelSpecItem,
  type WarnedFastCell,
} from "./video-generation-model-specs.ts";

type KeptGrid = Record<string, { fps_to_durations: Record<string, number[]> }>;

function specItem(
  kept: KeptGrid,
  dropped: DroppedFastCell[],
  a2v: KeptGrid = {},
  aspectDropped: DroppedFastAspectCell[] = [],
): VideoGenerationModelSpecItem {
  return {
    pipeline: "fast",
    spec: {
      display_name: "LTX 2.5 Fast",
      supported_resolutions_durations:
        kept as VideoGenerationModelSpecItem["spec"]["supported_resolutions_durations"],
      a2v_supported_resolutions_durations:
        a2v as VideoGenerationModelSpecItem["spec"]["a2v_supported_resolutions_durations"],
      capabilities: null,
      dropped_cells: dropped,
      dropped_aspect_cells: aspectDropped,
    },
  };
}

function dropped(
  resolution: DroppedFastCell["resolution"],
  fps: DroppedFastCell["fps"],
  duration: DroppedFastCell["duration"],
  message: string,
  audio = false,
): DroppedFastCell {
  return { resolution, fps, duration, reason: "stream_over", message, audio };
}

function aspectDropped(
  resolution: DroppedFastAspectCell["resolution"],
  fps: DroppedFastAspectCell["fps"],
  duration: DroppedFastAspectCell["duration"],
  aspectRatio: DroppedFastAspectCell["aspectRatio"],
  message: string,
  audio = false,
): DroppedFastAspectCell {
  return { resolution, fps, duration, aspectRatio, reason: "stream_over", message, audio };
}

describe("resolveVideoGenerationOptions disabled options", () => {
  it("returns exact duration messages and cheapest field messages, kept-only selections", () => {
    const item = specItem(
      {
        "540p": { fps_to_durations: { "24": [5, 10, 20] } },
        "720p": { fps_to_durations: { "24": [5, 10] } },
      },
      [
        dropped("720p", 24, 20, "needs 15.8"),
        dropped("1080p", 24, 5, "needs 18.3"),
        dropped("1080p", 24, 10, "needs 18.3+"),
      ],
    );
    const resolved = resolveVideoGenerationOptions({
      settings: { model: "fast", duration: 5, videoResolution: "720p", fps: 24 },
      modelSpecs: [item],
    });

    assert.deepEqual(resolved.durationOptions, [5, 10]);
    assert.deepEqual(resolved.disabledDurationOptions, [
      { value: 20, message: "needs 15.8" },
    ]);
    assert.deepEqual(resolved.disabledResolutionOptions, [
      { value: "1080p", message: "needs 18.3" },
    ]);
    assert.deepEqual(resolved.disabledFpsOptions, []);
    assert.equal(resolved.selectedDuration, 5);
    assert.equal(resolved.hasCompatibleOptions, true);
  });

  it("groups dropped-only fps with the cheapest message", () => {
    const item = specItem(
      { "720p": { fps_to_durations: { "24": [5] } } },
      [
        dropped("720p", 48, 5, "needs f48"),
        dropped("720p", 48, 10, "needs f48+"),
      ],
    );
    const resolved = resolveVideoGenerationOptions({
      settings: { model: "fast", duration: 5, videoResolution: "720p", fps: 24 },
      modelSpecs: [item],
    });

    assert.deepEqual(resolved.fpsOptions, [24]);
    assert.deepEqual(resolved.disabledFpsOptions, [
      { value: 48, message: "needs f48" },
    ]);
  });

  it("filters the dropped matrix by audio both ways", () => {
    const item = specItem(
      { "540p": { fps_to_durations: { "24": [5] } } },
      [
        dropped("540p", 24, 10, "ram-msg", true),
        dropped("540p", 24, 20, "plain-msg", false),
      ],
      { "540p": { fps_to_durations: { "24": [5] } } },
    );
    const audio = resolveVideoGenerationOptions({
      settings: { model: "fast", duration: 5, videoResolution: "540p", fps: 24 },
      modelSpecs: [item],
      hasAudio: true,
    });
    const plain = resolveVideoGenerationOptions({
      settings: { model: "fast", duration: 5, videoResolution: "540p", fps: 24 },
      modelSpecs: [item],
      hasAudio: false,
    });

    assert.deepEqual(audio.disabledDurationOptions, [
      { value: 10, message: "ram-msg" },
    ]);
    assert.deepEqual(plain.disabledDurationOptions, [
      { value: 20, message: "plain-msg" },
    ]);
  });

  it("falls back to first-row disabled resolutions when no model is compatible", () => {
    const item = specItem({}, [
      dropped("720p", 24, 5, "m720"),
      dropped("1080p", 24, 5, "m1080"),
    ]);
    const resolved = resolveVideoGenerationOptions({
      settings: { model: "fast", duration: 5, videoResolution: "540p", fps: 24 },
      modelSpecs: [item],
    });

    assert.equal(resolved.selectedModel, null);
    assert.deepEqual(resolved.disabledResolutionOptions, [
      { value: "720p", message: "m720" },
      { value: "1080p", message: "m1080" },
    ]);
    assert.deepEqual(resolved.disabledFpsOptions, []);
    assert.deepEqual(resolved.disabledDurationOptions, []);
  });

  it("does not leak a refused shape to sibling cells", () => {
    const item = specItem(
      { "540p": { fps_to_durations: { "24": [5, 20] } } },
      [],
      {},
      [aspectDropped("540p", 24, 20, "21:9", "needs 17.6")],
    );
    const refused = resolveVideoGenerationOptions({
      settings: { model: "fast", duration: 20, videoResolution: "540p", fps: 24 },
      modelSpecs: [item],
    });
    const sibling = resolveVideoGenerationOptions({
      settings: { model: "fast", duration: 5, videoResolution: "540p", fps: 24 },
      modelSpecs: [item],
    });

    assert.deepEqual(refused.disabledAspectRatioOptions, [
      { value: "21:9", message: "needs 17.6" },
    ]);
    assert.deepEqual(sibling.disabledAspectRatioOptions, []);
  });

  it("does not leak aspect refusals across audio matrices", () => {
    const item = specItem(
      { "540p": { fps_to_durations: { "24": [5] } } },
      [],
      { "540p": { fps_to_durations: { "24": [5] } } },
      [aspectDropped("540p", 24, 5, "21:9", "a2v-msg", true)],
    );
    const audio = resolveVideoGenerationOptions({
      settings: { model: "fast", duration: 5, videoResolution: "540p", fps: 24 },
      modelSpecs: [item],
      hasAudio: true,
    });
    const plain = resolveVideoGenerationOptions({
      settings: { model: "fast", duration: 5, videoResolution: "540p", fps: 24 },
      modelSpecs: [item],
      hasAudio: false,
    });

    assert.deepEqual(audio.disabledAspectRatioOptions, [
      { value: "21:9", message: "a2v-msg" },
    ]);
    assert.deepEqual(plain.disabledAspectRatioOptions, []);
  });
});

describe("resolveVideoGenerationOptions reckless promotion", () => {
  it("offers a stream-over duration and keeps an unsupported resolution disabled", () => {
    const item = specItem(
      { "720p": { fps_to_durations: { "24": [5] } } },
      [
        dropped("720p", 24, 20, "needs 14.1"),
        { ...dropped("1080p", 24, 5, "no pipeline"), reason: "unsupported" },
      ],
    );
    const standard = resolveVideoGenerationOptions({
      settings: { model: "fast", duration: 5, videoResolution: "720p", fps: 24 },
      modelSpecs: [item],
    });
    assert.deepEqual(standard.durationOptions, [5]);
    assert.deepEqual(standard.disabledDurationOptions, [
      { value: 20, message: "needs 14.1" },
    ]);

    const reckless = resolveVideoGenerationOptions({
      settings: { model: "fast", duration: 20, videoResolution: "720p", fps: 24 },
      modelSpecs: [item],
      budgetMode: "reckless",
    });
    assert.deepEqual(reckless.durationOptions, [5, 20]);
    assert.deepEqual(reckless.disabledDurationOptions, []);
    assert.deepEqual(reckless.warnedDurationOptions, [
      { value: 20, message: "needs 14.1", tier: "over" },
    ]);
    assert.deepEqual(
      reckless.disabledResolutionOptions.map((option) => option.value),
      ["1080p"],
    );
  });

  it("leaves a free-VRAM drop and a busy-card drop disabled", () => {
    const item = specItem(
      { "720p": { fps_to_durations: { "24": [5] } } },
      [
        {
          ...dropped("720p", 24, 10, "busy"),
          shortfall_kind: "free",
          readout: { kind: "gpu_busy", need_gb: 18, have_gb: 12 },
        },
        { ...dropped("720p", 24, 20, "free now"), shortfall_kind: "free" },
      ],
    );
    const reckless = resolveVideoGenerationOptions({
      settings: { model: "fast", duration: 5, videoResolution: "720p", fps: 24 },
      modelSpecs: [item],
      budgetMode: "reckless",
    });
    assert.deepEqual(reckless.durationOptions, [5]);
    assert.deepEqual(
      reckless.disabledDurationOptions.map((option) => option.value),
      [10, 20],
    );
  });

  it("marks a kept tight cell without removing it from the offered durations", () => {
    const item = specItem(
      { "720p": { fps_to_durations: { "24": [5] } } },
      [],
    );
    const warned: WarnedFastCell = {
      resolution: "720p",
      fps: 24,
      duration: 5,
      tier: "tight",
      message: "tight fit",
      audio: false,
    };
    item.spec.warned_cells = [warned];
    const resolved = resolveVideoGenerationOptions({
      settings: { model: "fast", duration: 5, videoResolution: "720p", fps: 24 },
      modelSpecs: [item],
    });
    assert.deepEqual(resolved.durationOptions, [5]);
    assert.deepEqual(resolved.warnedDurationOptions, [
      { value: 5, message: "tight fit", tier: "tight" },
    ]);
  });
});

describe("resolveVideoGenerationOptions chunked cells", () => {
  const chunked = (
    resolution: ChunkedFastCell["resolution"],
    fps: ChunkedFastCell["fps"],
    duration: ChunkedFastCell["duration"],
    windows: number,
    audio = false,
  ): ChunkedFastCell => ({ resolution, fps, duration, windows, window_frames: 241, audio });

  function item(cells: ChunkedFastCell[]): VideoGenerationModelSpecItem {
    const base = specItem(
      {
        "720p": { fps_to_durations: { "24": [5, 10, 20], "48": [5, 10] } },
        "1080p": { fps_to_durations: { "24": [5, 10, 20], "48": [5, 10] } },
      },
      [],
    );
    base.spec.chunked_cells = cells;
    return base;
  }

  it("lists the cells that run in windows on the duration, fps and resolution pickers", () => {
    const resolved = resolveVideoGenerationOptions({
      settings: { model: "fast", duration: 20, videoResolution: "1080p", fps: 24 },
      modelSpecs: [item([chunked("1080p", 24, 20, 3), chunked("1080p", 48, 10, 2)])],
    });
    assert.deepEqual(resolved.chunkedDurationOptions, [{ value: 20, windows: 3, windowFrames: 241 }]);
    assert.deepEqual(resolved.chunkedResolutionOptions, [{ value: "1080p", windows: 3, windowFrames: 241 }]);
    // The fps picker reads the selected duration: 24 fps is the selected cell, 48 fps at 20 s is not offered.
    assert.deepEqual(resolved.chunkedFpsOptions, [{ value: 24, windows: 3, windowFrames: 241 }]);
  });

  it("judges the fps picker at the selected duration", () => {
    const resolved = resolveVideoGenerationOptions({
      settings: { model: "fast", duration: 10, videoResolution: "1080p", fps: 24 },
      modelSpecs: [item([chunked("1080p", 48, 10, 2)])],
    });
    assert.deepEqual(resolved.chunkedFpsOptions, [{ value: 48, windows: 2, windowFrames: 241 }]);
    assert.deepEqual(resolved.chunkedDurationOptions, []);
  });

  it("does not mix the A2V grid into the plain one", () => {
    const resolved = resolveVideoGenerationOptions({
      settings: { model: "fast", duration: 20, videoResolution: "1080p", fps: 24 },
      modelSpecs: [item([chunked("1080p", 24, 20, 3, true)])],
    });
    assert.deepEqual(resolved.chunkedDurationOptions, []);
  });

  it("lists a cell Reckless mode promotes, from the window its refusal carries", () => {
    const base = item([]);
    base.spec.dropped_cells = [
      { ...dropped("1080p", 24, 20, "too big"), windows: 3, window_frames: 241 },
    ];
    const settings = { model: "fast", duration: 20, videoResolution: "1080p", fps: 24 } as const;
    const reckless = resolveVideoGenerationOptions({
      settings,
      modelSpecs: [base],
      budgetMode: "reckless",
    });
    assert.deepEqual(reckless.durationOptions.includes(20), true);
    assert.deepEqual(reckless.chunkedDurationOptions, [{ value: 20, windows: 3, windowFrames: 241 }]);
    const standard = resolveVideoGenerationOptions({ settings, modelSpecs: [base] });
    assert.deepEqual(standard.chunkedDurationOptions, []);
  });

  it("names the resolutions that offer the selected cell and run it in one pass", () => {
    const base = specItem(
      {
        "720p": { fps_to_durations: { "24": [5, 10, 20] } },
        "1080p": { fps_to_durations: { "24": [5, 10, 20] } },
        // 4K stops at 10 s: choosing it from 20 s changes the duration, so its cell is unknown.
        "2160p": { fps_to_durations: { "24": [5, 10] } },
      },
      [],
    );
    base.spec.chunked_cells = [chunked("1080p", 24, 20, 3)];
    const resolved = resolveVideoGenerationOptions({
      settings: { model: "fast", duration: 20, videoResolution: "1080p", fps: 24 },
      modelSpecs: [base],
    });
    assert.deepEqual(resolved.singlePassResolutionOptions, ["720p"]);
    assert.deepEqual(resolved.chunkedResolutionOptions, [{ value: "1080p", windows: 3, windowFrames: 241 }]);
  });

  it("leaves a cell that runs in one pass out", () => {
    const resolved = resolveVideoGenerationOptions({
      settings: { model: "fast", duration: 5, videoResolution: "720p", fps: 24 },
      modelSpecs: [item([chunked("1080p", 24, 20, 3)])],
    });
    assert.deepEqual(resolved.chunkedDurationOptions, []);
    assert.deepEqual(resolved.chunkedResolutionOptions, []);
  });
});

describe("cheapestDroppedCellMessage", () => {
  it("orders one way everywhere: resolution, then duration, then fps", () => {
    // Home unioned fps-first while GenSpace scoped duration-first, so the
    // same withheld option tooltiped different refusals per surface.
    assert.equal(
      cheapestDroppedCellMessage([
        dropped("720p", 48, 5, "720p-dur5"),
        dropped("540p", 50, 10, "540p-dur10-fps50"),
        dropped("540p", 48, 10, "540p-dur10-fps48"),
      ]),
      "540p-dur10-fps48",
    );
  });
});
