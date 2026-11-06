import { readBudgetMode } from "../../lib/budget-risk.ts";
import {
  GENSPACE_MIN_SELECTABLE_DURATION_S,
  getOfferingVideoGenerationModelSpecs,
  offeringSpecsAsPipelineItems,
  resolveVideoGenerationOptions,
  VIDEO_GENERATION_RESOLUTION_ORDER,
  untrainedAspectRatioWarning,
  orderAspectRatiosForPicker,
  type OfferingId,
  type ResolvedVideoGenerationOptions,
  type VideoGenerationDuration,
  type VideoGenerationFps,
  type VideoGenerationModelSpecsResponse,
  type VideoGenerationResolution,
} from "../../lib/video-generation-model-specs.ts";
import { videoGenerationResolutionLabel } from "../../lib/video-resolution.ts";
import { chunkedRunNotice, generationResolutionWarning } from "./lowPerformanceWarning.ts";
import type {
  FeatureFieldChange,
  FeatureFieldSchema,
  FeatureValues,
  FieldOption,
} from "../screens/Feature/types.ts";
import {
  clampVideoFields,
  collectModelDisabledFpsOptions,
  collectModelFpsOptions,
  retargetVideoFields,
  type ClampedVideoFields,
  type VideoFeatureAspectRatio,
} from "./videoFieldPolicy.ts";

/** Single source: the shared canonical order (one list, not two). */
export const VIDEO_SETTING_RESOLUTIONS: readonly VideoGenerationResolution[] =
  VIDEO_GENERATION_RESOLUTION_ORDER;
export const VIDEO_SETTING_FPS_VALUES: readonly VideoGenerationFps[] = [
  24, 25, 48, 50,
];
export const VIDEO_SETTING_DURATIONS: readonly VideoGenerationDuration[] = [
  2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20,
];

export type VideoSettingsFieldSchema = FeatureFieldSchema & {
  model: { kind: "options"; value: OfferingId };
  aspectRatio: { kind: "options"; value: VideoFeatureAspectRatio };
  resolution: { kind: "options"; value: VideoGenerationResolution };
  duration: { kind: "options"; value: VideoGenerationDuration };
  fps: { kind: "options"; value: VideoGenerationFps };
};

export type VideoSettingsValues<TSchema extends VideoSettingsFieldSchema> =
  FeatureValues<TSchema> & ClampedVideoFields;

type VideoSettingsModelField = {
  kind: "options";
  id: string;
  label: string;
  dataKey: "model";
  forcePicker: true;
  options: FieldOption<OfferingId>[];
};

type VideoSettingsAspectField<TAspect extends VideoFeatureAspectRatio> = {
  kind: "options";
  id: string;
  label: string;
  dataKey: "aspectRatio";
  options: FieldOption<TAspect>[];
};

type VideoSettingsResolutionField = {
  kind: "options";
  id: string;
  label: string;
  dataKey: "resolution";
  maxOptionCount: number;
  options: FieldOption<VideoGenerationResolution>[];
};

type VideoSettingsDurationField = {
  kind: "options";
  id: string;
  label: string;
  dataKey: "duration";
  forcePicker: true;
  options: FieldOption<VideoGenerationDuration>[];
};

type VideoSettingsFpsField = {
  kind: "options";
  id: string;
  label: string;
  dataKey: "fps";
  maxOptionCount: number;
  options: FieldOption<VideoGenerationFps>[];
};

export type VideoSettingsOptionField<
  TAspect extends VideoFeatureAspectRatio = VideoFeatureAspectRatio,
> =
  | VideoSettingsModelField
  | VideoSettingsAspectField<TAspect>
  | VideoSettingsResolutionField
  | VideoSettingsDurationField
  | VideoSettingsFpsField;

/** The "runs in N windows" notice of the option's cell, when the backend splits that cell. */
function chunkedNotice<T extends string | number>(
  chunked: readonly { value: T; windows: number }[],
  value: T,
): string | undefined {
  const match = chunked.find((option) => option.value === value);
  return match ? chunkedRunNotice(match.windows) : undefined;
}

/**
 * What the backend says about a resolution's cell at the selected fps and duration: its windows,
 * `null` for a single pass, or `undefined` when unknown. It is unknown with an automatic
 * duration, and where the resolution does not offer the selected duration.
 */
function resolutionChunk(
  resolved: ResolvedVideoGenerationOptions,
  resolution: VideoGenerationResolution,
): { windows: number } | null | undefined {
  const run = resolved.chunkedResolutionOptions.find((option) => option.value === resolution);
  if (run != null) return run;
  return resolved.singlePassResolutionOptions.includes(resolution) ? null : undefined;
}

function withCaveat(caveat: string | undefined): Pick<FieldOption, "warning"> {
  return caveat != null ? { warning: caveat } : {};
}

export function isSupportedVideoResolution(
  value: unknown,
): value is VideoGenerationResolution {
  return VIDEO_SETTING_RESOLUTIONS.some((resolution) => resolution === value);
}

export function isSupportedVideoFps(value: unknown): value is VideoGenerationFps {
  return VIDEO_SETTING_FPS_VALUES.some((fps) => fps === value);
}

export function isSupportedVideoDuration(
  value: unknown,
): value is VideoGenerationDuration {
  return VIDEO_SETTING_DURATIONS.some((duration) => duration === value);
}

export function createVideoSettingsOptionFields<
  TSchema extends VideoSettingsFieldSchema,
>(
  values: Pick<
    FeatureValues<TSchema>,
    "model" | "aspectRatio" | "resolution" | "duration" | "fps"
  >,
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  input: { includeAuto?: boolean } = {},
): [
  VideoSettingsModelField,
  VideoSettingsAspectField<
    Extract<TSchema["aspectRatio"]["value"], VideoFeatureAspectRatio>
  >,
  VideoSettingsResolutionField,
  VideoSettingsDurationField,
  VideoSettingsFpsField,
] {
  const offeringSpecs = getOfferingVideoGenerationModelSpecs(specs);
  const resolved = resolveVideoGenerationOptions({
    settings: {
      model: values.model,
      duration: values.duration,
      videoResolution: values.resolution,
      fps: values.fps,
      aspectRatio: values.aspectRatio === "auto" ? "16:9" : values.aspectRatio,
    },
    modelSpecs: offeringSpecsAsPipelineItems(offeringSpecs),
    hasAudio: false,
    minimumDuration: GENSPACE_MIN_SELECTABLE_DURATION_S,
    budgetMode: readBudgetMode(specs?.budgetMode),
  });
  const selectedModel =
    resolved.modelOptions.find(
      (item) => item.pipeline === resolved.selectedModel,
    ) ?? resolved.modelOptions[0];
  const durationOptions = resolved.durationOptions.filter(isSupportedVideoDuration);
  const fpsOptions = collectModelFpsOptions(
    selectedModel,
    resolved.fpsOptions,
  ).filter(isSupportedVideoFps);
  // Resolution-scoped like the aspect picker: a duration the current cell
  // cannot run renders greyed with the backend refusal instead of staying
  // enabled (union) and silently retargeting the resolution on pick.
  const disabledDurations = resolved.disabledDurationOptions;
  // Withheld fps values append after keeps (union scope, numeric): the budget
  // grows monotonically, so drops sit above keeps.
  const disabledFps = collectModelDisabledFpsOptions(selectedModel, fpsOptions);

  return [
    {
      kind: "options",
      id: "model",
      label: "Model",
      dataKey: "model",
      forcePicker: true,
      options: offeringSpecs
        .filter((row) =>
          resolved.modelOptions.some((item) => item.pipeline === String(row.model)),
        )
        .map((row) => ({
          value: row.model,
          label: row.spec.display_name,
        })),
    },
    {
      kind: "options",
      id: "aspectRatio",
      label: "Aspect Ratio",
      dataKey: "aspectRatio",
      options: [
        ...(input.includeAuto
          ? [{ value: "auto" as const, label: "Auto" }]
          : []),
        ...orderAspectRatiosForPicker(resolved.aspectRatioOptions).map((aspectRatio) => {
          const disabled = resolved.disabledAspectRatioOptions.find(
            (option) => option.value === aspectRatio,
          );
          return {
            value: aspectRatio,
            label: aspectRatio,
            ...withCaveat(untrainedAspectRatioWarning(aspectRatio)),
            ...(disabled
              ? {
                  disabledReason: disabled.message,
                  ...(disabled.readout ? { budget: disabled.readout } : {}),
                }
              : {}),
          };
        }),
      ] as VideoSettingsAspectField<
        Extract<TSchema["aspectRatio"]["value"], VideoFeatureAspectRatio>
      >["options"],
    },
    {
      kind: "options",
      id: "resolution",
      label: "Resolution",
      dataKey: "resolution",
      maxOptionCount: VIDEO_SETTING_RESOLUTIONS.length,
      options: [
        ...[...resolved.resolutionOptions]
          .sort(
            (left, right) =>
              VIDEO_SETTING_RESOLUTIONS.indexOf(left) -
              VIDEO_SETTING_RESOLUTIONS.indexOf(right),
          )
          .map((resolution) => ({
            value: resolution,
            label: videoGenerationResolutionLabel(resolution),
            ...withCaveat(
              generationResolutionWarning(
                resolution,
                specs?.low_performance_machine,
                resolutionChunk(resolved, resolution),
              ),
            ),
          })),
        ...resolved.disabledResolutionOptions.map((option) => ({
          value: option.value,
          label: videoGenerationResolutionLabel(option.value),
          disabledReason: option.message,
          ...(option.readout ? { budget: option.readout } : {}),
        })),
      ],
    },
    {
      kind: "options",
      id: "duration",
      label: "Duration",
      dataKey: "duration",
      forcePicker: true,
      options: [
        ...durationOptions.map((duration) => ({
          value: duration,
          label: `${duration}s`,
          ...withCaveat(chunkedNotice(resolved.chunkedDurationOptions, duration)),
        })),
        ...disabledDurations.map((option) => ({
          value: option.value,
          label: `${option.value}s`,
          disabledReason: option.message,
          ...(option.readout ? { budget: option.readout } : {}),
        })),
      ],
    },
    {
      kind: "options",
      id: "fps",
      label: "Frame Rate",
      dataKey: "fps",
      maxOptionCount: VIDEO_SETTING_FPS_VALUES.length,
      options: [
        ...fpsOptions.map((fps) => ({
          value: fps,
          label: String(fps),
          ...withCaveat(chunkedNotice(resolved.chunkedFpsOptions, fps)),
        })),
        ...disabledFps.map((option) => ({
          value: option.value,
          label: String(option.value),
          disabledReason: option.message,
          ...(option.readout ? { budget: option.readout } : {}),
        })),
      ],
    },
  ];
}

export function applyVideoSettingsChange<
  TSchema extends VideoSettingsFieldSchema,
>(
  values: VideoSettingsValues<TSchema>,
  change: FeatureFieldChange<TSchema>,
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  defaults: ClampedVideoFields,
): VideoSettingsValues<TSchema> {
  const merged = { ...values, [change.dataKey]: change.value };
  if (change.dataKey === "duration" || change.dataKey === "fps") {
    return clampVideoFields(
      retargetVideoFields(merged, specs, change.dataKey),
      specs,
      defaults,
    );
  }
  return clampVideoFields(merged, specs, defaults);
}
