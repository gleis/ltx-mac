import { isIcLoraTrackRecipeId } from "../../lib/ic-lora-recipes.ts";
import { roundDurationSeconds } from "../lib/roundDuration.ts";

/** Desktop live setting wins; Remote uses the status fetched when the feature form mounts. */
export function resolveShowManualEnhance(
  liveAutoEnhance: boolean | undefined,
  statusShowManualEnhance: boolean | undefined,
): boolean {
  if (liveAutoEnhance !== undefined) return !liveAutoEnhance;
  return statusShowManualEnhance ?? false;
}

/**
 * Enhance can run from a typed prompt or from media alone: a start frame is captioned, a
 * source video (IC-LoRA, Extend, Retake) is sampled into stills, and the images of an
 * image-list recipe (Ingredients) are read as panels. Audio alone is not enough.
 */
export function hasManualEnhanceInput(input: {
  prompt: string;
  startFrame?: { assetId: string } | null;
  video?: { assetId: string } | null;
  images?: readonly { assetId: string }[] | null;
}): boolean {
  return (
    input.prompt.trim().length > 0 ||
    Boolean(input.startFrame?.assetId) ||
    Boolean(input.video?.assetId) ||
    (input.images?.length ?? 0) > 0
  );
}

/**
 * The panel images an image-list recipe sends to Enhance, in order. Any other feature sends
 * none: its images are a start frame or not part of Enhance.
 */
export function panelImageAssetIds(input: {
  isSheetRecipe: boolean;
  images?: readonly { assetId: string }[] | null;
}): string[] | undefined {
  if (!input.isSheetRecipe || !input.images || input.images.length === 0) return undefined;
  return input.images.map((image) => image.assetId);
}

/**
 * Whether an Enhance result still belongs to the panel images it was built from. `live` is
 * undefined when the form values are gone. The order counts: the rewrite names the panels in it.
 */
export function isPanelListCurrent(
  live: readonly string[] | undefined,
  source: readonly string[],
): boolean {
  return (
    live === undefined ||
    (live.length === source.length && live.every((assetId, index) => assetId === source[index]))
  );
}

/** The asset id of a form field value that holds an asset, or null when it holds none. */
export function assetIdOf(value: unknown): string | null {
  return value != null &&
    typeof value === "object" &&
    "assetId" in value &&
    typeof value.assetId === "string"
    ? value.assetId
    : null;
}

/**
 * Whether an Enhance result still belongs to the source video it was built from. `live` is
 * undefined when the form values are gone, `null` when the form has no video.
 */
export function isSourceVideoCurrent(
  live: string | null | undefined,
  source: string | null,
): boolean {
  return live === undefined || live === source;
}

/**
 * The look image an IC-LoRA recipe sends beside its source video. An IC-LoRA driven by an
 * image, or any other feature, sends none: its image is a start frame or not part of Enhance.
 */
export function lookImageAssetId(input: {
  isIcLoraRecipe: boolean;
  video?: { assetId: string } | null;
  image?: { assetId: string } | null;
}): string | undefined {
  return input.isIcLoraRecipe && input.video?.assetId ? input.image?.assetId : undefined;
}

/**
 * The single image an Enhance request captions: the start frame of Image-to-Video, Audio-to-Video,
 * or a Motion Track recipe, or the look image beside the source video of a video IC-LoRA.
 */
export function enhanceImageAssetId(input: {
  featureId: string;
  isIcLoraRecipe: boolean;
  startFrame?: { assetId: string } | null;
  video?: { assetId: string } | null;
  image?: { assetId: string } | null;
}): string | undefined {
  if (
    input.featureId === "image-to-video" ||
    input.featureId === "audio-to-video" ||
    isIcLoraTrackRecipeId(input.featureId)
  ) {
    return input.startFrame?.assetId;
  }
  return lookImageAssetId(input);
}

/** The media an Enhance request was built from, as the form held it when the request started. */
export type EnhanceSourceMedia = {
  videoAssetId: string | null;
  startFrameAssetId: string | null;
  referenceImageAssetId: string | null;
  panelAssetIds: readonly string[];
  replacedWindow: { start: number; end: number } | undefined;
};

/** The same media as the form holds it now. A field is undefined when the form values are gone. */
export type LiveEnhanceSourceMedia = {
  videoAssetId: string | null | undefined;
  startFrameAssetId: string | null | undefined;
  referenceImageAssetId: string | null | undefined;
  panelAssetIds: readonly string[] | undefined;
  replacedWindow: { start: number; end: number } | undefined;
};

/**
 * Whether an Enhance result still belongs to the media it was built from. A rewrite of an old
 * video, start frame, look image, panel list, or Retake selection must not land on a new one.
 */
export function isEnhanceSourceCurrent(
  live: LiveEnhanceSourceMedia,
  source: EnhanceSourceMedia,
): boolean {
  return (
    isSourceVideoCurrent(live.videoAssetId, source.videoAssetId) &&
    isSourceVideoCurrent(live.startFrameAssetId, source.startFrameAssetId) &&
    isSourceVideoCurrent(live.referenceImageAssetId, source.referenceImageAssetId) &&
    isPanelListCurrent(live.panelAssetIds, source.panelAssetIds) &&
    (live.videoAssetId === undefined ||
      isSameReplacedWindow(live.replacedWindow, source.replacedWindow))
  );
}

/** Whether two Retake windows are the same selection. Both undefined means no window. */
export function isSameReplacedWindow(
  a: { start: number; end: number } | undefined,
  b: { start: number; end: number } | undefined,
): boolean {
  if (a === undefined || b === undefined) return a === b;
  return a.start === b.start && a.end === b.end;
}

/**
 * The part of the source video a Retake replaces, as the create body will send it. Other
 * features replace no part of their video, so they send none.
 */
export function replacedWindowFor(
  featureId: string,
  values: { startTime?: number; duration?: number },
): { start: number; end: number } | undefined {
  if (featureId !== "retake") return undefined;
  if (typeof values.startTime !== "number" || typeof values.duration !== "number") {
    return undefined;
  }
  return {
    start: values.startTime,
    end: values.startTime + roundDurationSeconds(values.duration),
  };
}
