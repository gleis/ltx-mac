import type { components } from "../../../../generated/backend-openapi";
import { IC_LORA_SHEET_MAX_IMAGES, type IcLoraSheetRecipe } from "../../../../lib/ic-lora-recipes.ts";
import {
  isVideoGenerationAspectRatio,
  type OfferingId,
  withDefaultOffering,
  type VideoGenerationDuration,
  type VideoGenerationFps,
  type VideoGenerationResolution,
} from "../../../../lib/video-generation-model-specs.ts";
import { readGenerationParams } from "../../../lib/resultsFeedModel.ts";
import {
  isLocalVideoSelectionReady,
  type VideoFeatureAspectRatio,
} from "../../../lib/videoFieldPolicy.ts";
import { applyVideoSettingsChange } from "../../../lib/videoSettingsControls.ts";
import {
  type AssetRef,
  type FeatureDefinition,
  type FeatureFormModel,
  type FeatureValues,
  type ValidationIssue,
} from "../types.ts";
import { icLoraVariantId } from "./icLoraNormalize.ts";
import {
  type IcLoraAudioMode,
  isIcLoraAudioMode,
  isSynthesizedIcLoraResolution,
  isSynthesizedIcLoraUnavailable,
  normalizeSynthesizedIcLoraValues,
  readSynthesizedCommonParams,
  type SynthesizedIcLoraCatalogSettings,
  type SynthesizedIcLoraContext,
  synthesizedIcLoraVideoOptionFields,
} from "./icLoraSynthesizedCommon.ts";
import { readAssetRef } from "./videoFeature.ts";

type CreateIcLoraSheetRequest = components["schemas"]["CreateIcLoraSheetRequest"];

export type IcLoraSheetSchema = {
  images: { kind: "image-assets"; value: AssetRef[] };
  prompt: { kind: "textarea" };
  model: { kind: "options"; value: OfferingId };
  variant: { kind: "options"; value: string };
  aspectRatio: { kind: "options"; value: VideoFeatureAspectRatio };
  resolution: { kind: "options"; value: VideoGenerationResolution };
  duration: { kind: "options"; value: VideoGenerationDuration };
  fps: { kind: "options"; value: VideoGenerationFps };
  audioMode: { kind: "options"; value: IcLoraAudioMode };
  loraStrength: { kind: "slider"; value: number };
};

export type IcLoraSheetValues = FeatureValues<IcLoraSheetSchema>;

export type IcLoraSheetContext = SynthesizedIcLoraContext & {
  /** The packaged pre-made sheet of the example. It fills a new form and Reset. */
  seed?: AssetRef[];
};

export type IcLoraSheetCatalogSettings = SynthesizedIcLoraCatalogSettings;

function defaultsFor(
  recipe: IcLoraSheetRecipe,
  settings?: IcLoraSheetCatalogSettings | null,
): IcLoraSheetValues {
  const audio = settings?.audio_mode;
  return {
    images: [],
    prompt: recipe.demoPrompt,
    model: "ltx-2.5-fast",
    variant: "",
    aspectRatio: "16:9",
    resolution: "720p",
    duration: recipe.homeDefaults.seconds,
    fps: recipe.homeDefaults.fps,
    audioMode: isIcLoraAudioMode(audio) ? audio : "generated",
    loraStrength: recipe.homeDefaults.loraStrength,
  };
}

export function readSheetImages(spec: unknown): AssetRef[] | null {
  if (typeof spec !== "object" || spec === null || !("inputs" in spec)) return null;
  const { inputs } = spec;
  if (typeof inputs !== "object" || inputs === null || !("images" in inputs)) return null;
  if (!Array.isArray(inputs.images)) return null;
  const images = inputs.images.map(readAssetRef);
  return images.every((image) => image != null) ? images : null;
}

export function createIcLoraSheetDefinition(
  recipe: IcLoraSheetRecipe,
  settings?: IcLoraSheetCatalogSettings | null,
): FeatureDefinition<
  IcLoraSheetSchema,
  CreateIcLoraSheetRequest,
  IcLoraSheetContext,
  IcLoraSheetRecipe["id"]
> {
  const defaults = defaultsFor(recipe, settings);

  function fromGeneration(spec: unknown, context: IcLoraSheetContext): IcLoraSheetValues | null {
    const params = readGenerationParams(spec);
    const images = readSheetImages(spec);
    if (!params || !images) return null;
    return {
      images,
      aspectRatio: isVideoGenerationAspectRatio(params.aspectRatio)
        ? params.aspectRatio
        : defaults.aspectRatio,
      ...readSynthesizedCommonParams(params, context, defaults),
    };
  }

  function normalize(values: IcLoraSheetValues, context: IcLoraSheetContext): IcLoraSheetValues {
    return normalizeSynthesizedIcLoraValues<IcLoraSheetSchema>(values, context, defaults);
  }

  function validate(values: IcLoraSheetValues): ValidationIssue[] {
    const issues: ValidationIssue[] = [];
    if (values.images.length === 0) {
      issues.push({
        id: "images-required",
        fieldId: "images",
        message: "Add at least one image.",
        alwaysRevealed: true,
      });
    }
    if (values.prompt.trim() === "") {
      issues.push({
        id: "prompt-required",
        fieldId: "prompt",
        message: "Prompt is required.",
        alwaysRevealed: true,
      });
    }
    return issues;
  }

  function toCreateBody(values: IcLoraSheetValues, context?: IcLoraSheetContext): CreateIcLoraSheetRequest {
    if (values.images.length === 0) throw new Error("An image is required.");
    if (
      !isVideoGenerationAspectRatio(values.aspectRatio) ||
      !isSynthesizedIcLoraResolution(values.resolution)
    ) {
      throw new Error("A valid aspect ratio and resolution are required.");
    }
    const variantId = icLoraVariantId(values, context);
    return {
      contract_version: 1,
      inputs: { images: values.images.map(({ assetId }) => ({ assetId })) },
      params: {
        prompt: values.prompt.trim(),
        model: values.model,
        aspectRatio: values.aspectRatio,
        resolution: values.resolution,
        duration: values.duration,
        fps: values.fps,
        audioMode: values.audioMode,
        scale: values.loraStrength,
        ...(variantId ? { variantId } : {}),
        promptProvenance: "typed",
      },
    };
  }

  function form(values: IcLoraSheetValues, context: IcLoraSheetContext): FeatureFormModel<IcLoraSheetSchema> {
    const { aspectRatio, tail } = synthesizedIcLoraVideoOptionFields<IcLoraSheetSchema>(
      values,
      context,
    );
    return {
      fields: [
        {
          kind: "image-assets",
          id: "images",
          label: "Ingredients images",
          hint: `Add a ready-made sheet, or up to ${IC_LORA_SHEET_MAX_IMAGES} images and we'll stitch them into one.`,
          dataKey: "images",
          maxCount: IC_LORA_SHEET_MAX_IMAGES,
        },
        {
          kind: "textarea",
          id: "prompt",
          label: "Prompt",
          dataKey: "prompt",
          placeholder: recipe.placeholder,
        },
        aspectRatio,
        ...tail,
      ],
    };
  }

  function seedValues(context: IcLoraSheetContext): IcLoraSheetValues {
    return { ...withDefaultOffering(context.specs, defaults), images: context.seed ?? [] };
  }

  return {
    id: recipe.id,
    title: recipe.title,
    defaults,
    initialValues: seedValues,
    resetValues: seedValues,
    form,
    applyChange: (values, change, context) =>
      normalize(
        applyVideoSettingsChange<IcLoraSheetSchema>(
          values,
          change,
          context.specs,
          withDefaultOffering(context.specs, defaults),
        ),
        context,
      ),
    normalize,
    fromGeneration,
    validate,
    toCreateBody,
    // Generate stays off until an image and a prompt exist. The run refuses a blank prompt.
    isReady: (values, context) =>
      values.images.length > 0 &&
      values.prompt.trim() !== "" &&
      isLocalVideoSelectionReady(values, context.specs),
    isUnavailable: isSynthesizedIcLoraUnavailable,
  };
}
