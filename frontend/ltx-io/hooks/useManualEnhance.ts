import { useCallback, useEffect, useRef, useState } from "react";

import type { components } from "@/generated/backend-openapi";
import type { EnhanceProvider } from "@/hooks/use-prompt-enhancer-provider";
import { isEnhanceBlockedByMissingGeminiKey } from "@/lib/enhance-gemini-key";
import { resolveUsableEnhanceProvider } from "@/lib/enhance-provider";
import type { HomeFeatureId } from "@/lib/home-features";
import {
  icLoraCatalogId,
  isIcLoraRecipeId,
  isIcLoraSheetRecipeId,
} from "@/lib/ic-lora-recipes";
import { getLoraRecipe, isLoraRecipeId } from "@/lib/lora-recipes";
import type { PromptProvenance } from "@/lib/prompt-provenance";
import {
  type PromptHistoryEntry,
  appendEnhanceResult,
  resolvePromptProvenance,
  shouldApplyEnhanceResult,
} from "@/lib/prompt-provenance";

import { useExploreRuntime } from "../runtime/ExploreRuntime";
import { useFeatureFormStore } from "../stores/featureFormStore";
import {
  assetIdOf,
  enhanceImageAssetId,
  hasManualEnhanceInput,
  isEnhanceSourceCurrent,
  lookImageAssetId,
  panelImageAssetIds,
  replacedWindowFor,
  resolveShowManualEnhance,
} from "./manualEnhanceVisibility";

type EnhancePromptRequest = components["schemas"]["EnhancePromptRequest"];

export type PromptValues = {
  prompt?: string;
  startFrame?: { assetId: string } | null;
  endFrame?: { assetId: string } | null;
  // The source video of an IC-LoRA recipe, Extend, or Retake.
  video?: { assetId: string } | null;
  // The look image of an IC-LoRA recipe that takes one (Layout to Render, Restore).
  image?: { assetId: string } | null;
  // The images of an image-list recipe (Ingredients), in order.
  images?: readonly { assetId: string }[] | null;
  aspectRatio?: EnhancePromptRequest["aspectRatio"];
  resolution?: EnhancePromptRequest["resolution"];
  // The Retake selection, in seconds.
  startTime?: number;
  duration?: number;
};

type PromptEnhancerStatus = {
  exploreAutoEnhancePrompts: boolean;
  hasGeminiApiKey: boolean;
  localEnhancementSupported: boolean;
  defaultProvider: EnhanceProvider;
  canToggleProvider: boolean;
  showManualEnhance: boolean;
};

const MISSING_GEMINI_KEY_MESSAGE =
  "Add a Gemini API key in Desktop Settings, or use local Enhance when Gemma is downloaded.";

function numberOrUndefined(value: unknown): number | undefined {
  return typeof value === "number" ? value : undefined;
}

function liveFormState(featureId: HomeFeatureId): {
  prompt: string | undefined;
  history: PromptHistoryEntry[] | undefined;
  index: number | undefined;
  // null: the form has no source video. undefined: no live values to compare.
  videoAssetId: string | null | undefined;
  // The live start frame of Image-to-Video, Audio-to-Video, or Motion Track.
  startFrameAssetId: string | null | undefined;
  // The live look image of an IC-LoRA recipe. Same null and undefined meaning as the video.
  referenceImageAssetId: string | null | undefined;
  // The live panel images of an image-list recipe, in order. Empty for any other feature.
  panelAssetIds: string[] | undefined;
  // The live Retake selection. undefined for any other feature, or when the values are gone.
  replacedWindow: { start: number; end: number } | undefined;
} {
  const state = useFeatureFormStore.getState();
  const values = state.getValues(featureId);
  const stored = state.promptHistoryByFeature[featureId];
  return {
    prompt:
      values != null && "prompt" in values && typeof values.prompt === "string"
        ? values.prompt
        : undefined,
    history: stored?.history,
    index: stored?.index,
    videoAssetId: values == null ? undefined : assetIdOf("video" in values ? values.video : null),
    startFrameAssetId:
      values == null ? undefined : assetIdOf("startFrame" in values ? values.startFrame : null),
    referenceImageAssetId:
      values == null ? undefined : assetIdOf("image" in values ? values.image : null),
    panelAssetIds:
      values == null
        ? undefined
        : panelImageAssetIds({
            isSheetRecipe: isIcLoraSheetRecipeId(featureId),
            images: "images" in values && Array.isArray(values.images) ? values.images : null,
          }) ?? [],
    replacedWindow:
      values == null
        ? undefined
        : replacedWindowFor(featureId, {
            startTime: "startTime" in values ? numberOrUndefined(values.startTime) : undefined,
            duration: "duration" in values ? numberOrUndefined(values.duration) : undefined,
          }),
  };
}

export function useManualEnhance({
  featureId,
  values,
  promptHistory,
  historyIndex,
  onPromptHistoryChange,
  isCreating,
}: {
  featureId: HomeFeatureId;
  values: PromptValues;
  promptHistory: PromptHistoryEntry[];
  historyIndex: number;
  onPromptHistoryChange: (history: PromptHistoryEntry[], index: number) => void;
  isCreating: boolean;
}) {
  const {
    api,
    modelsVersion,
    persistEnhanceProviderPreference,
    readEnhanceProviderPreference,
    autoEnhancePrompts,
    onGeminiKeyRequired,
  } = useExploreRuntime();
  const [status, setStatus] = useState<PromptEnhancerStatus | null>(null);
  const [providerOverride, setProviderOverride] = useState<EnhanceProvider | null>(
    () => readEnhanceProviderPreference?.() ?? null,
  );

  useEffect(() => {
    let cancelled = false;
    void api.getPromptEnhancer().then((result) => {
      if (!cancelled && result.ok) {
        setStatus(result.data);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [api, modelsVersion]);

  const hasGeminiApiKey = status?.hasGeminiApiKey ?? false;
  const hasLocalTextEncoder = status?.localEnhancementSupported ?? false;
  const enhanceProvider = resolveUsableEnhanceProvider({
    preference: providerOverride,
    hasGeminiApiKey,
    hasLocalTextEncoder,
    fallback: status?.defaultProvider ?? "api",
  });
  const canToggleEnhanceProvider = status?.canToggleProvider ?? false;
  const showManualEnhance = resolveShowManualEnhance(
    autoEnhancePrompts,
    status?.showManualEnhance,
  );

  const setEnhanceProvider = useCallback(
    (provider: EnhanceProvider) => {
      setProviderOverride(provider);
      persistEnhanceProviderPreference?.(provider);
    },
    [persistEnhanceProviderPreference],
  );

  const [isEnhancing, setIsEnhancing] = useState(false);
  const [enhanceError, setEnhanceError] = useState<string | null>(null);

  const prompt = String(values.prompt ?? "");
  const hasEnhanceInput = hasManualEnhanceInput({
    prompt,
    startFrame: values.startFrame,
    video: values.video,
    images: values.images,
  });
  const promptRef = useRef(prompt);
  promptRef.current = prompt;

  const runEnhance = useCallback(async () => {
    if (isEnhancing || !hasEnhanceInput) return;
    setIsEnhancing(true);
    setEnhanceError(null);

    const sourcePrompt = prompt;
    const loraCatalogIds: string[] = [];
    const startFrameAssetId = values.startFrame?.assetId ?? null;
    const referenceImageAssetId = lookImageAssetId({
      isIcLoraRecipe: isIcLoraRecipeId(featureId),
      video: values.video,
      image: values.image,
    });
    const imageAssetId = enhanceImageAssetId({
      featureId,
      isIcLoraRecipe: isIcLoraRecipeId(featureId),
      startFrame: values.startFrame,
      video: values.video,
      image: values.image,
    });
    const lastImageAssetId =
      featureId === "image-to-video" ? values.endFrame?.assetId : undefined;
    const panelAssetIds = panelImageAssetIds({
      isSheetRecipe: isIcLoraSheetRecipeId(featureId),
      images: values.images,
    });
    const videoAssetId = values.video?.assetId;
    const sourceVideoAssetId = videoAssetId ?? null;
    const replacedWindow =
      videoAssetId === undefined
        ? undefined
        : replacedWindowFor(featureId, {
            startTime: values.startTime,
            duration: values.duration,
          });
    if (isLoraRecipeId(featureId)) {
      loraCatalogIds.push(getLoraRecipe(featureId).catalogId);
    }
    const icLoraId = isIcLoraRecipeId(featureId)
      ? icLoraCatalogId(featureId)
      : undefined;

    try {
      const result = await api.enhancePrompt({
        prompt: sourcePrompt,
        loraCatalogIds,
        icLoraId,
        imageAssetId,
        lastImageAssetId,
        imageAssetIds: panelAssetIds,
        ...(panelAssetIds !== undefined
          ? {
              aspectRatio: values.aspectRatio,
              resolution: values.resolution,
            }
          : {}),
        videoAssetId,
        replacedWindow,
        provider: enhanceProvider,
        mediaType: "video",
      });

      if (!result.ok) {
        if (
          result.error.code === "GEMINI_INVALID_API_KEY" ||
          result.error.code === "GEMINI_API_KEY_MISSING"
        ) {
          onGeminiKeyRequired?.();
          setEnhanceError(
            onGeminiKeyRequired ? result.error.message : MISSING_GEMINI_KEY_MESSAGE,
          );
        } else {
          setEnhanceError(result.error.message);
        }
        return;
      }

      const live = liveFormState(featureId);
      const history = live.history ?? promptHistory;
      const index = live.index ?? historyIndex;
      if (
        !shouldApplyEnhanceResult(live.prompt ?? promptRef.current, sourcePrompt)
      ) {
        return;
      }
      if (
        !isEnhanceSourceCurrent(live, {
          videoAssetId: sourceVideoAssetId,
          startFrameAssetId,
          referenceImageAssetId: referenceImageAssetId ?? null,
          panelAssetIds: panelAssetIds ?? [],
          replacedWindow,
        })
      ) {
        return;
      }

      const next = appendEnhanceResult({
        history,
        index,
        sourcePrompt,
        sourceProvenance: resolvePromptProvenance(
          sourcePrompt,
          history,
          index,
        ),
        enhancedPrompt: result.data.enhancedPrompt,
      });
      onPromptHistoryChange(next.history, next.index);
    } catch (error) {
      setEnhanceError(error instanceof Error ? error.message : "Enhance failed.");
    } finally {
      setIsEnhancing(false);
    }
  }, [
    api,
    enhanceProvider,
    featureId,
    hasEnhanceInput,
    historyIndex,
    isEnhancing,
    onGeminiKeyRequired,
    onPromptHistoryChange,
    prompt,
    promptHistory,
    values.aspectRatio,
    values.duration,
    values.endFrame?.assetId,
    values.image?.assetId,
    values.images,
    values.resolution,
    values.startFrame?.assetId,
    values.startTime,
    values.video?.assetId,
  ]);

  const canEnhance =
    showManualEnhance &&
    hasEnhanceInput &&
    !isCreating &&
    !isEnhancing &&
    (enhanceProvider === "api" ? hasGeminiApiKey : hasLocalTextEncoder);

  const enhanceBlockedByMissingGeminiKey = isEnhanceBlockedByMissingGeminiKey({
    enhanceAvailableForMode: showManualEnhance,
    enhanceProvider,
    hasGeminiApiKey,
    hasEnhanceInput,
  });

  const promptProvenance: PromptProvenance = resolvePromptProvenance(
    prompt,
    promptHistory,
    historyIndex,
  );

  return {
    showManualEnhance,
    canEnhance:
      canEnhance || Boolean(enhanceBlockedByMissingGeminiKey && onGeminiKeyRequired),
    enhanceBlockedByMissingGeminiKey,
    isEnhancing,
    enhanceError,
    runEnhance,
    promptProvenance,
    enhanceProvider,
    canToggleEnhanceProvider,
    setEnhanceProvider,
  };
}
