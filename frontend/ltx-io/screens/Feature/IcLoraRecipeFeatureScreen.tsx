import { useMemo } from "react";
import { Navigate } from "react-router";

import { useIcLoras } from "@/hooks/use-catalog";
import {
  getIcLoraRecipe,
  isIcLoraSheetRecipeId,
  isIcLoraTrackRecipeId,
  type IcLoraRecipe,
  type IcLoraRecipeId,
} from "@/lib/ic-lora-recipes";
import { paths } from "@/paths";

import cannyThumbnailUrl from "../../assets/video-guide/control-canny.jpeg";
import depthThumbnailUrl from "../../assets/video-guide/control-depth.jpeg";
import poseThumbnailUrl from "../../assets/video-guide/control-pose.jpeg";
import { useExploreRuntime } from "../../runtime/ExploreRuntime";

import {
  isPackagedIcLoraSeedName,
  planIcLoraRecipeSeed,
  readAppliedIcLoraSeedRevision,
  writeAppliedIcLoraSeedRevision,
} from "../../hooks/icLoraRecipeSeed";
import { useIcLoraImageSeed } from "../../hooks/useIcLoraImageSeed";
import { usePackagedSeedAsset } from "../../hooks/usePackagedSeedAsset";
import { useFeatureFlags } from "../../hooks/useFeatureFlags";
import { useCreateIcLoraRecipe } from "../../hooks/useCreateIcLoraRecipe";
import { CatalogInstallPreflight, type CatalogGenerateControls } from "./CatalogInstallPreflight";
import {
  createIcLoraRecipeDefinition,
  icLoraRecipeDefaults,
} from "./definitions/icLoraRecipe";
import { recipeHighResCells } from "./definitions/icLoraResolution";
import { icLoraRecipeFormPresentation } from "./IcLoraRecipeFormPresentation";
import { IC_LORA_RECIPE_EXAMPLE_MEDIA } from "./icLoraRecipeExampleMedia";
import { FeatureFormScreen } from "./FeatureFormScreen";
import { ExampleEmptyMedia } from "./results/ExampleEmptyMedia";
import { ResultFrame, type ResultFrameProps } from "./results/ResultFrame";
import { IcLoraSheetFeatureScreen } from "./IcLoraSheetFeatureScreen";
import { IcLoraTrackFeatureScreen } from "./IcLoraTrackFeatureScreen";
import { useVideoToVideoFeature } from "./VideoToVideoFeatureScreen";

const CONTROL_THUMBNAILS = {
  pose: poseThumbnailUrl,
  depth: depthThumbnailUrl,
  canny: cannyThumbnailUrl,
};

function CutoutResultFrame(props: ResultFrameProps) {
  return <ResultFrame {...props} isCutoutFeature />;
}

function IcLoraRecipeForm({
  recipe,
  controls,
}: {
  recipe: IcLoraRecipe;
  controls: CatalogGenerateControls;
}) {
  const { api } = useExploreRuntime();
  const { icLoras, catalogStatus } = useIcLoras(true, api);
  const flagsQuery = useFeatureFlags(api);
  const flags = flagsQuery.data;
  const highResCells = useMemo(
    () => recipeHighResCells(flags?.highResVideoToVideo === true),
    [flags?.highResVideoToVideo],
  );
  // The form normalizes a stored resolution against the offered cells. It must not do that
  // before the flag is known, or a saved 2160p would fall back on every start. A failed
  // flags request is not "known": it shows an error instead.
  const flagsSettled = flagsQuery.isSuccess;
  const catalogItem = useMemo(
    () => icLoras.find((entry) => entry.ic_lora.id === recipe.catalogId),
    [icLoras, recipe.catalogId],
  );
  // Stage 2 keeps the LoRA. That layout upscales a small clip and has no tiled cell.
  const defaultSettings = catalogItem?.ic_lora.default_settings;
  const twoStage =
    defaultSettings?.stage_2_ic_lora === true && defaultSettings.skip_stage_2 !== true;
  const allowsEmptyPrompt = catalogItem?.ic_lora.allows_empty_prompt ?? false;
  const settings = icLoraRecipeDefaults({
    lora_strength: controls.defaultLoraStrength,
    audio_mode: controls.defaultAudioMode,
  });
  const definition = useMemo(
    () =>
      createIcLoraRecipeDefinition(recipe, {
        lora_strength: settings.loraStrength,
        audio_mode: settings.audioMode,
        allows_empty_prompt: allowsEmptyPrompt,
      }),
    [recipe, settings.audioMode, settings.loraStrength, allowsEmptyPrompt],
  );
  const createGeneration = useCreateIcLoraRecipe(recipe.id);
  const seedKind = recipe.seed.kind;
  // The form waits for the look image seed, so the first stored values include it.
  const imageSeed = useIcLoraImageSeed(recipe);
  const useSeed = (options?: { enabled?: boolean }) =>
    usePackagedSeedAsset(seedKind, {
      enabled: (options?.enabled ?? true) && imageSeed.settled,
      revision: recipe.seed.revision,
      media: "video",
    });
  const ExampleEmpty = useMemo(() => {
    const media = IC_LORA_RECIPE_EXAMPLE_MEDIA[recipe.id];
    return function RecipeExampleEmpty() {
      return (
        <ExampleEmptyMedia posterUrl={media.posterUrl} videoUrl={media.videoUrl} />
      );
    };
  }, [recipe.id]);
  const screen = useVideoToVideoFeature({
    featureId: recipe.id,
    definition,
    presentation: icLoraRecipeFormPresentation,
    EmptyStateAdapter: ExampleEmpty,
    createGeneration,
    useSeed,
    planSeed: (input) =>
      planIcLoraRecipeSeed(recipe, {
        ...input,
        appliedSeedRevision: readAppliedIcLoraSeedRevision(recipe),
        packagedSeedRevision: recipe.seed.revision,
      }),
    isPackagedSeedName: (name) => isPackagedIcLoraSeedName(recipe, name),
    writeAppliedRevision: () => writeAppliedIcLoraSeedRevision(recipe),
    seedValues: (context) =>
      definition.initialValues?.({ ...context, imageSeed: imageSeed.data ?? null }) ??
      definition.defaults,
  });
  // A failed look image must show an error. The form needs it to run.
  const resourceError = screen.resourceError ?? imageSeed.error ?? flagsQuery.error ?? null;
  // A recipe that needs the flag has no page without it. The backend refuses it too.
  if (recipe.requiresHighRes === true && flagsSettled && flags?.highResVideoToVideo !== true) {
    return <Navigate to={paths.home} replace />;
  }
  // The offered cells depend on the stage layout. A stored cell must not be normalized before
  // the catalog says it. A failed catalog keeps the single-stage rule.
  const catalogSettled = catalogStatus !== "loading";
  const contextReady = screen.contextReady && imageSeed.settled && flagsSettled && catalogSettled;
  if (!contextReady && resourceError == null) return null;
  return (
    <FeatureFormScreen
      {...screen}
      contextReady={contextReady}
      resourceError={resourceError}
      context={{
        ...screen.context,
        catalogVariants: controls.variants,
        supportedModels: catalogItem?.ic_lora.supported_models,
        imageSeed: imageSeed.data ?? null,
        controlThumbnails: CONTROL_THUMBNAILS,
        referenceImageRequired: catalogItem?.ic_lora.reference_image_required,
        highResCells,
        twoStage,
      }}
      ResultFrameAdapter={recipe.cutout ? CutoutResultFrame : ResultFrame}
      generateLabel={controls.generateLabel}
      generateHint={controls.generateHint}
      generateBusy={controls.generateBusy}
      generateEnabled={controls.generateEnabled && screen.generateEnabled}
      actionError={controls.actionError}
      interceptGenerate={controls.interceptGenerate}
      onCancelGenerate={controls.onCancelGenerate}
    />
  );
}

export function IcLoraRecipeFeatureScreen({ recipeId }: { recipeId: IcLoraRecipeId }) {
  if (isIcLoraSheetRecipeId(recipeId)) return <IcLoraSheetFeatureScreen recipeId={recipeId} />;
  if (isIcLoraTrackRecipeId(recipeId)) return <IcLoraTrackFeatureScreen recipeId={recipeId} />;
  const recipe = getIcLoraRecipe(recipeId);
  return (
    <CatalogInstallPreflight
      kind="ic-lora"
      catalogId={recipe.catalogId}
      title={recipe.title}
      hold={false}
      renderForm={(controls) => <IcLoraRecipeForm recipe={recipe} controls={controls} />}
    />
  );
}
