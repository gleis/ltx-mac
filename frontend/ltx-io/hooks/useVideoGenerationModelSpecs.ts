import { keepPreviousData, useQuery } from "@tanstack/react-query";

import { useExploreRuntime } from "../runtime/ExploreRuntime";
import {
  fetchVideoGenerationModelSpecs,
  generationQueryKeys,
  VIDEO_GENERATION_MODEL_SPECS_QUERY_OPTIONS,
} from "./generationQueryKeys";

export function useVideoGenerationModelSpecs() {
  const { api, modelsVersion, budgetMode, budgetModeSaving } = useExploreRuntime();
  const query = useQuery({
    queryKey: generationQueryKeys.specs(modelsVersion, budgetMode ?? "server"),
    queryFn: () => fetchVideoGenerationModelSpecs(api),
    placeholderData: keepPreviousData,
    ...VIDEO_GENERATION_MODEL_SPECS_QUERY_OPTIONS,
  });
  const budgetModeReady =
    budgetMode == null ||
    (!budgetModeSaving && query.data?.budgetMode === budgetMode);
  return {
    data: query.data,
    error: query.error,
    isError: query.isError,
    isFetching: query.isFetching,
    isPending: query.isPending,
    isSuccess: query.isSuccess,
    refetch: query.refetch,
    status: query.status,
    budgetModeReady,
  };
}
