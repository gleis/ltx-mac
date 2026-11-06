import { useExploreRuntime } from "../runtime/ExploreRuntime.tsx";
import { useFeatureFlags } from "./useFeatureFlags.ts";

/**
 * The `highResVideoToVideo` flag, on Desktop and on Remote. It is false until the backend answers,
 * so a feature that needs the flag stays hidden until the flag is known.
 */
export function useHighResVideoToVideoEnabled(): boolean {
  const { api } = useExploreRuntime();
  return useFeatureFlags(api).data?.highResVideoToVideo === true;
}
