import {
  preflightCellStatus,
  readBudgetMode,
  type PreflightCellStatus,
} from "../../lib/budget-risk.ts";
import type { components } from "../../generated/backend-openapi";

export type EditPreflightResponse =
  components["schemas"]["EditPreflightResponse"];
export type EditPreflightCell = components["schemas"]["EditPreflightCell"];
export type EditPreflightResolution =
  components["schemas"]["EditPreflightResolution"];

export type EditPreflightFeature = "retake" | "extend";

/**
 * Preflight verdicts as carried on the video feature context. The screen only
 * sets this while the verdicts match the selected video (otherwise null), so
 * definitions read it as fresh-or-absent. Unknown (null, or a cell/row that is
 * absent) always means allowed: the create-time preflight stays authoritative.
 */
export type EditPreflightState = {
  assetId: string;
  response: EditPreflightResponse;
};

/** Request grid: 0.5s keeps trim drags from refetching every pixel. */
export function roundPreflightSeconds(value: number): number {
  return Math.round(value * 2) / 2;
}

export function findPreflightCell(
  preflight: EditPreflightState | null | undefined,
  key: string,
  duration: number,
): EditPreflightCell | undefined {
  return preflight?.response.cells.find(
    (cell) => cell.key === key && cell.duration === duration,
  );
}

export function preflightResolutionRow(
  preflight: EditPreflightState | null | undefined,
  key: string,
): EditPreflightResolution | undefined {
  return preflight?.response.resolutions.find((row) => row.key === key);
}

/**
 * Fallback when a failing cell carries no message. The backend always sends
 * the exact refusal Generate would return; this covers wire skew only.
 */
export const PREFLIGHT_TOO_LARGE_MESSAGE = "Too large for this machine.";

/** Failing cells fall back to "too large". */
export function preflightFallbackMessage(status: PreflightCellStatus): string {
  if (status.status === "clear") return "";
  return status.message || PREFLIGHT_TOO_LARGE_MESSAGE;
}

export function preflightStatus(
  preflight: EditPreflightState | null | undefined,
  cell: EditPreflightCell | null | undefined,
): PreflightCellStatus {
  return preflightCellStatus(
    cell,
    readBudgetMode(preflight?.response.budgetMode),
  );
}

/**
 * Resolution-level status from the minimum-duration cell. Unknown stays clear
 * so the create-time preflight remains authoritative.
 */
export function resolutionPreflightStatus(
  preflight: EditPreflightState | null | undefined,
  key: string,
  minDuration: number,
): PreflightCellStatus {
  return preflightStatus(preflight, findPreflightCell(preflight, key, minDuration));
}

/**
 * Longest duration that passes at this tier, for trim clamps. Null while
 * unknown — callers fall back to the video length.
 */
export function preflightMaxDuration(
  preflight: EditPreflightState | null | undefined,
  key: string,
): number | null {
  return preflightResolutionRow(preflight, key)?.maxDurationSeconds ?? null;
}

/**
 * Fresh-or-absent gate: verdicts for another video never reach the
 * definitions, so a new selection never wears the old video's disables.
 */
export function selectFreshPreflight(
  result: EditPreflightState | null,
  assetId: string | null,
): EditPreflightState | null {
  if (result == null || assetId == null || result.assetId !== assetId) {
    return null;
  }
  return result;
}

/**
 * Trim clamp at the current tier. Null while unknown, or when the max sits
 * below the minimum (nothing passes there — the slider stays whole and the
 * issue explains, since TrimRange cannot take max < min).
 */
export function selectTrimMaxSeconds(
  preflight: EditPreflightState | null | undefined,
  resolutionKey: string,
  minDuration: number,
): number | null {
  const max = preflightMaxDuration(preflight, resolutionKey);
  if (max == null || max < minDuration) return null;
  return max;
}

/**
 * Highest tier (in display order) whose minimum-duration cell passes. Null
 * while unknown or when nothing passes — callers keep the current selection.
 */
export function firstPassingResolutionKey<T extends string>(
  preflight: EditPreflightState | null | undefined,
  keysInDisplayOrder: readonly T[],
  minDuration: number,
): T | null {
  if (preflight == null) return null;
  return (
    keysInDisplayOrder.find((key) => {
      const cell = findPreflightCell(preflight, key, minDuration);
      return cell != null && preflightStatus(preflight, cell).status !== "blocked";
    }) ?? null
  );
}
