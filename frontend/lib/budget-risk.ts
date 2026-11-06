import { budgetTooltip, type BudgetReadout } from "./budget-figure.ts";

/** Reckless-mode warning tier. The backend decides the tier; this module only renders it. */

export type BudgetMode = "standard" | "reckless";

export type WarningTone = "caveat" | "risk";

/** Tight is a kept cell. Over and unmeasured are the tiers that block in standard mode. */
export type SelectionTier = "tight" | "over" | "unmeasured";

export type SelectionRisk = {
  tier: SelectionTier;
  message: string;
  readout?: BudgetReadout | null;
};

const TIER_RANK: Record<SelectionTier, number> = {
  tight: 1,
  over: 2,
  unmeasured: 3,
};

export function readBudgetMode(value: string | null | undefined): BudgetMode {
  return value === "reckless" ? "reckless" : "standard";
}

/**
 * Tier of the warning a refusal becomes when reckless mode turns it into a
 * selectable option. Null for a hard refusal.
 */
export function rejectReasonTier(reason: string): "over" | "unmeasured" | null {
  if (reason === "stream_over" || reason === "ram_shortfall") return "over";
  if (reason === "unmeasured_cell") return "unmeasured";
  return null;
}

/** Measurement refusals reckless mode may turn into a warned, selectable option. */
export function isDowngradableRejectReason(reason: string): boolean {
  return rejectReasonTier(reason) !== null;
}

/**
 * A failing preflight cell still blocks unless reckless mode downgraded it.
 * Cells with no tier are hard refusals in every mode.
 */
export function preflightCellBlocks(
  cell: { ok: boolean; tier?: "over" | "unmeasured" | null },
  budgetMode: BudgetMode,
): boolean {
  if (cell.ok) return false;
  return !(
    budgetMode === "reckless" &&
    (cell.tier === "over" || cell.tier === "unmeasured")
  );
}

export type PreflightCellStatus =
  | { status: "clear" }
  | {
      status: "warned";
      message: string | null;
      tier: "over" | "unmeasured";
      readout: BudgetReadout | null;
    }
  | { status: "blocked"; message: string | null; readout: BudgetReadout | null };

/** A downgraded failure is warned. A passing cell with a confirmable tier is warned only in reckless mode. */
export function preflightCellStatus(
  cell:
    | {
        ok: boolean;
        message?: string | null;
        tier?: "over" | "unmeasured" | null;
        readout?: BudgetReadout | null;
      }
    | null
    | undefined,
  budgetMode: BudgetMode,
): PreflightCellStatus {
  if (cell == null) return { status: "clear" };
  const message = cell.message ?? null;
  const readout = cell.readout ?? null;
  if (cell.ok) {
    if (
      budgetMode === "reckless" &&
      (cell.tier === "over" || cell.tier === "unmeasured")
    ) {
      return { status: "warned", message, tier: cell.tier, readout };
    }
    return { status: "clear" };
  }
  if (!preflightCellBlocks(cell, budgetMode)) {
    return {
      status: "warned",
      message,
      tier: cell.tier === "unmeasured" ? "unmeasured" : "over",
      readout,
    };
  }
  return { status: "blocked", message, readout };
}

/** Over and unmeasured ask before generate. Tight stays on the option icon. */
export function isConfirmableBudgetTier(
  tier: string | null | undefined,
): tier is "over" | "unmeasured" {
  return tier === "over" || tier === "unmeasured";
}

export function hasDowngradableDrop(
  cells: readonly { audio: boolean; reason: string }[] | null | undefined,
  audio?: boolean,
): boolean {
  return (cells ?? []).some(
    (cell) =>
      (audio === undefined || cell.audio === audio) &&
      isDowngradableRejectReason(cell.reason),
  );
}

export function worseRisk(
  current: SelectionRisk | null,
  next: SelectionRisk | null,
): SelectionRisk | null {
  if (current == null) return next;
  if (next == null) return current;
  return TIER_RANK[next.tier] > TIER_RANK[current.tier] ? next : current;
}

/** Risk outranks a quality caveat. One icon, risk line first. */
export function riskOptionFields(
  risk: string | undefined,
  caveat: string | undefined,
): { warning?: string; warningTone?: WarningTone } {
  if (risk != null && caveat != null) {
    return { warning: `${risk} ${caveat}`, warningTone: "risk" };
  }
  if (risk != null) return { warning: risk, warningTone: "risk" };
  if (caveat != null) return { warning: caveat };
  return {};
}

export const RECKLESS_FAILURE_BACKSTOP = "This job ran in Reckless mode.";

export const RECKLESS_REMOTE_ATTRIBUTION =
  "Reckless mode is on for your Desktop — this job may fail or stall.";

export const NOTHING_FITS_STRIP =
  "Nothing fits this machine — every option may fail or stall.";

export const T2_SESSION_NOTICE =
  "Over-budget jobs are allowed while Reckless mode is on.";

export const T3_CONFIRM_TITLE = "Generate anyway?";

export const T3_CONFIRM_PAGING =
  "This can hang the app while paging. You can stop it once it starts running.";

export function generateHintForTier(
  tier: SelectionTier | null | undefined,
): string | undefined {
  if (tier === "over") return "May exceed your GPU memory.";
  if (tier === "unmeasured") return "Unmeasured on your hardware.";
  return undefined;
}

export function queueRiskTooltip(
  message: string,
  readout?: BudgetReadout | null,
): string {
  const line = readout ? budgetTooltip(readout) : message;
  return `Queued in Reckless mode — ${line}`;
}

export function withRecklessBackstop(body: string, ranReckless: boolean): string {
  if (!ranReckless) return body;
  return `${body} ${RECKLESS_FAILURE_BACKSTOP}`;
}
