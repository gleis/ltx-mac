import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { preflightCellStatus } from "./budget-risk.ts";

describe("preflightCellStatus", () => {
  it("warns a passing unmeasured cell only while reckless is on", () => {
    const cell = { ok: true, tier: "unmeasured" as const, message: null };
    assert.equal(preflightCellStatus(cell, "standard").status, "clear");
    const warned = preflightCellStatus(cell, "reckless");
    assert.equal(warned.status, "warned");
    if (warned.status === "warned") assert.equal(warned.tier, "unmeasured");
  });

  it("keeps a failing cell blocked in standard and warned in reckless", () => {
    const cell = {
      ok: false,
      tier: "over" as const,
      message: "needs 18",
    };
    assert.equal(preflightCellStatus(cell, "standard").status, "blocked");
    assert.equal(preflightCellStatus(cell, "reckless").status, "warned");
  });
});
