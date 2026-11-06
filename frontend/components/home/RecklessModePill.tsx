import { AlertTriangle } from "lucide-react";

import { useAppSettings } from "../../contexts/AppSettingsContext";

import styles from "./RecklessModePill.module.scss";

/** Header chip while Reckless mode is on, plus the last failed toggle. */
export function RecklessModePill() {
  const { settings, setBudgetMode, budgetModeSaving, settingsSyncError } = useAppSettings();
  const showPill = settings.budgetMode === "reckless";
  if (!showPill && settingsSyncError == null) return null;
  return (
    <span className={styles.pill}>
      {showPill ? <AlertTriangle className={styles.icon} aria-hidden /> : null}
      {showPill ? (
        <span role="status">Reckless mode is on — over-budget jobs are allowed.</span>
      ) : null}
      {settingsSyncError ? <span>{settingsSyncError}</span> : null}
      {showPill ? (
        <button
          type="button"
          className={styles.action}
          disabled={budgetModeSaving}
          onClick={() => void setBudgetMode("standard")}
        >
          Turn off
        </button>
      ) : null}
    </span>
  );
}
