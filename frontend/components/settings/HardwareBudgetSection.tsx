import type { AppSettings } from "../../contexts/AppSettingsContext";

import { SettingToggle } from "./SettingToggle";
import { SettingsSubsection } from "./SettingsContent";

export function HardwareBudgetSection({
  settings,
  onToggleReckless,
  saving = false,
  syncError,
}: {
  settings: Pick<AppSettings, "budgetMode">;
  onToggleReckless: () => void;
  saving?: boolean;
  syncError?: string | null;
}) {
  return (
    <SettingsSubsection
      showDivider
      title="Hardware budget"
      description="The switch moves when the save finishes."
      alert={syncError}
    >
      <SettingToggle
        rowLayout="preference"
        title="Reckless mode"
        description="Allow over-budget and unmeasured jobs with a warning instead of blocking them. Jobs may fail, run out of memory, or stall the app."
        enabled={settings.budgetMode === "reckless"}
        disabled={saving}
        onToggle={onToggleReckless}
      />
    </SettingsSubsection>
  );
}
