import { T2_SESSION_NOTICE } from "@/lib/budget-risk";

import styles from "./RecklessT2Notice.module.scss";

/**
 * Shown the first time you generate an over-budget setup this session.
 * Got it starts that generation and is not asked again until the app restarts.
 * Turning the mode off stays on the header pill.
 */
export function RecklessT2Notice({
  onAcknowledge,
}: {
  onAcknowledge: () => void;
}) {
  return (
    <p className={styles.notice} role="status">
      <span>{T2_SESSION_NOTICE}</span>
      <button type="button" className={styles.action} onClick={onAcknowledge}>
        Got it
      </button>
    </p>
  );
}
