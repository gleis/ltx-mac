import { AlertTriangle } from "lucide-react";

import { Button } from "@ds/Button/Button";
import { Text } from "@ds/Text/Text";

import { Modal } from "../../components/shared/Modal/Modal";
import { T3_CONFIRM_PAGING, T3_CONFIRM_TITLE } from "@/lib/budget-risk";

import styles from "./RiskyGenerateDialog.module.scss";

export function RiskyGenerateDialog({
  message,
  onConfirm,
  onAdjust,
}: {
  message: string;
  onConfirm: () => void;
  onAdjust: () => void;
}) {
  return (
    <Modal onClose={onAdjust} variant="secondary">
      <div className={styles.body}>
        <AlertTriangle className={styles.icon} aria-hidden />
        <Text as="h2" variant="heading" size="sm">
          {T3_CONFIRM_TITLE}
        </Text>
        <Text as="p" variant="body" size="md">
          {message}
        </Text>
        <Text as="p" variant="body" size="md">
          {T3_CONFIRM_PAGING}
        </Text>
        <div className={styles.actions}>
          <Button
            appearance="neutral"
            hierarchy="secondary"
            label="Adjust settings"
            onClick={onAdjust}
          />
          <Button
            appearance="brand"
            hierarchy="primary"
            label="Generate anyway"
            onClick={onConfirm}
          />
        </div>
      </div>
    </Modal>
  );
}
