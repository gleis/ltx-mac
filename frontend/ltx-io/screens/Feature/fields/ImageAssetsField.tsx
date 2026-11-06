import { Text } from "@ds/Text/Text";
import MediaReplaceIcon from "@ds/assets/Icons/MediaReplace.svg?react";
import PhotosIcon from "@ds/assets/Icons/Photos.svg?react";
import RemoveIcon from "@ds/assets/Icons/Remove.svg?react";
import { useQueryClient } from "@tanstack/react-query";
import { clsx } from "clsx";
import { useEffect, useRef, useState } from "react";

import { useUploadDnd } from "../../../components/shared/Upload/hooks/useUploadDnd";
import { assetQueryKeys, isImageAsset } from "../../../hooks/assetQueryKeys";
import { type PendingImport, useExploreRuntime } from "../../../runtime/ExploreRuntime";
import formStyles from "../FeatureFormView.module.scss";
import { type AssetRef, type ValidationIssue, toDurableAssetRef } from "../types";
import {
  IMAGE_ACCEPT,
  IMAGE_EXTENSIONS,
  imageIngestErrorMessage,
  imageLookupErrorMessage,
  isSupportedDroppedImage,
  UNSUPPORTED_IMAGE_MESSAGE,
} from "./imageAssetInput";
import { collectImageBatch, describeImageBatch, planImageBatch } from "./imageBatch";
import dropStyles from "./ImageDropzone.module.scss";
import styles from "./ImageAssetsField.module.scss";
import { useMediaAssetField } from "./useMediaAssetField";
import { OverlayButton } from "./OverlayButton";

/** One image slot: the single-image field logic with the image options filled in. */
function useImageSlot(label: string, value: AssetRef | null, onChange: (value: AssetRef | null) => void) {
  return useMediaAssetField({
    value,
    onChange,
    accept: IMAGE_ACCEPT,
    extensions: IMAGE_EXTENSIONS,
    chooserTitle: label,
    isSupportedDropped: isSupportedDroppedImage,
    isExpectedAsset: isImageAsset,
    unsupportedMessage: UNSUPPORTED_IMAGE_MESSAGE,
    ingestErrorMessage: imageIngestErrorMessage,
    lookupErrorMessage: imageLookupErrorMessage,
  });
}

function Thumb({
  label,
  value,
  onChange,
}: {
  label: string;
  value: AssetRef;
  onChange: (value: AssetRef | null) => void;
}) {
  const slot = useImageSlot(label, value, onChange);
  return (
    <div className={styles.thumb} title={slot.errorMessage ?? undefined}>
      <input ref={slot.fileInputRef} className={dropStyles.hiddenInput} type="file" accept={slot.accept} />
      {slot.previewUrl ? <img className={styles.thumbImage} src={slot.previewUrl} alt="" /> : null}
      <div className={styles.thumbActions}>
        <OverlayButton
          label="Replace"
          ariaLabel="Replace image"
          icon={<MediaReplaceIcon />}
          onClick={slot.choose}
        />
        <OverlayButton
          label="Remove"
          ariaLabel="Remove image"
          icon={<RemoveIcon />}
          onClick={slot.clear}
        />
      </div>
    </div>
  );
}

/**
 * A list of images in the ltx.io style: one large drop zone while the list is empty,
 * then a grid with an add tile first and one thumbnail for each image.
 */
export function ImageAssetsField({
  field,
  value,
  maxCount,
  issue,
  onChange,
}: {
  field: { id: string; label: string; hint?: string };
  value: AssetRef[];
  maxCount: number;
  issue?: ValidationIssue;
  onChange: (value: AssetRef[]) => void;
}) {
  const { mediaInput } = useExploreRuntime();
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [addError, setAddError] = useState<string | null>(null);
  // An ingest ends after this render. It must add to the list as it is then.
  const latest = useRef({ value, onChange });
  useEffect(() => {
    latest.current = { value, onChange };
  });

  // Add a whole batch with one `onChange`. Separate adds would each start from the same old list.
  const addImports = async (pending: PendingImport[], unsupported: number) => {
    setAddError(null);
    const plan = planImageBatch(pending, maxCount - latest.current.value.length);
    const results = await Promise.allSettled(plan.take.map((run) => run()));
    const batch = collectImageBatch(results, isImageAsset);
    for (const asset of batch.assets) {
      queryClient.setQueryData(assetQueryKeys.detail(asset.id), asset);
    }
    // Another add can finish first, so count the free slots again now.
    const { value: current, onChange: change } = latest.current;
    const accepted = batch.assets.slice(0, Math.max(0, maxCount - current.length));
    if (accepted.length > 0) {
      change([...current, ...accepted.map(toDurableAssetRef)]);
    }
    setAddError(
      describeImageBatch({
        unsupported: unsupported + batch.wrongKind,
        overLimit: plan.overLimit + (batch.assets.length - accepted.length),
        maxCount,
        failures: batch.failures,
      }),
    );
  };

  const dropFiles = (files: File[]) => {
    const supported = files.filter(isSupportedDroppedImage);
    void addImports(
      supported.map((file) => () => mediaInput.ingestDroppedFile(file)),
      files.length - supported.length,
    );
  };

  const pickFiles = () => {
    setAddError(null);
    mediaInput
      .chooseMany({
        title: field.label,
        accept: IMAGE_ACCEPT,
        extensions: IMAGE_EXTENSIONS,
        fileInput: fileInputRef.current,
      })
      .then((pending) => addImports(pending, 0))
      .catch((error: unknown) => setAddError(imageIngestErrorMessage(error)));
  };

  const upload = useUploadDnd({
    isMultiple: true,
    acceptedFilesTypes: IMAGE_ACCEPT,
    onUpload: dropFiles,
  });
  const dropProps = {
    onDragEnter: upload.handlers.handleDragEnter,
    onDragOver: upload.handlers.handleDragOver,
    onDragLeave: upload.handlers.handleDragLeave,
    onDrop: upload.handlers.handleDrop,
    onClick: pickFiles,
  };
  const message = addError ?? issue?.message;
  return (
    <div className={formStyles.field}>
      <div className={styles.labelGroup}>
        <Text as="span" variant="body" size="md" className={formStyles.fieldLabel}>
          {field.label}
        </Text>
        {field.hint ? (
          <Text as="span" variant="body" size="sm" className={styles.hint}>
            {field.hint}
          </Text>
        ) : null}
      </div>
      <input
        ref={fileInputRef}
        className={dropStyles.hiddenInput}
        type="file"
        accept={IMAGE_ACCEPT}
        multiple
      />
      {value.length === 0 ? (
        <button
          type="button"
          className={clsx(dropStyles.dropzone, upload.isDragOver && dropStyles.dragOver)}
          {...dropProps}
        >
          <span className={dropStyles.empty}>
            <PhotosIcon className={dropStyles.emptyIcon} aria-hidden />
            <Text as="span" variant="label" size="md">
              Add images
            </Text>
            {field.hint ? null : (
              <Text as="span" variant="body" size="md" className={styles.subtitle}>
                Upload up to {maxCount} images of characters, props, and location
              </Text>
            )}
          </span>
        </button>
      ) : (
        <div className={styles.grid}>
          {value.length < maxCount ? (
            <button
              type="button"
              className={clsx(styles.addTile, upload.isDragOver && dropStyles.dragOver)}
              aria-label="Add image"
              {...dropProps}
            >
              +
            </button>
          ) : null}
          {value.map((image, index) => (
            <Thumb
              // Two entries can hold the same asset, so the index is the key.
              key={index}
              label={field.label}
              value={image}
              onChange={(next) => {
                // The old add message is about a list that no longer exists.
                setAddError(null);
                onChange(
                  next == null
                    ? value.filter((_, at) => at !== index)
                    : value.map((current, at) => (at === index ? next : current)),
                );
              }}
            />
          ))}
        </div>
      )}
      {message ? <span className={formStyles.fieldError}>{message}</span> : null}
    </div>
  );
}
