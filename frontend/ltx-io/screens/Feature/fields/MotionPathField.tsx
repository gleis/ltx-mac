import { Text } from "@ds/Text/Text";
import { useMemo, useState } from "react";

import { isImageAsset } from "../../../hooks/assetQueryKeys";
import formStyles from "../FeatureFormView.module.scss";
import { MotionPathEditModal } from "../motionPath/MotionPathEditModal";
import { MotionPathRefTile } from "../motionPath/MotionPathRefTile";
import { useMotionTrackFrameAspect } from "../motionPath/useMotionTrackFrameAspect";
import type { MotionPath } from "../../../lib/motionPathSampler";
import type { AssetRef, FieldOption, MotionPathFieldEdit, ValidationIssue } from "../types";
import {
  IMAGE_ACCEPT,
  IMAGE_EXTENSIONS,
  imageIngestErrorMessage,
  imageLookupErrorMessage,
  isSupportedDroppedImage,
  UNSUPPORTED_IMAGE_MESSAGE,
} from "./imageAssetInput";
import dropStyles from "./ImageDropzone.module.scss";
import { useMediaAssetField } from "./useMediaAssetField";

/**
 * The reference image of a Motion Track run with the path drawn on it, as on ltx.io.
 * The tile shows the image and the path. A click opens the two-step path editor.
 * A new image opens the editor at once. The definition clears the old path.
 */
export function MotionPathField({
  field,
  image,
  paths,
  duration,
  issue,
  onChange,
}: {
  field: { id: string; label: string; durationOptions: FieldOption<number>[] };
  image: AssetRef | null;
  paths: MotionPath[];
  duration: number;
  issue?: ValidationIssue;
  onChange: (edit: MotionPathFieldEdit) => void;
}) {
  const [isEditing, setIsEditing] = useState(false);
  const media = useMediaAssetField({
    value: image,
    issue,
    onChange: (nextImage) => {
      onChange({ image: nextImage });
      setIsEditing(nextImage != null);
    },
    accept: IMAGE_ACCEPT,
    extensions: IMAGE_EXTENSIONS,
    chooserTitle: field.label,
    isSupportedDropped: isSupportedDroppedImage,
    isExpectedAsset: isImageAsset,
    unsupportedMessage: UNSUPPORTED_IMAGE_MESSAGE,
    ingestErrorMessage: imageIngestErrorMessage,
    lookupErrorMessage: imageLookupErrorMessage,
  });
  const { previewUrl } = media;
  // The one place that loads the image to read its frame. The tile and the editor get the result.
  const imageAspect = useMotionTrackFrameAspect(previewUrl ?? undefined);
  const tracks = useMemo(() => paths.map((path) => path.points), [paths]);

  return (
    <div className={formStyles.field}>
      <Text as="span" variant="body" size="md" className={formStyles.fieldLabel}>
        {field.label}
      </Text>
      <input
        ref={media.fileInputRef}
        className={dropStyles.hiddenInput}
        type="file"
        accept={media.accept}
      />
      <div
        onDragEnter={media.dragHandlers.handleDragEnter}
        onDragOver={media.dragHandlers.handleDragOver}
        onDragLeave={media.dragHandlers.handleDragLeave}
        onDrop={media.dragHandlers.handleDrop}
      >
        <MotionPathRefTile
          tracks={tracks}
          imageUrl={previewUrl}
          imageAspect={imageAspect}
          onEdit={() => setIsEditing(true)}
          onUpload={media.choose}
          onReplace={previewUrl ? media.choose : undefined}
        />
      </div>
      {media.errorMessage ? (
        <span className={formStyles.fieldError}>{media.errorMessage}</span>
      ) : null}
      {isEditing && previewUrl ? (
        <MotionPathEditModal
          imageUrl={previewUrl}
          imageAspect={imageAspect}
          paths={paths}
          durationSec={duration}
          durationOptions={field.durationOptions
            .filter((option) => option.disabledReason == null)
            .map((option) => option.value)}
          onAccept={(nextPaths, durationSec) => {
            onChange({ paths: nextPaths, duration: durationSec });
            setIsEditing(false);
          }}
          onClose={() => setIsEditing(false)}
        />
      ) : null}
    </div>
  );
}
