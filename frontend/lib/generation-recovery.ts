import { ApiClient } from './api-client'
import { addVisualAssetToProject } from './asset-copy'
import { logger } from './logger'
import type { GenerationSettings } from '../components/SettingsPanel'
import type { Asset } from '../types/project-model'

export const GENERATION_RECOVERY_KEY = 'ltx-generation-recovery'

export interface GenerationRecoveryContext {
  projectId: string
  prompt: string
  settings?: GenerationSettings
  inputImageUrl?: string
  inputAudioUrl?: string
  genType?: 'video' | 'image'
  baselineId: string | null
  generationId?: string
}

export interface RecoveryImporterApi {
  addAsset: (projectId: string, asset: Omit<Asset, 'id' | 'createdAt'>) => unknown
}

let activeOwnerProjectId: string | null = null

export function setActiveGenerationOwner(projectId: string | null): void {
  activeOwnerProjectId = projectId
}

export function clearGenerationRecoveryMarker(projectId?: string): void {
  if (!projectId) {
    localStorage.removeItem(GENERATION_RECOVERY_KEY)
    return
  }
  const saved = localStorage.getItem(GENERATION_RECOVERY_KEY)
  if (!saved) return
  try {
    const ctx = JSON.parse(saved) as Partial<GenerationRecoveryContext>
    if (ctx.projectId === projectId) localStorage.removeItem(GENERATION_RECOVERY_KEY)
  } catch {
    localStorage.removeItem(GENERATION_RECOVERY_KEY)
  }
}

export async function writeGenerationRecoveryMarker(
  ctx: Omit<GenerationRecoveryContext, 'baselineId' | 'generationId'>,
): Promise<void> {
  const progress = await ApiClient.getGenerationProgress()
  const baselineId = progress.ok ? progress.data.id ?? null : null
  localStorage.setItem(GENERATION_RECOVERY_KEY, JSON.stringify({ ...ctx, baselineId }))
}

function readRecoveryContext(): GenerationRecoveryContext | null {
  const saved = localStorage.getItem(GENERATION_RECOVERY_KEY)
  if (!saved) return null
  try {
    const ctx = JSON.parse(saved) as GenerationRecoveryContext
    if (!ctx.projectId || typeof ctx.baselineId === 'undefined') {
      localStorage.removeItem(GENERATION_RECOVERY_KEY)
      return null
    }
    return ctx
  } catch {
    localStorage.removeItem(GENERATION_RECOVERY_KEY)
    return null
  }
}

async function importVideo(ctx: GenerationRecoveryContext, result: string | string[], api: RecoveryImporterApi): Promise<void> {
  const videoPath = typeof result === 'string' ? result : result[0]
  if (!videoPath) return

  const copied = await addVisualAssetToProject(videoPath, ctx.projectId, 'video')
  if (!copied) throw new Error('Could not persist generated video to project storage')

  const settings = ctx.settings
  const mode = ctx.inputAudioUrl ? 'audio-to-video' : ctx.inputImageUrl ? 'image-to-video' : 'text-to-video'
  api.addAsset(ctx.projectId, {
    type: 'video',
    path: copied.path,
    bigThumbnailPath: copied.bigThumbnailPath,
    smallThumbnailPath: copied.smallThumbnailPath,
    width: copied.width,
    height: copied.height,
    prompt: ctx.prompt,
    resolution: settings?.videoResolution ?? '',
    duration: settings?.duration,
    generationParams: {
      mode,
      prompt: ctx.prompt,
      model: settings?.model ?? 'fast',
      duration: settings?.duration ?? 0,
      resolution: settings?.videoResolution ?? '',
      fps: settings?.fps ?? 24,
      audio: settings?.audio ?? false,
      cameraMotion: 'none',
      imageAspectRatio: settings?.aspectRatio,
      imageSteps: 4,
      inputImageUrl: ctx.inputImageUrl,
      inputAudioUrl: ctx.inputAudioUrl,
    },
    takes: [{
      path: copied.path,
      bigThumbnailPath: copied.bigThumbnailPath,
      smallThumbnailPath: copied.smallThumbnailPath,
      width: copied.width,
      height: copied.height,
      createdAt: Date.now(),
    }],
    activeTakeIndex: 0,
  })
}

async function importImages(ctx: GenerationRecoveryContext, result: string | string[], api: RecoveryImporterApi): Promise<void> {
  const paths = Array.isArray(result) ? result : [result]
  const settings = ctx.settings
  let importedAny = false

  for (const imagePath of paths) {
    const copied = await addVisualAssetToProject(imagePath, ctx.projectId, 'image')
    if (!copied) {
      logger.error(`Could not persist generated image to project storage: ${imagePath}`)
      continue
    }
    importedAny = true
    api.addAsset(ctx.projectId, {
      type: 'image',
      path: copied.path,
      bigThumbnailPath: copied.bigThumbnailPath,
      smallThumbnailPath: copied.smallThumbnailPath,
      width: copied.width,
      height: copied.height,
      prompt: ctx.prompt,
      resolution: settings?.imageResolution ?? '',
      generationParams: {
        mode: 'text-to-image',
        prompt: ctx.prompt,
        model: 'fast',
        duration: 5,
        resolution: settings?.imageResolution ?? '',
        fps: 24,
        audio: false,
        cameraMotion: 'none',
        imageAspectRatio: settings?.aspectRatio,
        imageSteps: settings?.imageSteps ?? 4,
      },
      takes: [{
        path: copied.path,
        bigThumbnailPath: copied.bigThumbnailPath,
        smallThumbnailPath: copied.smallThumbnailPath,
        width: copied.width,
        height: copied.height,
        createdAt: Date.now(),
      }],
      activeTakeIndex: 0,
    })
  }

  if (!importedAny && paths.length > 0) {
    throw new Error('Could not persist any generated image to project storage')
  }
}

export async function checkAndConsumeGenerationRecovery(api: RecoveryImporterApi): Promise<void> {
  let ctx = readRecoveryContext()
  if (!ctx) return
  if (ctx.projectId === activeOwnerProjectId) return

  const progress = await ApiClient.getGenerationProgress()
  if (!progress.ok) return

  const observedId = progress.data.id ?? null
  if (ctx.generationId == null) {
    if (observedId === ctx.baselineId) return
    ctx = { ...ctx, generationId: observedId ?? undefined }
    localStorage.setItem(GENERATION_RECOVERY_KEY, JSON.stringify(ctx))
  } else if (observedId !== ctx.generationId) {
    localStorage.removeItem(GENERATION_RECOVERY_KEY)
    return
  }

  if (progress.data.status === 'running') return

  if (progress.data.status === 'complete' && progress.data.result != null) {
    if (ctx.genType === 'image') {
      await importImages(ctx, progress.data.result, api)
    } else {
      await importVideo(ctx, progress.data.result, api)
    }
  }

  localStorage.removeItem(GENERATION_RECOVERY_KEY)
}
