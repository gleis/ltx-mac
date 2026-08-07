import { useEffect, useRef } from 'react'
import { useProjects } from '../contexts/ProjectContext'
import { checkAndConsumeGenerationRecovery } from '../lib/generation-recovery'

const RECOVERY_POLL_INTERVAL_MS = 2000

export function useGenerationRecoveryWatcher(): void {
  const { addAsset } = useProjects()
  const isCheckingRef = useRef(false)

  useEffect(() => {
    const tick = () => {
      if (isCheckingRef.current) return
      isCheckingRef.current = true
      void checkAndConsumeGenerationRecovery({ addAsset })
        .catch(() => {
          // Keep the marker so a later tick can retry.
        })
        .finally(() => { isCheckingRef.current = false })
    }

    tick()
    const intervalId = window.setInterval(tick, RECOVERY_POLL_INTERVAL_MS)
    return () => window.clearInterval(intervalId)
  }, [addAsset])
}
