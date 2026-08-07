export async function withGenerationActive<T>(fn: () => Promise<T>): Promise<T> {
  void window.electronAPI.notifyGenerationActive({ active: true })
  try {
    return await fn()
  } finally {
    void window.electronAPI.notifyGenerationActive({ active: false })
  }
}
