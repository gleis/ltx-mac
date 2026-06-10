import { app } from 'electron'
import fs from 'fs'
import path from 'path'
import os from 'os'

export const APP_FOLDER_NAME = 'LTXDesktop'

function resolveUserDataPath(): string {
  if (process.env.LTX_APP_DATA_DIR) {
    return path.resolve(process.env.LTX_APP_DATA_DIR)
  }

  if (!app.isPackaged) {
    return path.resolve(process.cwd(), '.ltx-data')
  }

  if (process.platform === 'win32') {
    const localAppData = process.env.LOCALAPPDATA
      || path.join(os.homedir(), 'AppData', 'Local')
    return path.join(localAppData, APP_FOLDER_NAME)
  }
  if (process.platform === 'darwin') {
    return path.join(
      os.homedir(),
      'Library',
      'Application Support',
      APP_FOLDER_NAME,
    )
  }
  const xdgData = process.env.XDG_DATA_HOME || path.join(os.homedir(), '.local', 'share')
  return path.join(xdgData, APP_FOLDER_NAME)
}

const userDataPath = resolveUserDataPath()
fs.mkdirSync(userDataPath, { recursive: true })
app.setPath('userData', userDataPath)

export function getAppDataDir(): string {
  return app.getPath('userData')
}

export function getLogDir(): string {
  return path.join(app.getPath('userData'), 'logs')
}
