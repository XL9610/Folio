// Preload — bridges window controls and the per-launch API token to the
// renderer. Exposed on window.electronAPI; the renderer uses its presence to
// detect Electron and only then renders the custom title bar.
const { contextBridge, ipcRenderer } = require('electron');

// Token passed by the main process via additionalArguments; the renderer sends
// it as X-Auth-Token so the backend can reject foreign web pages.
const tokenArg = process.argv.find((a) => a.startsWith('--folio-api-token='));
const apiToken = tokenArg ? tokenArg.slice('--folio-api-token='.length) : '';

contextBridge.exposeInMainWorld('electronAPI', {
  isElectron: true,
  platform: process.platform, // 'win32' | 'darwin' | 'linux'
  apiToken,
  minimize: () => ipcRenderer.send('window:minimize'),
  maximize: () => ipcRenderer.send('window:maximize'),
  close: () => ipcRenderer.send('window:close'),
});
