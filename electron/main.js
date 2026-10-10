// Electron main process — frameless shell for the RAG Study Companion.
//
// Spawns the Python backend (dev: `python -m uvicorn`; the production build
// swaps this for a bundled PyInstaller exe in a later step), waits for it to
// become healthy, then opens a frameless window loading the built frontend.
// Window controls are native on Windows/macOS (titleBarOverlay / traffic
// lights) and custom on Linux (frame:false + IPC).

const { app, BrowserWindow, ipcMain, shell } = require('electron');
const { spawn } = require('node:child_process');
const path = require('node:path');
const http = require('node:http');
const crypto = require('node:crypto');

const BACKEND_PORT = 8000;
// Per-launch API token, handed to the backend (env) and to the renderer
// (additionalArguments) so the local API rejects foreign web pages.
const API_TOKEN = crypto.randomBytes(32).toString('hex');
const BACKEND_DIR = path.join(__dirname, '..', 'backend');
const FRONTEND_INDEX = path.join(__dirname, '..', 'frontend', 'index.html');

let backendProc = null;

function probe(path, headers = {}) {
  return new Promise((resolve) => {
    const req = http.get(
      `http://127.0.0.1:${BACKEND_PORT}${path}`,
      { headers },
      (res) => {
        res.resume();
        resolve(res.statusCode === 200);
      },
    );
    req.on('error', () => resolve(false));
    req.setTimeout(1500, () => { req.destroy(); resolve(false); });
  });
}

// A backend is "ours" only when it answers /health AND accepts our per-launch
// token. A leftover backend from a previous run enforces a different token, so
// an anonymous /health alone would falsely reuse it and every API call would 401.
async function backendIsOurs() {
  return (
    (await probe('/health')) &&
    (await probe('/api/notebooks', { 'X-Auth-Token': API_TOKEN }))
  );
}

async function waitForBackend(timeoutMs = 30000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (await backendIsOurs()) return true;
    await new Promise((r) => setTimeout(r, 500));
  }
  return false;
}

function spawnBackend() {
  let cmd;
  let args;
  let opts;
  if (app.isPackaged) {
    // Packaged build: the PyInstaller onedir lives in resources/backend/.
    cmd = path.join(process.resourcesPath, 'backend', 'rag-backend.exe');
    args = [];
    opts = { stdio: 'ignore', windowsHide: true, env: { ...process.env, FOLIO_API_TOKEN: API_TOKEN } };
  } else {
    // Dev: run the repo's Python backend directly.
    cmd = 'python';
    args = ['-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', String(BACKEND_PORT)];
    opts = { cwd: BACKEND_DIR, stdio: 'ignore', windowsHide: true, env: { ...process.env, FOLIO_API_TOKEN: API_TOKEN } };
  }
  backendProc = spawn(cmd, args, opts);
  backendProc.on('exit', () => { backendProc = null; });
}

async function ensureBackend() {
  if (await backendIsOurs()) return true; // already running with matching token
  spawnBackend();
  return waitForBackend();
}

function createWindow() {
  const opts = {
    width: 1280,
    height: 800,
    minWidth: 720,
    minHeight: 520,
    backgroundColor: '#F4F1E8',
    show: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      additionalArguments: ['--folio-api-token=' + API_TOKEN],
    },
  };

  if (process.platform === 'win32') {
    opts.titleBarStyle = 'hidden';
    opts.titleBarOverlay = { color: '#F4F1E8', symbolColor: '#15171C', height: 40 };
  } else if (process.platform === 'darwin') {
    opts.titleBarStyle = 'hiddenInset';
    opts.trafficLightPosition = { x: 12, y: 11 };
  } else {
    opts.frame = false; // Linux: draw our own controls via the title bar
  }

  const win = new BrowserWindow(opts);
  win.once('ready-to-show', () => win.show());

  // The renderer renders clickable Markdown links, but a foreign page loaded
  // into this window would receive the per-launch token from preload. Block all
  // in-window navigation and new windows, and hand only http(s) links to the
  // system browser.
  const openExternalIfHttp = (url) => {
    if (/^https?:\/\//i.test(url)) shell.openExternal(url);
  };
  win.webContents.setWindowOpenHandler(({ url }) => {
    openExternalIfHttp(url);
    return { action: 'deny' };
  });
  win.webContents.on('will-navigate', (event, url) => {
    event.preventDefault();
    openExternalIfHttp(url);
  });

  win.loadFile(FRONTEND_INDEX);
  return win;
}

ipcMain.on('window:minimize', (e) => BrowserWindow.fromWebContents(e.sender)?.minimize());
ipcMain.on('window:maximize', (e) => {
  const w = BrowserWindow.fromWebContents(e.sender);
  if (!w) return;
  if (w.isMaximized()) w.unmaximize();
  else w.maximize();
});
ipcMain.on('window:close', (e) => BrowserWindow.fromWebContents(e.sender)?.close());

// A second instance would generate a fresh token but reuse the first instance's
// backend on port 8000, making every API call 401. Reject extra instances.
const gotTheLock = app.requestSingleInstanceLock();
if (!gotTheLock) {
  app.quit();
} else {
  app.on('second-instance', () => {
    const win = BrowserWindow.getAllWindows()[0];
    if (win) {
      if (win.isMinimized()) win.restore();
      win.focus();
    }
  });

  app.whenReady().then(async () => {
    const ok = await ensureBackend();
    if (!ok) {
      const { dialog } = require('electron');
      dialog.showErrorBox(
        'Backend failed to start',
        `The backend could not start on port ${BACKEND_PORT}.\n\n` +
          'Check that port 8000 is free and that the bundled backend is not blocked by antivirus, then restart the app.',
      );
      app.quit();
      return;
    }
    createWindow();
    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
  });

  app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit();
  });

  app.on('before-quit', () => {
    if (backendProc) {
      try { backendProc.kill(); } catch { /* ignore */ }
    }
  });
}
