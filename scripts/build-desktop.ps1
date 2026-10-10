# build-desktop.ps1 — one-shot desktop build.
#
# Encodes the three manual build steps:
#   1. frontend bundle   -> npm run build
#   2. backend onedir    -> PyInstaller (CPU-torch venv)
#   3. NSIS installer    -> electron-builder
# plus the embedding-model fetch and a .env leak tripwire.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/build-desktop.ps1
#   # point at a different packaging venv:
#   powershell -File scripts/build-desktop.ps1 -PythonExe "C:/venv/Scripts/python.exe"
#   # skip the post-build backend smoke test:
#   powershell -File scripts/build-desktop.ps1 -SkipSmoke

[CmdletBinding()]
param(
    # Python interpreter of the CPU-torch packaging venv (PyInstaller + CPU torch 2.x).
    [string]$PythonExe = $env:RAG_PKG_PYTHON,
    # Skip the post-build exe smoke test (still runs the .env tripwire).
    [switch]$SkipSmoke
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

if (-not $PythonExe) {
    # Default to a `rag-pkg-venv` under the user's home (CPU-torch + PyInstaller).
    $PythonExe = Join-Path $env:USERPROFILE "rag-pkg-venv/Scripts/python.exe"
}
if (-not (Test-Path $PythonExe)) {
    throw "Packaging venv python not found: $PythonExe  (override with -PythonExe or `$env:RAG_PKG_PYTHON)"
}

function Info([string]$msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Fail([string]$msg) { throw $msg }
function Assert-NoEnvLeak([string[]]$Paths, [string]$Label) {
    $leaked = Get-ChildItem -Path $Paths -Recurse -Force -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -eq ".env" -or ($_.Name -like ".env.*" -and $_.Name -ne ".env.example") }
    if ($leaked) {
        $leaked | ForEach-Object { Write-Host "  LEAK: $($_.FullName)" -ForegroundColor Red }
        Fail "refusing to package: .env file(s) found in $Label"
    }
    Info "no .env in $Label"
}

# --- 0. prerequisites ---------------------------------------------------------
if (-not (Get-Command node -ErrorAction SilentlyContinue)) { Fail "node not found on PATH" }

# --- 1. frontend --------------------------------------------------------------
Info "Building frontend (npm run build)"
npm run build
if ($LASTEXITCODE -ne 0) { Fail "npm run build failed" }

# --- 2. embedding model -------------------------------------------------------
$FetchScript = Join-Path $Root "backend/scripts/fetch_embedding_model.py"
$ModelDir = Join-Path $Root "backend/packaging/embedding_model"
if (Test-Path $FetchScript) {
    Info "Fetching embedding model"
    & $PythonExe $FetchScript
    if ($LASTEXITCODE -ne 0) { Fail "embedding model fetch failed" }
} elseif (-not (Test-Path $ModelDir)) {
    Write-Warning "embedding model not bundled (fetch script missing) — first run will download from HuggingFace"
}

# --- 2b. tiktoken cache (required for offline PDF chunking) ------------------
$TiktokenFetch = Join-Path $Root "backend/scripts/fetch_tiktoken_cache.py"
if (-not (Test-Path $TiktokenFetch)) { Fail "tiktoken cache fetch script missing: $TiktokenFetch" }
Info "Fetching tiktoken cache"
& $PythonExe $TiktokenFetch
if ($LASTEXITCODE -ne 0) { Fail "tiktoken cache fetch failed" }

# --- 3. backend onedir --------------------------------------------------------
Info "Building backend (PyInstaller)"
& $PythonExe -m PyInstaller --noconfirm --clean --distpath build --workpath build/pyinstaller-work backend/packaging/rag_backend.spec
if ($LASTEXITCODE -ne 0) { Fail "PyInstaller failed" }

# --- 4. .env leak tripwire (pre) ----------------------------------------------
Assert-NoEnvLeak @((Join-Path $Root "build/rag-backend"), (Join-Path $Root "frontend"), (Join-Path $Root "electron")) "packaging sources"

# --- 5. installer -------------------------------------------------------------
Info "Building installer (electron-builder)"
npx electron-builder --win nsis
if ($LASTEXITCODE -ne 0) { Fail "electron-builder failed" }

# --- 6. .env leak tripwire (post) ---------------------------------------------
Assert-NoEnvLeak @((Join-Path $Root "release/win-unpacked")) "installer output"

# --- 7. smoke test ------------------------------------------------------------
if (-not $SkipSmoke) {
    Info "Smoke-testing packaged backend (health check on :8123)"
    $BackendExe = Join-Path $Root "build/rag-backend/rag-backend.exe"
    if (-not (Test-Path $BackendExe)) { Fail "backend exe not found: $BackendExe" }
    $env:PORT = "8123"
    $proc = Start-Process -FilePath $BackendExe -PassThru -WindowStyle Hidden
    try {
        $ready = $false
        foreach ($i in 1..30) {
            Start-Sleep -Milliseconds 1000
            try {
                $r = Invoke-WebRequest -Uri "http://127.0.0.1:8123/health" -UseBasicParsing -TimeoutSec 2
                if ($r.StatusCode -eq 200) { $ready = $true; break }
            } catch { }
        }
        if (-not $ready) { Fail "packaged backend did not become healthy within 30s" }
        Info "packaged backend healthy (health 200)"
    } finally {
        if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
        Remove-Item Env:\PORT -ErrorAction SilentlyContinue
    }
}

$Version = (Get-Content (Join-Path $Root "package.json") -Raw | ConvertFrom-Json).version
Info "Done. Installer: release/Folio Setup $Version.exe"
