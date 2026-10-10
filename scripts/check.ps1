#Requires -Version 5.1
<#
.SYNOPSIS
    Local quality gate for RAGStudyCompanion.

.DESCRIPTION
    Runs two steps and aggregates the result:
      1. frontend - the committed build artifacts match the .jsx sources
      2. pytest    - backend/tests

    Exits 0 only if no step FAILED. A SKIP is not a failure.

.PARAMETER PythonExe
    Interpreter to drive pytest. Default: python.

.PARAMETER SkipFrontend
    Skip step 1 (e.g. you are only touching Python).

.PARAMETER SkipTests
    Skip step 2.

.EXAMPLE
    pwsh -File scripts/check.ps1
#>
[CmdletBinding()]
param(
    [string] $PythonExe = 'python',
    [switch] $SkipFrontend,
    [switch] $SkipTests
)

$ErrorActionPreference = 'Continue'   # native tools signal failure via exit codes
$LASTEXITCODE = 0

$RepoRoot = Split-Path -Parent $PSScriptRoot

# Offline: point tiktoken at the bundled BPE cache so a clean, network-free
# machine can run pytest without tiktoken fetching cl100k_base from Azure.
$TiktokenCache = Join-Path $RepoRoot 'backend/packaging/tiktoken_cache'
if (Test-Path -LiteralPath $TiktokenCache) {
    $env:TIKTOKEN_CACHE_DIR = $TiktokenCache
}

Push-Location -LiteralPath $RepoRoot

$results = [System.Collections.ArrayList]::new()

function Add-Result {
    param([string]$Step, [string]$State, [string]$Detail)
    [void]$results.Add([pscustomobject]@{ Step = $Step; State = $State; Detail = $Detail })
}

function Write-Head {
    param([string]$Text)
    Write-Host ''
    Write-Host "=== $Text " -ForegroundColor Cyan -NoNewline
    Write-Host ('=' * [Math]::Max(3, 64 - $Text.Length)) -ForegroundColor DarkCyan
}

try {
    Write-Host "gate root: $RepoRoot" -ForegroundColor DarkGray

    # ── Step 1: frontend build freshness ────────────────────────────────────
    if ($SkipFrontend) {
        Add-Result 'frontend build' 'SKIP' '-SkipFrontend'
    }
    else {
        Write-Head 'frontend build freshness'
        $node = Get-Command node -ErrorAction SilentlyContinue
        if (-not $node) {
            Add-Result 'frontend build' 'FAIL' 'node not on PATH'
        }
        elseif (-not (Test-Path -LiteralPath (Join-Path $RepoRoot 'node_modules/esbuild/package.json'))) {
            Add-Result 'frontend build' 'FAIL' 'esbuild not installed - run: npm install'
        }
        else {
            # --check rebuilds in memory and byte-compares against the committed
            # index.html and dist/. Non-zero means a .jsx was edited without
            # rebuilding.
            & node 'frontend/build.mjs' '--check'
            $code = $LASTEXITCODE
            if ($code -eq 0) {
                Add-Result 'frontend build' 'PASS' 'artifacts match sources'
            } else {
                Add-Result 'frontend build' 'FAIL' 'stale or missing - run: npm run build'
            }
        }
    }

    # ── Step 2: pytest ──────────────────────────────────────────────────────
    if ($SkipTests) {
        Add-Result 'pytest' 'SKIP' '-SkipTests'
    }
    else {
        Write-Head 'pytest (backend/tests)'
        $py = Get-Command $PythonExe -ErrorAction SilentlyContinue
        if (-not $py) {
            Add-Result 'pytest' 'FAIL' "$PythonExe not on PATH"
        }
        else {
            & $PythonExe -m pytest -q
            $code = $LASTEXITCODE
            if ($code -eq 0) {
                Add-Result 'pytest' 'PASS' 'all tests passed'
            } else {
                Add-Result 'pytest' 'FAIL' "exit $code - see output above"
            }
        }
    }

    # ── Summary ────────────────────────────────────────────────────────────
    Write-Host ''
    $results | Format-Table -AutoSize -Property Step, State, Detail | Out-String -Width 200 | Write-Host

    $failed  = @($results | Where-Object { $_.State -eq 'FAIL' })
    $skipped = @($results | Where-Object { $_.State -eq 'SKIP' })

    if ($failed.Count -gt 0) {
        Write-Host ("GATE FAILED ({0} step(s))" -f $failed.Count) -ForegroundColor Red
        exit 1
    }
    if ($skipped.Count -gt 0) {
        Write-Host ("GATE PASSED (with {0} SKIPPED: {1})" -f `
            $skipped.Count, (($skipped | ForEach-Object { $_.Step }) -join ', ')) -ForegroundColor Yellow
    } else {
        Write-Host 'GATE PASSED' -ForegroundColor Green
    }
    exit 0
}
finally {
    Pop-Location
}
