# 3. Frontend build step and local gate — esbuild concat

- **Status:** Accepted
- **Date:** 2026-09-21

## Context

The frontend was a classic-script React SPA: 15 `.jsx` files (~9,800 lines)
loaded via React 18 UMD + Babel-standalone `type="text/babel"` tags from unpkg,
with every cross-file reference passing through `window.*` globals. Cache
invalidation was 15 hand-edited `?v=` query strings in `frontend/index.html`;
the repo's own history records four different `?v=` conventions and a "blank
screen on notebook entry" incident, and the README's advice was "hard-reload
with DevTools open". Babel bytecode compilation also ran in every visitor's
browser.

There was also no gate: no CI, no git remote, no shared pytest config, and a
Playwright suite whose `webServer` served a root `index.html` that Step 1 of the
audit had deleted.

A constraint shaped the fix:

- **Naive concatenation would black-screen the app.** `api.jsx:5` and
  `app.jsx:3` both declare a top-level `const API_BASE`. Two classic scripts
  concatenated would throw `SyntaxError: Identifier 'API_BASE' has already been
  declared`; the app worked only because Babel-standalone evaluated each file in
  its own scope. The build therefore had to wrap each file in its own function
  scope.

## Decision

- **Build:** esbuild `transform()` each `.jsx` in a fixed, dependency-ordered
  list (classic JSX runtime → `React.createElement` against the global `React`),
  wrap each result in its own IIFE, concatenate, and append the inline
  bootstrap. The bundle is content-hashed (`dist/app.<sha256-8>.js`) and a
  generated `frontend/index.html` is written from
  `frontend/index.template.html` by replacing a single `<!--APP_BUNDLE-->`
  marker. The template is the single source of truth, so the `?v=` class of bug
  is structurally impossible rather than merely detectable.
- **Source map:** the bundle is a concatenation, so it gets an *indexed*
  ("sectioned") source map — one section per original `.jsx`, offset by that
  file's line position — rather than a single linear map.
- **Gate:** a local `scripts/check.ps1` runs, in order, the build-freshness check
  (`--check`, byte-compares emitted output against disk) and `pytest` over
  `backend/tests`.

## Consequences

- Babel-standalone no longer ships to the browser and the `?v=` cache-buster is
  gone by construction. Components are unchanged — the build is
  behavior-preserving (the 15 `.jsx` files were not edited).
- `frontend/index.html` and `frontend/dist/` are build artifacts that are now
  committed (not gitignored), so a clean clone can serve the browser route
  without running the unsigned esbuild binary; a drift check
  (`npm run build:check` / `scripts/check.ps1`) still fails a change that
  edited a `.jsx` without rebuilding.
- **Vite/ESM is deferred, not rejected.** Converting to modules means rewriting
  every `window.*` cross-reference across ~10k lines in a UI with no component
  tests, for no immediate user benefit. An ESM migration is the deliberate
  follow-up when the frontend next needs real growth.
