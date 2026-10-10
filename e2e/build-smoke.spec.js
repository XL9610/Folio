// Build smoke test — proves the esbuild bundle preserved the app's load-time
// behavior. The old notebook-integration.spec.js / workspace-responsive.spec.js
// targeted the deleted multi-notebook "Notebooks" dashboard and were removed in
// the Step 4 DX sweep; this is their replacement.
//
// Precondition: `npm run build` must have produced frontend/index.html + dist/
// (both committed to the repo). `npx playwright test` serves frontend/ via webServer.
const { test, expect } = require('@playwright/test');

test('built bundle mounts the Folio workspace with its libraries intact', async ({ page }) => {
  test.setTimeout(60_000);

  // A SyntaxError in the bundle (e.g. the duplicate-`const API_BASE` collision
  // that IIFE wrapping prevents) or a missing global would surface here, not as
  // a failed assertion below.
  const pageErrors = [];
  page.on('pageerror', (err) => pageErrors.push(err.message));

  await page.goto('/index.html');
  await expect(page).toHaveTitle(/Study Companion/);

  // Without a backend the app correctly renders the notebook catalog (the
  // landing view), not the workspace. This is the mount marker: if the bundle
  // did not execute (parse error) or React never rendered, it never appears.
  await expect(page.locator('[data-screen-label="Notebook Catalog"]')).toBeVisible({
    timeout: 30_000,
  });

  // Commit 2 restored marked/DOMPurify; the bootstrap depends on window.FolioApp.
  // Assert the exact API surface the code calls (ask.jsx -> marked.parse /
  // DOMPurify.sanitize; bootstrap -> window.FolioApp), not a brittle typeof.
  const libs = await page.evaluate(() => ({
    folioApp: typeof window.FolioApp === 'function',
    marked: typeof window.marked?.parse === 'function',
    dompurify: typeof window.DOMPurify?.sanitize === 'function',
  }));
  expect(libs).toEqual({ folioApp: true, marked: true, dompurify: true });

  // The app fetches /api/config/* on load and falls back silently when the
  // backend is absent, so those requests failing are expected — but an uncaught
  // exception is never expected.
  expect(pageErrors).toEqual([]);
});
