// Builds the frontend bundle from the *.jsx files and generates index.html.
//
// The app is a classic-script (window-global) React SPA. Each .jsx file is
// transpiled with esbuild's classic JSX transform and wrapped in its own IIFE so
// that top-level `const`/`let`/`class` bindings do not collide across files —
// api.jsx and app.jsx both declare `const API_BASE`, which is legal only because
// Babel standalone used to evaluate each file in its own scope. Cross-file
// references already go through `window.*`, so IIFE scoping is
// behavior-preserving.
//
// The generated frontend/index.html and frontend/dist/ are build artifacts:
// the single source of truth is this script plus index.template.html, so the
// `?v=` cache-buster class of bug is impossible by construction.
//
// A source map is emitted alongside the bundle. The bundle is a concatenation
// of independently-transpiled files, so it cannot have a single linear map:
// instead it is an indexed ("sectioned") source map whose sections point at
// each original .jsx, offset by that file's line position in the bundle.
//
// Usage:
//   node frontend/build.mjs            build dist/app.<hash>.js[.map] + index.html
//   node frontend/build.mjs --check    fail (exit 1) if emitted output differs
//                                      from what is on disk; a missing build
//                                      is reported as "not built yet", a
//                                      mismatching one as "drift"

import { build as esbuildBuild, transform } from 'esbuild';
import { createHash } from 'node:crypto';
import {
  readFileSync,
  writeFileSync,
  mkdirSync,
  existsSync,
  readdirSync,
  unlinkSync,
} from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const nodeRequire = createRequire(import.meta.url);
const FRONTEND_DIR = __dirname;
const DIST_DIR = join(FRONTEND_DIR, 'dist');
const TEMPLATE_PATH = join(FRONTEND_DIR, 'index.template.html');
const INDEX_PATH = join(FRONTEND_DIR, 'index.html');
const BUNDLE_PREFIX = 'app';
// The template must contain this exact placeholder; build.mjs replaces it with
// the hashed <script src>. Its absence means the template and builder are out of
// sync, and the page would load no bundle at all.
const BUNDLE_MARKER = '<!--APP_BUNDLE-->';
const VENDOR_BUNDLE_MARKER = '<!--VENDOR_BUNDLE-->';
const TAILWIND_CSS_MARKER = '<!--TAILWIND_CSS-->';
const INPUT_CSS_PATH = join(FRONTEND_DIR, 'input.css');
const TAILWIND_CONFIG_PATH = join(FRONTEND_DIR, 'tailwind.config.cjs');

// Load order is dependency-ordered and load-bearing: earlier files install the
// `window.*` globals later files read. Do not reorder without tracing the
// references.
const LOAD_ORDER = [
  'themes.jsx',
  'preferences.jsx',
  'data.jsx',
  'icons.jsx',
  'primitives.jsx',
  'error-boundary.jsx',
  'api.jsx',
  'sources.jsx',
  'ask.jsx',
  'studio.jsx',
  'modals.jsx',
  'panels.jsx',
  'settings.jsx',
  'tweaks-panel.jsx',
  'app.jsx',
  'titlebar.jsx',
  'notebooks.jsx',
];

// The inline bootstrap that used to sit at the bottom of index.html. It runs
// last, after React/ReactDOM UMD globals and window.FolioApp are in place.
const BOOTSTRAP = `
(function() {
  const root = ReactDOM.createRoot(document.getElementById('root'));
  root.render(React.createElement(window.FolioRoot));
})();
`;

function hashOf(bytes) {
  return createHash('sha256').update(bytes).digest('hex').slice(0, 8);
}

function countLines(s) {
  let n = 0;
  for (let i = 0; i < s.length; i++) {
    if (s[i] === '\n') n++;
  }
  return n;
}

// Build the bundle, the source map, and the generated index.html in memory.
async function build() {
  const template = readFileSync(TEMPLATE_PATH, 'utf8');
  for (const marker of [BUNDLE_MARKER, VENDOR_BUNDLE_MARKER, TAILWIND_CSS_MARKER]) {
    if (!template.includes(marker)) {
      throw new Error(
        `frontend/index.template.html must contain the ${marker} marker; ` +
          'the template and build.mjs are out of sync.',
      );
    }
  }

  const parts = [];
  const sections = [];
  let lineOffset = 0;

  for (const file of LOAD_ORDER) {
    const code = readFileSync(join(FRONTEND_DIR, file), 'utf8');
    const result = await transform(code, {
      loader: 'jsx',
      jsx: 'transform',   // classic runtime -> React.createElement(global React)
      target: 'es2020',
      legalComments: 'none',
      minifyWhitespace: true,
      minifySyntax: true,
      sourcemap: 'external',
      sourcefile: file,
      sourcesContent: true,
    });

    // The part is newline-terminated and self-contained. `parts.join('')` keeps
    // the running line offset exact for the sectioned map (no separator lines).
    const part = `/* ${file} */\n(function() {\n${result.code}\n})();\n`;

    const submap = JSON.parse(result.map);
    delete submap.file;   // the top-level `file` wins for an indexed map
    sections.push({ offset: { line: lineOffset, column: 0 }, map: submap });

    parts.push(part);
    lineOffset += countLines(part);
  }
  parts.push(`/* bootstrap */\n${BOOTSTRAP}`);

  // Hash the code only, so the filename does not depend on the sourceMappingURL
  // comment that names it (which would be circular). The bootstrap has no
  // section, so its few lines are simply unmapped.
  const bundle = parts.join('');
  const hash = hashOf(bundle);
  const bundleName = `${BUNDLE_PREFIX}.${hash}.js`;
  const mapName = `${bundleName}.map`;
  const finalBundle = `${bundle}\n//# sourceMappingURL=${mapName}\n`;
  const map = JSON.stringify({ version: 3, file: bundleName, sections });

  const vendor = await buildVendor();
  const css = await buildCSS();
  const html = template
    .replace(TAILWIND_CSS_MARKER, `<link rel="stylesheet" href="dist/${css.name}">`)
    .replace(BUNDLE_MARKER, `<script src="dist/${bundleName}"></script>`)
    .replace(VENDOR_BUNDLE_MARKER, `<script src="dist/${vendor.name}"></script>`);
  return { bundle: finalBundle, bundleName, mapName, map, html, vendor, css };
}

// Bundle vendor-entry.js (and its npm imports) into a single IIFE that exposes
// React/ReactDOM/marked/DOMPurify on `window`. Hashed like the app
// bundle so the browser can cache it independently of app-code changes.
async function buildVendor() {
  const result = await esbuildBuild({
    entryPoints: [join(FRONTEND_DIR, 'vendor-entry.js')],
    bundle: true,
    format: 'iife',
    target: 'es2020',
    minify: true,
    write: false,
    define: { 'process.env.NODE_ENV': '"production"' },
    logLevel: 'silent',
  });
  const out = result.outputFiles && result.outputFiles[0];
  if (!out) throw new Error('vendor bundle produced no output');
  const code = `${out.text}\n`;
  const name = `vendor.${hashOf(code)}.js`;
  return { code, name };
}

// Compile input.css through Tailwind + Autoprefixer at build time, so no
// Tailwind runtime (Play CDN) ships to the browser. The output is hashed,
// linked from index.html, and drift-checked by --check.
async function buildCSS() {
  const tailwindcss = nodeRequire('tailwindcss');
  const postcss = nodeRequire('postcss');
  const autoprefixer = nodeRequire('autoprefixer');
  const config = nodeRequire(TAILWIND_CONFIG_PATH);
  const source = readFileSync(INPUT_CSS_PATH, 'utf8');
  const result = await postcss([tailwindcss(config), autoprefixer]).process(source, {
    from: INPUT_CSS_PATH,
    to: undefined,
  });
  const code = result.css;
  const name = `tailwind.${hashOf(code)}.css`;
  return { code, name };
}

function removeStaleBundles(keep) {
  if (!existsSync(DIST_DIR)) return;
  const keepSet = new Set(keep);
  for (const name of readdirSync(DIST_DIR)) {
    const isBundle = name.startsWith(`${BUNDLE_PREFIX}.`) || name.startsWith('vendor.') || name.startsWith('tailwind.');
    if (isBundle && !keepSet.has(name)) {
      unlinkSync(join(DIST_DIR, name));
    }
  }
}

async function write() {
  const { bundle, bundleName, mapName, map, html, vendor, css } = await build();
  mkdirSync(DIST_DIR, { recursive: true });
  writeFileSync(join(DIST_DIR, bundleName), bundle, 'utf8');
  writeFileSync(join(DIST_DIR, mapName), map, 'utf8');
  writeFileSync(join(DIST_DIR, vendor.name), vendor.code, 'utf8');
  writeFileSync(join(DIST_DIR, css.name), css.code, 'utf8');
  writeFileSync(INDEX_PATH, html, 'utf8');
  removeStaleBundles([bundleName, mapName, vendor.name, css.name]);
  console.log(`built dist/${bundleName} (${bundle.length} bytes)`);
  console.log(`built dist/${vendor.name} (${vendor.code.length} bytes)`);
  console.log(`built dist/${css.name} (${css.code.length} bytes)`);
  console.log(`wrote dist/${mapName} and index.html`);
}

async function check() {
  const { bundle, bundleName, mapName, map, html, vendor, css } = await build();

  // A missing index.html means the build has never been run on this checkout,
  // not that it has drifted; say so distinctly so the fix is obvious.
  if (!existsSync(INDEX_PATH)) {
    console.error(
      'frontend is not built yet: frontend/index.html and frontend/dist/ are missing.\n' +
        'Run "npm run build" to generate them.',
    );
    process.exit(1);
  }

  const mismatches = [];
  if (readFileSync(INDEX_PATH, 'utf8') !== html) {
    mismatches.push('index.html');
  }
  const bundlePath = join(DIST_DIR, bundleName);
  if (!existsSync(bundlePath) || readFileSync(bundlePath, 'utf8') !== bundle) {
    mismatches.push(`dist/${bundleName}`);
  }
  const mapPath = join(DIST_DIR, mapName);
  if (!existsSync(mapPath) || readFileSync(mapPath, 'utf8') !== map) {
    mismatches.push(`dist/${mapName}`);
  }
  const vendorPath = join(DIST_DIR, vendor.name);
  if (!existsSync(vendorPath) || readFileSync(vendorPath, 'utf8') !== vendor.code) {
    mismatches.push(`dist/${vendor.name}`);
  }
  const cssPath = join(DIST_DIR, css.name);
  if (!existsSync(cssPath) || readFileSync(cssPath, 'utf8') !== css.code) {
    mismatches.push(`dist/${css.name}`);
  }

  if (mismatches.length) {
    console.error(
      `build drift detected: ${mismatches.join(', ')} does not match source.\n` +
        'Run "npm run build" to regenerate.',
    );
    process.exit(1);
  }
  console.log('build is up to date');
}

const isCheck = process.argv.includes('--check');
if (isCheck) {
  await check();
} else {
  await write();
}
