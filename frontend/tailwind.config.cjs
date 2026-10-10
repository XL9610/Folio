// Tailwind CSS build-time configuration — generates a static CSS file (no Play
// CDN runtime). The theme mirrors the inline `tailwind.config` that the removed
// `frontend/vendor/tailwind-3.4.17.js` runtime previously read from
// index.template.html. content scans the generated HTML template and every .jsx;
// all className values resolve to literal strings in these files, so no safelist
// is required.
const path = require('path');

// Bind content to this file's location (frontend/) so the generated CSS is
// identical whether the build runs from the repo root or from frontend/.
const here = __dirname.replace(/\\/g, '/');

module.exports = {
  content: [
    `${here}/index.template.html`,
    `${here}/*.jsx`,
  ],
  safelist: [],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Geist', 'ui-sans-serif', 'system-ui'],
        serif: ['Newsreader', 'ui-serif', 'Georgia'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
      colors: {
        paper: '#F4F1E8',
        cream: '#FAF8F1',
        card: '#FFFFFF',
        ink: '#15171C',
        ink2: '#3A3D45',
        muted: '#74777F',
        line: '#E4DFCF',
        line2: '#EDE8D8',
        indigo: {
          50: '#EEF0FB',
          100: '#DCE0F5',
          200: '#B8C0EB',
          400: '#5C6BD9',
          500: '#3D49C7',
          600: '#2E39A8',
          700: '#23297F',
        },
        ok: '#3F8A5B',
        warn: '#B97A1E',
      },
      boxShadow: {
        card: '0 1px 0 rgba(20,23,28,0.04), 0 1px 2px rgba(20,23,28,0.04)',
        soft: '0 1px 2px rgba(20,23,28,0.04), 0 8px 24px -12px rgba(20,23,28,0.10)',
        ring: '0 0 0 1px rgba(61,73,199,0.18), 0 1px 2px rgba(20,23,28,0.04), 0 8px 24px -12px rgba(20,23,28,0.12)',
      },
    },
  },
};
