const THEMES = {
  midnight: {
    label: 'Midnight',
    vars: {
      '--surface-main': '#0a0a0a', '--surface-left': '#0a0a0a',
      '--surface-right': '#0a0a0a', '--surface-card': '#191919',
      '--surface-card-hover': '#1f2125', '--surface-selected': '#1a1c20',
      '--border-soft': '#212327', '--border-medium': '#363a3f',
      '--text-primary': '#ffffff', '--text-secondary': '#dadbdf',
      '--text-muted': '#7d8187', '--accent': '#ffffff', '--accent-soft': '#a0c3ec',
      '--accent-bg': 'rgba(255,255,255,0.08)', '--ok': '#a0c3ec', '--warn': '#ff7a17',
      '--danger': '#ff7a17', '--radius-card': '8px', '--radius-button': '9999px',
      '--shadow-card': 'none', '--shadow-soft': 'none', '--shadow-ring': '0 0 0 1px rgba(255,255,255,0.35)',
      '--kind-quiz': '#f59e0b', '--kind-flashcards': '#60a5fa',
      '--kind-mindmap': '#fbbf24', '--kind-summary': 'var(--accent)',
      '--font-display': 'Geist, Inter, ui-sans-serif, system-ui, sans-serif',
      '--font-body': 'Geist, Inter, ui-sans-serif, system-ui, sans-serif',
      '--font-mono': '"JetBrains Mono", ui-monospace, monospace',
    },
  },
  paper: {
    label: 'Paper',
    vars: {
      '--surface-main': '#f7f3e8', '--surface-left': '#eee8d8',
      '--surface-right': '#e9dfcc', '--surface-card': '#f4efe2',
      '--surface-card-hover': '#fbf7ee', '--surface-selected': '#e2d6bd',
      '--border-soft': '#d8ceb8', '--border-medium': '#c9bda3',
      '--text-primary': '#241f18', '--text-secondary': '#756b5a',
      '--text-muted': '#9a8f7b', '--accent': '#74613f', '--accent-soft': '#b8a780',
      '--accent-bg': '#eef0fb', '--ok': '#3F8A5B', '--warn': '#B97A1E',
      '--danger': '#B6552A', '--radius-card': '8px', '--radius-button': '9999px',
      '--shadow-card': '0 1px 0 rgba(20,23,28,0.04), 0 1px 2px rgba(20,23,28,0.04)',
      '--shadow-soft': '0 1px 2px rgba(20,23,28,0.04), 0 8px 24px -12px rgba(20,23,28,0.10)',
      '--shadow-ring': '0 0 0 1px rgba(61,73,199,0.18), 0 1px 2px rgba(20,23,28,0.04), 0 8px 24px -12px rgba(20,23,28,0.12)',
      '--kind-quiz': '#d97706', '--kind-flashcards': '#2563eb',
      '--kind-mindmap': '#b45309', '--kind-summary': 'var(--accent)',
      '--font-display': '"Newsreader", Georgia, serif',
      '--font-body': '"Newsreader", Georgia, serif',
      '--font-mono': '"JetBrains Mono", ui-monospace, monospace',
    },
  },
  ember: {
    label: 'Ember',
    vars: {
      '--surface-main': '#f7f7f4', '--surface-left': '#f7f7f4',
      '--surface-right': '#efeee8', '--surface-card': '#ffffff',
      '--surface-card-hover': '#fafaf7', '--surface-selected': '#e6e5e0',
      '--border-soft': '#e6e5e0', '--border-medium': '#cfcdc4',
      '--text-primary': '#26251e', '--text-secondary': '#5a5852',
      '--text-muted': '#807d72', '--accent': '#f54e00', '--accent-soft': '#dfa88f',
      '--accent-bg': 'rgba(245,78,0,0.08)', '--ok': '#1f8a65', '--warn': '#f54e00',
      '--danger': '#cf2d56', '--radius-card': '12px', '--radius-button': '8px',
      '--shadow-card': 'none', '--shadow-soft': 'none', '--shadow-ring': '0 0 0 1px rgba(245,78,0,0.25)',
      '--kind-quiz': '#d97706', '--kind-flashcards': '#3b82f6',
      '--kind-mindmap': '#ca8a04', '--kind-summary': 'var(--accent)',
      '--font-display': "'Inter', system-ui, 'Helvetica Neue', Helvetica, Arial, sans-serif",
      '--font-body': "'Inter', system-ui, 'Helvetica Neue', Helvetica, Arial, sans-serif",
      '--font-mono': "'JetBrains Mono', 'Fira Code', monospace",
    },
  },
  sage: {
    label: 'Sage',
    vars: {
      '--surface-main': '#f2f6f0', '--surface-left': '#e8efe4',
      '--surface-right': '#eef4e8', '--surface-card': '#f6faf4',
      '--surface-card-hover': '#edf5e8', '--surface-selected': '#dce8d4',
      '--border-soft': '#c8d8be', '--border-medium': '#a8c09a',
      '--text-primary': '#1c2818', '--text-secondary': '#4a6040',
      '--text-muted': '#7a9470', '--accent': '#4a7c3f', '--accent-soft': '#8cba7e',
      '--accent-bg': '#edf5e8', '--ok': '#3F8A5B', '--warn': '#B97A1E',
      '--danger': '#B6552A', '--radius-card': '8px', '--radius-button': '9999px',
      '--shadow-card': '0 1px 0 rgba(20,23,28,0.04), 0 1px 2px rgba(20,23,28,0.04)',
      '--shadow-soft': '0 1px 2px rgba(20,23,28,0.04), 0 8px 24px -12px rgba(20,23,28,0.10)',
      '--shadow-ring': '0 0 0 1px rgba(74,124,63,0.20)',
      '--kind-quiz': '#d97706', '--kind-flashcards': '#2563eb',
      '--kind-mindmap': '#b45309', '--kind-summary': 'var(--accent)',
      '--font-display': '"Newsreader", Georgia, serif',
      '--font-body': '"Newsreader", Georgia, serif',
      '--font-mono': '"JetBrains Mono", ui-monospace, monospace',
    },
  },
  coastal: {
    label: 'Coastal',
    vars: {
      '--surface-main': '#f0f4f8', '--surface-left': '#e4ecf4',
      '--surface-right': '#ecf2f8', '--surface-card': '#f4f8fc',
      '--surface-card-hover': '#e8f0f8', '--surface-selected': '#d8e4f0',
      '--border-soft': '#c0d0e4', '--border-medium': '#a0b8d4',
      '--text-primary': '#182838', '--text-secondary': '#405870',
      '--text-muted': '#688098', '--accent': '#3d65a8', '--accent-soft': '#80a8d8',
      '--accent-bg': '#e8f0f8', '--ok': '#3F8A5B', '--warn': '#B97A1E',
      '--danger': '#B6552A', '--radius-card': '8px', '--radius-button': '9999px',
      '--shadow-card': '0 1px 0 rgba(20,23,28,0.04), 0 1px 2px rgba(20,23,28,0.04)',
      '--shadow-soft': '0 1px 2px rgba(20,23,28,0.04), 0 8px 24px -12px rgba(20,23,28,0.10)',
      '--shadow-ring': '0 0 0 1px rgba(61,101,168,0.22)',
      '--kind-quiz': '#ea580c', '--kind-flashcards': '#0891b2',
      '--kind-mindmap': '#ca8a04', '--kind-summary': 'var(--accent)',
      '--font-display': '"Newsreader", Georgia, serif',
      '--font-body': '"Newsreader", Georgia, serif',
      '--font-mono': '"JetBrains Mono", ui-monospace, monospace',
    },
  },
  gallery: {
    label: 'Gallery',
    vars: {
      '--surface-main': '#ffffff', '--surface-left': '#f5f5f7',
      '--surface-right': '#f5f5f7', '--surface-card': '#ffffff',
      '--surface-card-hover': '#fafafc', '--surface-selected': '#f0f0f0',
      '--border-soft': '#f0f0f0', '--border-medium': '#e0e0e0',
      '--text-primary': '#1d1d1f', '--text-secondary': '#333333',
      '--text-muted': '#7a7a7a', '--accent': '#0066cc', '--accent-soft': '#2997ff',
      '--accent-bg': 'rgba(0,102,204,0.08)', '--ok': '#1f8a65', '--warn': '#0066cc',
      '--danger': '#cf2d56', '--radius-card': '18px', '--radius-button': '9999px',
      '--shadow-card': 'none', '--shadow-soft': 'none', '--shadow-ring': '0 0 0 2px rgba(0,102,204,0.4)',
      '--kind-quiz': '#d97706', '--kind-flashcards': '#0891b2',
      '--kind-mindmap': '#ca8a04', '--kind-summary': 'var(--accent)',
      '--font-display': "system-ui, -apple-system, BlinkMacSystemFont, 'Inter', sans-serif",
      '--font-body': "system-ui, -apple-system, BlinkMacSystemFont, 'Inter', sans-serif",
      '--font-mono': "'JetBrains Mono', 'Fira Code', monospace",
    },
  },
};

window.THEMES = THEMES;

// Apply a theme's CSS variables to the document root. Single source of truth
// for theme application: the App calls it on mount/switch, and it is also
// invoked at load time so landing views (the notebook catalog) render with the
// theme before the App mounts.
const applyFolioTheme = (themeName) => {
  const theme = window.THEMES[themeName] || window.THEMES.paper;
  const id = '__folio_theme';
  let el = document.getElementById(id);
  if (!el) { el = document.createElement('style'); el.id = id; document.head.appendChild(el); }
  const toRgbTriplet = (value) => {
    const s = String(value || '').trim();
    const hex = s.match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
    if (hex) {
      let h = hex[1];
      if (h.length === 3) h = h.split('').map((ch) => ch + ch).join('');
      return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)].join(' ');
    }
    const rgb = s.match(/rgba?\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)/i);
    if (rgb) return `${Math.round(Number(rgb[1]))} ${Math.round(Number(rgb[2]))} ${Math.round(Number(rgb[3]))}`;
    return null;
  };
  const colorAliases = [
    ['--surface-main', '--color-paper'],
    ['--surface-card-hover', '--color-cream'],
    ['--surface-card', '--color-card'],
    ['--text-primary', '--color-ink'],
    ['--text-secondary', '--color-ink2'],
    ['--text-muted', '--color-muted'],
    ['--border-medium', '--color-line'],
    ['--border-soft', '--color-line2'],
    ['--surface-selected', '--color-indigo-50'],
    ['--surface-card-hover', '--color-indigo-100'],
    ['--surface-card', '--color-indigo-200'],
    ['--accent-soft', '--color-indigo-400'],
    ['--accent', '--color-indigo-500'],
    ['--text-primary', '--color-indigo-700'],
    ['--ok', '--color-ok'],
    ['--warn', '--color-warn'],
  ];
  const derived = colorAliases.flatMap(([source, alias]) => {
    const rgb = toRgbTriplet(theme.vars[source]);
    if (!rgb) return [];
    const entries = [`${alias}:${rgb}`];
    if (alias === '--color-indigo-500') entries.push(`--color-indigo-600:${rgb}`);
    return entries;
  });
  const rules = Object.entries(theme.vars).map(([k, v]) => `${k}:${v}`).concat(derived).join(';');
  el.textContent = `:root { ${rules} }`;
};
window.applyFolioTheme = applyFolioTheme;

// Apply the saved (or OS-default) theme immediately so the catalog and any
// other pre-App view render with theme CSS variables instead of undefined.
(function () {
  const osPrefersLight = typeof window !== 'undefined' && window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches;
  const defaultTheme = osPrefersLight ? 'paper' : 'midnight';
  let initial = defaultTheme;
  try {
    const version = localStorage.getItem('folio.themeDefaultVersion');
    const saved = localStorage.getItem('folio.theme');
    if (version === 'theme-v2' && saved && window.THEMES[saved]) initial = saved;
  } catch (e) {}
  applyFolioTheme(initial);
})();
