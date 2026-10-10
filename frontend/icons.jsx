// Minimal stroked icon set — 1.5 stroke, 20px box
const Icon = ({ name, size = 18, className = '', stroke = 1.6 }) => {
  const common = {
    width: size, height: size, viewBox: '0 0 24 24', fill: 'none',
    stroke: 'currentColor', strokeWidth: stroke,
    strokeLinecap: 'round', strokeLinejoin: 'round',
    className,
  };
  const paths = {
    upload: <><path d="M12 16V4"/><path d="m7 9 5-5 5 5"/><path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/></>,
    book: <><path d="M4 5a2 2 0 0 1 2-2h12v17H6a2 2 0 0 0-2 2V5z"/><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H18"/></>,
    file: <><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/></>,
    sparkle: <><path d="M12 3v4"/><path d="M12 17v4"/><path d="M3 12h4"/><path d="M17 12h4"/><path d="m5.5 5.5 2.8 2.8"/><path d="m15.7 15.7 2.8 2.8"/><path d="m18.5 5.5-2.8 2.8"/><path d="m8.3 15.7-2.8 2.8"/></>,
    arrowRight: <><path d="M5 12h14"/><path d="m13 5 7 7-7 7"/></>,
    check: <path d="m5 12 5 5 9-12"/>,
    x: <><path d="M6 6l12 12"/><path d="M18 6 6 18"/></>,
    chevronDown: <path d="m6 9 6 6 6-6"/>,
    chevronRight: <path d="m9 6 6 6-6 6"/>,
    chevronLeft: <path d="m15 6-6 6 6 6"/>,
    quote: <><path d="M3 7h7v7H3z"/><path d="M14 7h7v7h-7z"/><path d="M3 14c0 3 0 4-3 4"/><path d="M14 14c0 3 0 4-3 4"/></>,
    cards: <><rect x="3" y="6" width="14" height="12" rx="2"/><path d="M7 3h13a2 2 0 0 1 2 2v9"/></>,
    pen: <><path d="M16.5 3.5a2.121 2.121 0 1 1 3 3L7 19l-4 1 1-4z"/></>,
    note: <><path d="M4 4h16v16H4z"/><path d="M8 8h8"/><path d="M8 12h8"/><path d="M8 16h5"/></>,
    star: <path d="m12 2 3.1 6.3 6.9 1-5 4.9 1.2 6.8L12 17.8 5.8 21l1.2-6.8-5-4.9 6.9-1z"/>,
    target: <><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/></>,
    refresh: <><path d="M3 12a9 9 0 0 1 15.5-6.3L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-15.5 6.3L3 16"/><path d="M3 21v-5h5"/></>,
    summary: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6"
           strokeLinecap="round" strokeLinejoin="round">
        <path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" />
        <polyline points="14 3 14 9 20 9" />
        <line x1="8" y1="13" x2="16" y2="13" />
        <line x1="8" y1="17" x2="13" y2="17" />
      </svg>
    ),
    flashcards: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6"
           strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="7" width="14" height="11" rx="1.5" />
        <path d="M7 4h13a1.5 1.5 0 0 1 1.5 1.5V15" opacity="0.55" />
      </svg>
    ),
    quiz: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6"
           strokeLinecap="round" strokeLinejoin="round">
        <path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" />
        <polyline points="14 3 14 9 20 9" />
        <path d="M10 14a2 2 0 1 1 2 2v0.5" />
        <line x1="12" y1="18.5" x2="12" y2="18.6" strokeWidth="2.2" strokeLinecap="round" />
      </svg>
    ),
    mindmap: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6"
           strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="5" r="2" />
        <line x1="12" y1="7" x2="12" y2="11" />
        <circle cx="5" cy="16" r="2" />
        <circle cx="12" cy="16" r="2" />
        <circle cx="19" cy="16" r="2" />
        <line x1="12" y1="13" x2="5" y2="14.5" />
        <line x1="12" y1="13" x2="12" y2="14.5" />
        <line x1="12" y1="13" x2="19" y2="14.5" />
      </svg>
    ),
  };
  return <svg {...common}>{paths[name] || null}</svg>;
};

window.Icon = Icon;
