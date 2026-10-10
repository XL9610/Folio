// Notebook catalog — landing view: a shelf of notebooks, each rendered as a
// book cover. Lists notebooks, creates one, opens one.
const { useState, useEffect } = React;

// Muted, bookish cover palette. Deterministic per notebook id so a notebook
// always renders with the same colour, giving the shelf a stable, varied look.
const BOOK_COLORS = ['#3D49C7', '#3F8A5B', '#8C3A3A', '#2E3A59', '#B97A1E', '#6B4A8C', '#46636B'];

const bookColor = (id) => {
  let h = 0;
  const s = String(id || '');
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
  return BOOK_COLORS[h % BOOK_COLORS.length];
};

const BookCard = ({ nb, onOpen, onDelete }) => {
  const color = bookColor(nb.id);
  const handleDelete = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (onDelete) onDelete(nb);
  };
  return (
    <div
      onClick={() => onOpen(nb.id)}
      onContextMenu={handleDelete}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onOpen(nb.id); } }}
      className="group relative w-full text-left"
      style={{ background: 'transparent', border: 'none', padding: 0, cursor: 'pointer' }}
    >
      <div
        className="relative w-full overflow-hidden rounded-lg lift"
        style={{ aspectRatio: '3 / 4', background: color, boxShadow: '0 1px 2px rgba(20,23,28,0.06), 0 4px 10px -4px rgba(20,23,28,0.10)' }}
      >
        {/* Spine (binding) + crease */}
        <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: 12, background: 'rgba(0,0,0,0.22)' }} />
        <div style={{ position: 'absolute', left: 14, top: 0, bottom: 0, width: 1, background: 'rgba(255,255,255,0.22)' }} />

        {/* Cover content */}
        <div style={{ position: 'absolute', inset: 0, padding: '20px 15px 16px 28px', display: 'flex', flexDirection: 'column' }}>
          <div
            className="serif-display"
            style={{
              color: '#FFFFFF', fontSize: 'calc(21px * var(--folio-font-scale, 1))', lineHeight: 1.15,
              fontWeight: 500, letterSpacing: '-0.01em',
              display: '-webkit-box', WebkitLineClamp: 3, WebkitBoxOrient: 'vertical', overflow: 'hidden',
            }}
          >
            {nb.name}
          </div>
          {nb.description ? (
            <div
              className="sans mt-2"
              style={{
                color: 'rgba(255,255,255,0.82)', fontSize: 'calc(12px * var(--folio-font-scale, 1))', lineHeight: 1.45,
                display: '-webkit-box', WebkitLineClamp: 4, WebkitBoxOrient: 'vertical', overflow: 'hidden',
              }}
            >
              {nb.description}
            </div>
          ) : null}
          <div
            className="mono mt-auto"
            style={{ color: 'rgba(255,255,255,0.72)', fontSize: 'calc(10.5px * var(--folio-font-scale, 1))', letterSpacing: '0.08em', textTransform: 'uppercase' }}
          >
            {nb.source_count} {nb.source_count === 1 ? 'source' : 'sources'}
          </div>
        </div>
      </div>
      {onDelete && (
        <button
          type="button"
          onClick={handleDelete}
          title="Delete notebook"
          aria-label={`Delete ${nb.name}`}
          className="absolute top-2 right-2 z-10 opacity-0 group-hover:opacity-100 transition-opacity"
          style={{ background: 'rgba(0,0,0,0.55)', border: 'none', borderRadius: 8, width: 28, height: 28, display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', color: '#fff' }}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 6h18" /><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" /></svg>
        </button>
      )}
    </div>
  );
};

// Poll the backend's embedding-model status so the app can show a small
// "preparing semantic search" banner on first run instead of silently
// degrading to keyword search. Returns 'loading' | 'error' | null.
const useEmbeddingStatus = () => {
  const [state, setState] = useState(null);
  useEffect(() => {
    let cancelled = false;
    let tries = 0;
    const poll = async () => {
      try {
        const s = await window.FOLIO_API.embeddingStatus();
        if (cancelled) return;
        if (s.status === 'ready') { setState(null); return; }
        if (s.status === 'error') { setState('error'); return; }
        setState('loading');
        tries += 1;
        if (tries < 45) setTimeout(poll, 2000);
      } catch (e) {
        // Backend not up yet, or an older backend without the endpoint — stay quiet.
        if (!cancelled) setState(null);
      }
    };
    poll();
    return () => { cancelled = true; };
  }, []);
  return state;
};

const NotebookCatalog = ({ onOpen }) => {
  const [notebooks, setNotebooks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState('');
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [theme, setTheme] = useState(() => { try { return localStorage.getItem('folio.theme') || 'paper'; } catch (e) { return 'paper'; } });

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const list = await window.FOLIO_API.listNotebooks();
        if (!cancelled) {
          setNotebooks(Array.isArray(list) ? list : []);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) setError(e.message || 'Failed to load notebooks');
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const handleDeleteNotebook = async (nb) => {
    if (!window.confirm(`Delete "${nb.name}" and all its sources? This cannot be undone.`)) return;
    try {
      await window.FOLIO_API.deleteNotebook(nb.id);
      setNotebooks(prev => prev.filter(n => n.id !== nb.id));
    } catch (e) {
      alert('Failed to delete notebook: ' + (e.message || 'Unknown error'));
    }
  };

  const submitCreate = async (e) => {
    e.preventDefault();
    const trimmed = name.trim();
    if (!trimmed) return;
    try {
      const nb = await window.FOLIO_API.createNotebook(trimmed);
      onOpen(nb.id);
    } catch (err) {
      alert('Failed to create notebook: ' + (err.message || 'Unknown error'));
    }
  };

  return (
    <div
      className="h-full w-full flex flex-col"
      data-screen-label="Notebook Catalog"
      style={{ background: 'var(--surface-main)', color: 'var(--text-primary)' }}
    >
      <div className="flex-1 overflow-y-auto scroll-thin">
        <div className="max-w-4xl mx-auto px-10 pt-14 pb-16">
          {/* Header */}
          <div className="flex items-end justify-between">
            <div>
              <div className="serif-display" style={{ fontSize: 'calc(40px * var(--folio-font-scale, 1))', letterSpacing: '-0.01em' }}>Folio</div>
              <div className="sans mt-2" style={{ fontSize: 'calc(13px * var(--folio-font-scale, 1))', color: 'var(--text-secondary)' }}>
                Your library of study notebooks
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setSettingsOpen(true)}
                title="Settings"
                aria-label="Settings"
                className="lift"
                style={{ color: 'var(--text-muted)', padding: 6, borderRadius: 8, background: 'transparent', border: 'none', cursor: 'pointer', display: 'inline-flex', alignItems: 'center' }}
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
              </button>
              <Button onClick={() => setCreating(true)}>New notebook</Button>
            </div>
          </div>

          {error && (
            <div className="sans rounded-lg px-4 py-3 mt-8" style={{ fontSize: 'calc(13px * var(--folio-font-scale, 1))', color: 'var(--warn)', background: 'var(--surface-card)', border: '1px solid var(--border-soft)' }}>
              {error} — is the backend running?
            </div>
          )}
          {loading && (
            <div className="sans mt-8" style={{ fontSize: 'calc(13px * var(--folio-font-scale, 1))', color: 'var(--text-muted)' }}>
              Loading notebooks…
            </div>
          )}

          {/* The shelf */}
          <div className="mt-10 grid grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-5">
            {notebooks.map((nb) => <BookCard key={nb.id} nb={nb} onOpen={onOpen} onDelete={handleDeleteNotebook} />)}
            {!loading && (
              <button
                onClick={() => setCreating(true)}
                className="group"
                style={{ background: 'transparent', border: 'none', padding: 0, cursor: 'pointer' }}
              >
                <div
                  className="w-full rounded-lg flex flex-col items-center justify-center gap-3 lift"
                  style={{ aspectRatio: '3 / 4', border: '1.5px dashed var(--border-medium)', background: 'var(--surface-card)', color: 'var(--text-muted)' }}
                >
                  <span style={{ fontSize: 'calc(30px * var(--folio-font-scale, 1))', lineHeight: 1, color: 'var(--text-muted)' }}>+</span>
                  <span className="sans" style={{ fontSize: 'calc(12px * var(--folio-font-scale, 1))', color: 'var(--text-muted)' }}>New notebook</span>
                </div>
              </button>
            )}
          </div>

          {!loading && !error && notebooks.length === 0 && (
            <div className="mt-12 rounded-2xl overflow-hidden" style={{ background: 'var(--surface-card)', border: '1px solid var(--border-soft)', boxShadow: '0 12px 40px -24px rgba(20,23,28,0.25)' }}>
              <div className="grid grid-cols-1 md:grid-cols-2">
                {/* Left: intro + CTA */}
                <div className="px-10 py-12" style={{ borderBottom: '1px solid var(--border-soft)' }}>
                  <div className="mono" style={{ fontSize: 'calc(11px * var(--folio-font-scale, 1))', letterSpacing: '0.16em', textTransform: 'uppercase', color: '#3D49C7' }}>
                    Getting started
                  </div>
                  <div className="serif-display mt-4" style={{ fontSize: 'calc(30px * var(--folio-font-scale, 1))', color: 'var(--text-primary)', letterSpacing: '-0.01em', lineHeight: 1.15 }}>
                    Build your study library
                  </div>
                  <div className="sans mt-3" style={{ fontSize: 'calc(14px * var(--folio-font-scale, 1))', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
                    Turn your textbooks and notes into a study assistant you can ask — every answer cites the exact chapter and section it came from.
                  </div>
                  <div className="mt-8">
                    <Button onClick={() => setCreating(true)}>Create your first notebook</Button>
                  </div>
                  <div className="mono mt-6" style={{ fontSize: 'calc(10.5px * var(--folio-font-scale, 1))', color: 'var(--text-muted)', letterSpacing: '0.04em' }}>
                    PDF · EPUB · PPTX · DOCX · MD · TXT — processed privately on your machine
                  </div>
                </div>

                {/* Right: three-step timeline */}
                <div className="px-10 py-12" style={{ background: 'var(--surface-main)' }}>
                  <div className="sans" style={{ fontSize: 'calc(11px * var(--folio-font-scale, 1))', letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--text-muted)' }}>
                    In three steps
                  </div>
                  <ol className="mt-7">
                    {[
                      { n: '1', title: 'Create a notebook', body: 'A home for a course, a book, or a topic.', active: true },
                      { n: '2', title: 'Add your books', body: 'Drop in PDFs, EPUBs, slides, notes and more.', active: false },
                      { n: '3', title: 'Ask anything', body: 'Get answers that cite the exact chapter and section.', active: false },
                    ].map((s, i, arr) => (
                      <li key={s.n} className="flex gap-4 relative" style={{ paddingBottom: i < arr.length - 1 ? 24 : 0 }}>
                        {i < arr.length - 1 && (
                          <span className="absolute left-[15px] top-9 bottom-0 w-px" style={{ background: 'var(--border-medium)' }} />
                        )}
                        <span
                          className="shrink-0 w-8 h-8 rounded-full flex items-center justify-center z-10"
                          style={{
                            background: s.active ? '#3D49C7' : 'var(--surface-card)',
                            border: s.active ? 'none' : '1px solid var(--border-medium)',
                            color: s.active ? '#fff' : 'var(--text-muted)',
                            fontSize: 'calc(13px * var(--folio-font-scale, 1))',
                            fontWeight: 600,
                          }}
                        >
                          {s.n}
                        </span>
                        <div className="min-w-0">
                          <div className="sans" style={{ fontSize: 'calc(14px * var(--folio-font-scale, 1))', color: s.active ? 'var(--text-primary)' : 'var(--text-secondary)', fontWeight: 600 }}>{s.title}</div>
                          <div className="sans mt-0.5" style={{ fontSize: 'calc(12.5px * var(--folio-font-scale, 1))', color: 'var(--text-muted)', lineHeight: 1.5 }}>{s.body}</div>
                        </div>
                      </li>
                    ))}
                  </ol>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Create modal */}
      {creating && (
        <div className="fixed inset-0 flex items-center justify-center" style={{ background: 'rgba(20,23,28,0.35)', zIndex: 1000 }} onClick={() => setCreating(false)}>
          <div
            className="rounded-xl px-8 py-7"
            style={{ background: 'var(--surface-card)', border: '1px solid var(--border-soft)', boxShadow: '0 8px 24px -12px rgba(20,23,28,0.18)', width: 380, maxWidth: 'calc(100vw - 40px)' }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="serif-display" style={{ fontSize: 'calc(22px * var(--folio-font-scale, 1))', color: 'var(--text-primary)' }}>New notebook</div>
            <form onSubmit={submitCreate} className="mt-5">
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                autoFocus
                placeholder="Notebook name"
                className="sans w-full"
                style={{ fontSize: 'calc(15px * var(--folio-font-scale, 1))', color: 'var(--text-primary)', background: 'var(--surface-main)', border: '1px solid var(--border-medium)', borderRadius: 8, padding: '10px 12px', outline: 'none' }}
              />
              <div className="mt-5 flex justify-end gap-2">
                <Button variant="ghost" onClick={() => setCreating(false)}>Cancel</Button>
                <Button type="submit">Create</Button>
              </div>
            </form>
          </div>
        </div>
      )}

      <window.SettingsModal
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        theme={theme}
        onThemeChange={(v) => { setTheme(v); if (window.applyFolioTheme) window.applyFolioTheme(v); }}
        themes={window.THEMES}
      />
    </div>
  );
};

// Top-level entry: shows the catalog until a notebook is chosen, then the App
// with a way back. This replaces the old direct-mount of <App notebookId="default">.
const FolioRoot = () => {
  const [notebookId, setNotebookId] = useState(null);
  const [booting, setBooting] = useState(true);
  const embeddingState = useEmbeddingStatus();

  useEffect(() => {
    // Reopen the most recently opened notebook on launch; otherwise land on the
    // catalog. Only auto-opens when a notebook has a recorded last_opened_at.
    let cancelled = false;
    (async () => {
      try {
        const list = await window.FOLIO_API.listNotebooks();
        if (cancelled) return;
        const recent = (Array.isArray(list) ? list : [])
          .filter((n) => n.last_opened_at)
          .sort((a, b) => (a.last_opened_at < b.last_opened_at ? 1 : -1))[0];
        if (recent) setNotebookId(recent.id);
      } catch (e) {
        // Backend not up yet — the catalog will surface its own error state.
      } finally {
        if (!cancelled) setBooting(false);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const handleOpen = (id) => {
    // Record "recently opened" for the shelf ordering (fire-and-forget).
    if (window.FOLIO_API && window.FOLIO_API.touchNotebook) {
      window.FOLIO_API.touchNotebook(id).catch(() => {});
    }
    setNotebookId(id);
  };
  const content = booting
    ? null
    : notebookId === null
      ? <NotebookCatalog onOpen={handleOpen} />
      : React.createElement(window.FolioApp, { notebookId, onBack: () => setNotebookId(null) });
  return (
    <div className="h-full w-full flex flex-col" style={{ background: 'var(--surface-main)', color: 'var(--text-primary)' }}>
      <window.TitleBar />
      {embeddingState && (
        <div
          className="sans text-center"
          style={{
            padding: '6px 16px',
            fontSize: 'calc(12px * var(--folio-font-scale, 1))',
            color: embeddingState === 'error' ? 'var(--warn)' : 'var(--text-secondary)',
            background: 'var(--surface-card)',
            borderBottom: '1px solid var(--border-soft)',
          }}
        >
          {embeddingState === 'error'
            ? 'Semantic search unavailable — using keyword search.'
            : 'Preparing semantic search…'}
        </div>
      )}
      <div className="flex-1 min-h-0 overflow-hidden">{content}</div>
    </div>
  );
};

window.NotebookCatalog = NotebookCatalog;
window.FolioRoot = FolioRoot;
