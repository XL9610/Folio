// Main app shell ...chat-centric three-column workspace
const { useState, useEffect, useLayoutEffect, useRef } = React;
const API_BASE = 'http://127.0.0.1:8000';
const NARROW_LAYOUT_BREAKPOINT = 980;

// Drag-to-resize handle. Lives between two columns; expects the column it grows
// against to have its width controlled by `width` state.
const ColumnResizer = ({ side, width, setWidth, min, max, onDragStateChange }) => {
  const [hovering, setHovering] = useState(false);
  const [dragging, setDragging] = useState(false);

  const onMouseDown = (e) => {
    e.preventDefault();
    setDragging(true);
    onDragStateChange && onDragStateChange(true);
    const startX = e.clientX;
    const startW = width;
    const sign = side === 'left' ? 1 : -1; // left handle grows on +dx; right handle grows on -dx
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';

    const onMove = (ev) => {
      const dx = ev.clientX - startX;
      const next = Math.max(min, Math.min(max, startW + sign * dx));
      setWidth(next);
    };
    const onUp = () => {
      setDragging(false);
      onDragStateChange && onDragStateChange(false);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  };

  const active = hovering || dragging;
  return (
    <div
      onMouseDown={onMouseDown}
      onMouseEnter={() => setHovering(true)}
      onMouseLeave={() => setHovering(false)}
      onDoubleClick={() => setWidth(side === 'left' ? 300 : 280)}
      className="relative shrink-0 h-full select-none"
      style={{ width: 8, cursor: 'col-resize', zIndex: 10, marginLeft: -4, marginRight: -4 }}
      title="Drag to resize - double-click to reset"
    >
      <span
        aria-hidden
        className="absolute top-0 bottom-0 left-1/2 transition-colors"
        style={{
          width: 1,
          marginLeft: -0.5,
          background: active ? 'var(--accent)' : 'transparent',
        }}
      />
      <span
        aria-hidden
        className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 transition-opacity"
        style={{
          width: 2,
          height: 28,
          borderRadius: 1,
          background: 'var(--accent)',
          opacity: active ? 0.9 : 0,
        }}
      />
    </div>
  );
};

const FOLIO_DEFAULTS = {
  "theme": "midnight",
  "density": "comfortable",
  "showSidebar": true,
  "showStudio": true
};

const getArtifactId = (artifact) => artifact?.artifact_id || artifact?.id || '';

const normalizeArtifactIdentity = (artifact) => {
  const artifactId = getArtifactId(artifact);
  if (!artifact || !artifactId) return artifact;
  return {
    ...artifact,
    id: artifactId,
    artifact_id: artifact.artifact_id || artifactId,
  };
};

// Map backend document to frontend source shape
const SOURCE_COLORS = ['#3D49C7', '#7A5BD2', '#3F8A5B', '#B97A1E', '#B6552A', '#2F7A52'];
const THEME_DEFAULT_VERSION = 'theme-v2';
let _colorIdx = 0;

const getInitialFolioTheme = () => {
  const osPrefersLight = typeof window !== 'undefined'
    && window.matchMedia
    && window.matchMedia('(prefers-color-scheme: light)').matches;
  const defaultTheme = osPrefersLight ? 'paper' : 'midnight';
  try {
    const version = localStorage.getItem('folio.themeDefaultVersion');
    const savedTheme = localStorage.getItem('folio.theme');
    if (version === THEME_DEFAULT_VERSION && savedTheme && window.THEMES[savedTheme]) {
      return savedTheme;
    }
  } catch (e) {}
  return defaultTheme;
};

/** Clean up a filename into a readable display title. */
const cleanFileName = (raw) => {
  let name = (raw || 'Untitled').replace(/\.(pdf|epub|pptx|docx|md|txt)$/i, '');
  // Replace underscores with spaces
  name = name.replace(/_/g, ' ');
  // Remove common library/site suffixes (before collapsing spaces)
  name = name.replace(/\bz[_\s-]*library\b/gi, '');
  name = name.replace(/\b1lib\b/gi, '');
  name = name.replace(/\b1lib\s*sk\b/gi, '');
  name = name.replace(/\bsk\b/gi, '');
  // Remove trailing commas, dots, dashes, and excess whitespace between tokens
  name = name.replace(/[,.\-]+/g, ' ');
  // Collapse multiple spaces
  name = name.replace(/\s+/g, ' ').trim();
  // Remove a lone trailing comma if it survived
  name = name.replace(/,\s*$/, '').trim();
  return name || 'Untitled Document';
};

const mapDocumentToSource = (doc) => {
  const color = SOURCE_COLORS[_colorIdx % SOURCE_COLORS.length];
  _colorIdx++;
  const name = doc.display_title || cleanFileName(doc.filename);
  return {
    id: doc.document_id,
    title: name,
    originalFilename: doc.filename,
    authors: '',
    edition: (doc.status === 'completed' || doc.status === 'already_ingested') ? 'Indexed'
      : (doc.status === 'indexed_partial') ? 'Indexed (partial)'
      : (doc.status === 'fts_only') ? 'Keyword only'
      : doc.status,
    pages: doc.page_count || 0,
    chunks: doc.chunk_count || 0,
    status: (doc.status === 'completed' || doc.status === 'already_ingested' || doc.status === 'indexed_partial' || doc.status === 'fts_only') ? 'ready'
      : (doc.status === 'failed') ? 'failed'
      : 'processing',
    toc: true,
    frontMatter: true,
    addedRel: doc.created_at || 'just now',
    color,
    summary: '',
    keyTopics: [],
    chapters: [],
  };
};

// Track window width for responsive layout
const useWindowWidth = () => {
  const [w, setW] = useState(typeof window !== 'undefined' ? window.innerWidth : 1200);
  useEffect(() => {
    const onResize = () => setW(window.innerWidth);
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);
  return w;
};

const getFolioSettingsSnapshot = () => (
  window.FOLIO_SETTINGS?.getSettings?.()
  || window.getSettings?.()
  || window.FOLIO_SETTINGS_DEFAULTS
  || {}
);

const clampNumber = (value, fallback, min, max) => {
  const n = typeof value === 'number' ? value : Number.parseInt(value, 10);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, Math.round(n)));
};

const buildAnswerPreferences = (settings) => {
  const answer = settings?.answer || {};
  const preferences = {
    length: answer.length || 'normal',
    format: answer.format || 'paragraph',
    citations: answer.citations || 'normal',
    auto_summary_table: !!answer.autoSummaryTable,
  };
  const language = normalizeAnswerLanguage(answer.language);
  if (language) {
    preferences.language = language;
  }
  return preferences;
};

const normalizeAnswerLanguage = (value) => {
  const raw = typeof value === 'string' ? value.trim() : '';
  const key = raw.toLowerCase().replace(/_/g, '-');
  if (!raw || ['auto', 'default', 'system'].includes(key)) return '';
  if (['zh', 'zh-cn', 'zh-hans', 'chinese', 'simplified-chinese'].includes(key)) return 'Chinese';
  if (['en', 'en-us', 'english'].includes(key)) return 'English';
  return raw;
};

const App = ({ notebookId = 'default', onBack, onReady }) => {
  const { useTweaks } = window;
  const initialTweaksRef = useRef(null);
  if (!initialTweaksRef.current) {
    initialTweaksRef.current = { ...FOLIO_DEFAULTS, theme: getInitialFolioTheme() };
  }
  const [t, setTweak] = useTweaks(initialTweaksRef.current);
  const winW = useWindowWidth();
  const [settings, setSettingsState] = useState(() => getFolioSettingsSnapshot());
  const [notebookInfo, setNotebookInfo] = useState(null); // { id, name, description }

  useEffect(() => {
    const getSettings = window.FOLIO_SETTINGS?.getSettings || window.getSettings;
    const subscribeSettings = window.FOLIO_SETTINGS?.subscribeSettings || window.subscribeSettings;
    if (!getSettings || !subscribeSettings) return undefined;
    setSettingsState(getSettings());
    return subscribeSettings(setSettingsState);
  }, []);

  // Load theme from localStorage (overrides tweak panel defaults)
  useEffect(() => {
    const version = localStorage.getItem('folio.themeDefaultVersion');
    const savedTheme = localStorage.getItem('folio.theme');
    if (version !== THEME_DEFAULT_VERSION) {
      const osPrefersLight = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches;
      const defaultTheme = osPrefersLight ? 'paper' : 'midnight';
      localStorage.setItem('folio.theme', defaultTheme);
      localStorage.setItem('folio.themeDefaultVersion', THEME_DEFAULT_VERSION);
      setTweak('theme', defaultTheme);
      return;
    }
    if (savedTheme && window.THEMES[savedTheme]) {
      setTweak('theme', savedTheme);
    } else if (!savedTheme) {
      const osPrefersLight = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches;
      if (osPrefersLight) setTweak('theme', 'paper');
    }
  }, []);
  const isNarrowLayout = winW < NARROW_LAYOUT_BREAKPOINT;
  const effectiveShowStudio = t.showStudio !== false;
  const [slideInLeft, setSlideInLeft] = useState(false);
  const [slideInRight, setSlideInRight] = useState(false);
  const hoverCloseRef = useRef(null);
  const readyNotifiedRef = useRef(false);

  // Close slide-in panels when resizing above the narrow breakpoint
  useEffect(() => {
    if (!isNarrowLayout) { setSlideInLeft(false); setSlideInRight(false); }
  }, [isNarrowLayout]);

  // Close on Escape
  useEffect(() => {
    if (!slideInLeft && !slideInRight) return;
    const onKey = (e) => { if (e.key === 'Escape') { setSlideInLeft(false); setSlideInRight(false); } };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [slideInLeft, slideInRight]);

  const [sources, setSources] = useState([]);
  const [sourcesLoading, setSourcesLoading] = useState(true);
  const [sourcesError, setSourcesError] = useState(null);
  const [usingFallback, setUsingFallback] = useState(false);
  const [includedIds, setIncludedIds] = useState([]);
  const sourceScopeInitializedRef = useRef(false);
  const [thinking, setThinking] = useState(false);
  const [view, setView] = useState('chat'); // chat | quiz | flashcards | summary  (legacy artifact viewers, reachable via Tweaks only)
  const [uploadOpen, setUploadOpen] = useState(false);

  // Chat persistence (localStorage, last 50 messages)
  const CHAT_STORAGE_KEY = 'folio.chat.' + notebookId + '.v1';
  const CHAT_SCROLL_KEY = 'folio.chatScroll.' + notebookId + '.v1';

  const loadStoredChat = () => {
    try {
      const raw = localStorage.getItem(CHAT_STORAGE_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : [];
    } catch (e) {
      console.warn('Failed to load stored chat', e);
      return [];
    }
  };

  const [chat, setChat] = useState(() => loadStoredChat());
  const [chatDraft, setChatDraft] = useState('');
  const [conversationId, setConversationId] = useState('');
  const [lastSavedSig, setLastSavedSig] = useState('');

  // Persist chat to localStorage on every change
  useEffect(() => {
    try {
      const toPersist = chat.slice(-50);
      if (toPersist.length > 0) {
        localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(toPersist));
      } else {
        localStorage.removeItem(CHAT_STORAGE_KEY);
      }
    } catch (e) {
      console.warn('Failed to persist chat', e);
    }
  }, [chat]);

  const clearChat = () => {
    setChat([]);
    try { localStorage.removeItem(CHAT_STORAGE_KEY); } catch (e) {}
  };

  // Load chat from backend on notebook entry
  useEffect(() => {
    if (!notebookId) return;
    (async () => {
      try {
        const data = await window.FOLIO_API.listConversations(notebookId);
        const conversations = data?.conversations || data || [];
        if (conversations.length > 0) {
          const latest = conversations[0];
          setConversationId(latest.id || '');
          const msgs = latest.messages || [];
          if (msgs.length > 0) {
            setChat(prev => prev.length > 0 ? prev : msgs);
            setLastSavedSig(msgs.map(m => m.id || '').join('|'));
          }
        }
      } catch (e) { console.warn('Failed to load conversations:', e); }
    })();
  }, [notebookId]);

  // Auto-save chat to backend every 60 seconds, updating the same conversation
  // row via a stable conversation id so no duplicate snapshots accumulate.
  useEffect(() => {
    const interval = setInterval(async () => {
      const sig = chat.map(m => m.id || '').join('|');
      if (chat.length > 0 && sig !== lastSavedSig) {
        try {
          const saved = await window.FOLIO_API.saveConversation(
            `Chat - ${new Date().toLocaleDateString()}`,
            chat.slice(-100),
            includedIds,
            notebookId,
            conversationId
          );
          if (saved && saved.id) setConversationId(saved.id);
          setLastSavedSig(sig);
        } catch (e) { console.warn('Auto-save failed:', e); }
      }
    }, 60000);
    return () => clearInterval(interval);
  }, [chat, lastSavedSig, notebookId, includedIds, conversationId]);

  // Save chat on beforeunload
  useEffect(() => {
    const handler = () => {
      if (chat.length > 0) {
        const payload = JSON.stringify({
          id: conversationId,
          notebook_id: notebookId,
          title: `Chat - ${new Date().toLocaleDateString()}`,
          messages: chat.slice(-100),
          source_document_ids: includedIds,
        });
        // sendBeacon can't carry X-Auth-Token; use the token-injecting fetch
        // wrapper with keepalive so the save still lands during unload.
        fetch(`${API_BASE}/api/conversations`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: payload,
          keepalive: true,
        });
      }
    };
    window.addEventListener('beforeunload', handler);
    return () => window.removeEventListener('beforeunload', handler);
  }, [chat, notebookId, includedIds, conversationId]);

  const sourceScopeKey = `folio.scope.${notebookId}.v1`;
  const resolveInitialSourceScope = (mapped, currentIds = []) => {
    const readyIds = mapped.filter(s => s.status === 'ready').map(s => s.id);
    const validReady = new Set(readyIds);
    const cleanedCurrent = currentIds.filter(id => validReady.has(id));
    const mode = settings?.sources?.openScope || 'last-or-all';
    if (mode === 'all') return readyIds;
    if (mode === 'none') return [];
    try {
      const saved = JSON.parse(localStorage.getItem(sourceScopeKey) || '[]');
      const cleanedSaved = Array.isArray(saved) ? saved.filter(id => validReady.has(id)) : [];
      if (cleanedSaved.length > 0) return cleanedSaved;
    } catch (e) {}
    return cleanedCurrent.length > 0 ? cleanedCurrent : readyIds;
  };

  useEffect(() => {
    sourceScopeInitializedRef.current = false;
  }, [notebookId, settings?.sources?.openScope]);

  useEffect(() => {
    if (sourcesLoading || sources.length === 0) return;
    sourceScopeInitializedRef.current = true;
    setIncludedIds(prev => resolveInitialSourceScope(sources, prev));
  }, [settings?.sources?.openScope]);

  useEffect(() => {
    if (sourcesLoading || sources.length === 0 || !sourceScopeInitializedRef.current) return;
    try { localStorage.setItem(sourceScopeKey, JSON.stringify(includedIds)); } catch (e) {}
  }, [includedIds, sourcesLoading, sources.length, sourceScopeKey]);

  // Fetch real documents on mount
  const refreshSources = async () => {
    setSourcesLoading(true);
    setSourcesError(null);
    try {
      const docs = await window.FOLIO_API.listDocuments(notebookId);
      _colorIdx = 0;
      const mapped = docs.map(mapDocumentToSource);
      setSources(mapped);
      // Apply the configured source scope on first load, then preserve user changes on refreshes.
      setIncludedIds(prev => {
        const validSet = new Set(mapped.map(s => s.id));
        const cleaned = prev.filter(id => validSet.has(id));
        if (!sourceScopeInitializedRef.current) {
          sourceScopeInitializedRef.current = true;
          return resolveInitialSourceScope(mapped, cleaned);
        }
        return cleaned;
      });
      setUsingFallback(false);
    } catch (err) {
      console.error('Failed to load documents:', err);
      setSourcesError(err.message);
      setUsingFallback(true);
      // Keep whatever sources are already shown; don't wipe the workspace on a
      // transient backend failure.
    } finally {
      setSourcesLoading(false);
    }
  };

  useEffect(() => { refreshSources(); }, [notebookId]);

  // Latest refreshSources, so the background-job poller never captures a stale
  // closure after notebook changes.
  const refreshSourcesRef = useRef(refreshSources);
  useEffect(() => { refreshSourcesRef.current = refreshSources; });

  const pendingJobsRef = useRef([]);

  // Recover jobs still queued/running from a previous workspace visit (or an
  // earlier app run) so their completion is tracked after a remount.
  useEffect(() => {
    if (!notebookId) return;
    (async () => {
      try {
        const jobs = await window.FOLIO_API.listJobs(notebookId);
        const running = (Array.isArray(jobs) ? jobs : [])
          .filter(j => j.status === 'queued' || j.status === 'running')
          .map(j => j.id);
        if (running.length) {
          pendingJobsRef.current = [...new Set([...pendingJobsRef.current, ...running])];
        }
      } catch (e) { console.warn('Failed to recover running jobs:', e); }
    })();
  }, [notebookId]);

  // Poll background ingestion jobs at low frequency until they finish, then
  // refresh so the real completed/failed state replaces the placeholder.
  useEffect(() => {
    const id = setInterval(async () => {
      const jobs = pendingJobsRef.current;
      if (jobs.length === 0) return;
      const remaining = [];
      for (const jobId of jobs) {
        try {
          const job = await window.FOLIO_API.getJob(jobId);
          if (job && (job.status === 'completed' || job.status === 'failed')) {
            if (job.status === 'failed' && job.error_message) {
              alert('A background import failed: ' + job.error_message);
            }
            refreshSourcesRef.current();
          } else {
            remaining.push(jobId);
          }
        } catch (e) {
          remaining.push(jobId);
        }
      }
      pendingJobsRef.current = remaining;
    }, 4000);
    return () => clearInterval(id);
  }, []);

  // Check LLM configuration status (on mount and after settings save).
  const refreshLlmStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/config/llm-status`);
      const data = await res.json();
      setLlmConfigured(data.configured);
    } catch (e) { /* backend not running */ }
  };

  useEffect(() => { refreshLlmStatus(); }, []);

  // Studio drawer state
  const [savedItems, setSavedItems] = useState([]);
  const [generatingKind, setGeneratingKind] = useState(null);
  const generatingCounts = React.useMemo(() => {
    return savedItems.reduce((acc, item) => {
      if (item && item._pending && item.kind) {
        acc[item.kind] = (acc[item.kind] || 0) + 1;
      }
      return acc;
    }, {});
  }, [savedItems]);
  const [justAddedId, setJustAddedId] = useState(null);
  const [expandedSavedId, setExpandedSavedId] = useState(null);
  const expandedItem = expandedSavedId ? savedItems.find(s => s.id === expandedSavedId) : null;
  const [reviewItems, setReviewItems] = useState([]);
  const [expandedReviewItemId, setExpandedReviewItemId] = useState(null);
  const expandedReviewItem = expandedReviewItemId ? reviewItems.find(i => i.id === expandedReviewItemId) : null;
  const studioExpanded = expandedItem || expandedReviewItem;
  const [continueFromPrev, setContinueFromPrev] = useState(false);
  const [internetEnabled, setInternetEnabled] = useState(() => !!getFolioSettingsSnapshot()?.ai?.internetDefault);
  const [summaryModalOpen, setSummaryModalOpen] = useState(false);
  const [summaryDocIds, setSummaryDocIds] = useState([]);
  const [flashcardsModalOpen, setFlashcardsModalOpen] = useState(false);
  const [flashcardsDocIds, setFlashcardsDocIds] = useState([]);
  const [quizModalOpen, setQuizModalOpen] = useState(false);
  const [quizDocIds, setQuizDocIds] = useState([]);
  const [mindmapModalOpen, setMindmapModalOpen] = useState(false);
  const [mindmapDocIds, setMindmapDocIds] = useState([]);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [llmConfigured, setLlmConfigured] = useState(null); // null=unknown, true/false

  useEffect(() => {
    setInternetEnabled(!!settings?.ai?.internetDefault);
  }, [settings?.ai?.internetDefault, notebookId]);

  // Expose settings opener for Studio cards
  useEffect(() => { window._openSettings = () => setSettingsOpen(true); return () => { delete window._openSettings; }; }, []);

  // Load saved artifacts from backend on startup
  useEffect(() => {
    (async () => {
      try {
        const resp = await window.FOLIO_API.listArtifacts(notebookId);
        const artifacts = resp.artifacts || [];
        const mapped = artifacts.map(a => {
          const normalizedArtifact = normalizeArtifactIdentity(a);
          const artifactId = getArtifactId(normalizedArtifact);
          return {
            id: artifactId,
            kind: normalizedArtifact.kind,
            title: normalizedArtifact.title,
            scopeCount: (normalizedArtifact.source_document_ids || []).length,
            when: normalizedArtifact.created_at ? new Date(normalizedArtifact.created_at).toLocaleDateString() : 'unknown',
            _artifact: normalizedArtifact,
          };
        });
        setSavedItems(mapped);
      } catch (e) {
        console.warn('Failed to load saved artifacts:', e);
      }
    })();
  }, [notebookId]);

  useEffect(() => {
    (async () => {
      try {
        const resp = await window.FOLIO_API.listReviewItems(notebookId);
        setReviewItems(resp.review_items || []);
      } catch (e) {
        console.warn('Failed to load review basket:', e);
        setReviewItems([]);
      }
    })();
  }, [notebookId]);

  // Notebook title (persisted)
  const [notebookTitle, setNotebookTitle] = useState(() => {
    try { return localStorage.getItem(`folio.notebookTitle.${notebookId}`) || 'Study Companion'; }
    catch (e) { return 'Study Companion'; }
  });
  const handleNotebookTitleChange = (title) => {
    const trimmed = title.trim() || 'Study Companion';
    const previous = notebookTitle;
    setNotebookTitle(trimmed);
    try { localStorage.setItem(`folio.notebookTitle.${notebookId}`, trimmed); } catch (e) {}
    if (notebookId !== 'default') {
      window.FOLIO_API.updateNotebook(notebookId, { name: trimmed }).catch(e => {
        console.warn('Failed to sync notebook title to backend:', e);
        // Backend is the source of truth — roll back the optimistic rename.
        setNotebookTitle(previous);
        try { localStorage.setItem(`folio.notebookTitle.${notebookId}`, previous); } catch (e2) {}
        alert('Failed to rename notebook: ' + (e.message || 'Unknown error'));
      });
    }
  };

  // Load notebook metadata when notebookId changes
  useEffect(() => {
    if (notebookId !== 'default') {
      (async () => {
        try {
          const info = await window.FOLIO_API.getNotebook(notebookId);
          if (info) {
            setNotebookInfo(info);
            if (info.name) {
              setNotebookTitle(info.name);
              try { localStorage.setItem(`folio.notebookTitle.${notebookId}`, info.name); } catch (e) {}
            }
          }
        } catch (e) { console.warn('Failed to load notebook info:', e); }
      })();
    }
  }, [notebookId]);

  const currentNotebookName = notebookTitle !== 'Study Companion' ? notebookTitle : (notebookInfo?.name || notebookTitle);

  // Resizable column widths (persisted)
  const readW = (k, fallback) => {
    const v = Number(typeof localStorage !== 'undefined' && localStorage.getItem(k));
    return Number.isFinite(v) && v > 0 && v >= 200 ? v : fallback;
  };
  const [leftW, setLeftW] = useState(() => {
    const saved = readW('folio.leftW.v2', -1);
    if (saved > 0) return saved;
    return Math.max(280, Math.min(420, Math.floor(winW * 0.22)));
  });
  const [rightWCol, setRightWCol] = useState(() => {
    const saved = readW('folio.rightWCol.v2', -1);
    if (saved > 0) return saved;
    return Math.max(240, Math.min(340, Math.floor(winW * 0.19)));
  });
  const [rightWExp, setRightWExp] = useState(() => {
    const saved = readW('folio.rightWExp.v2', -1);
    if (saved > 0) return saved;
    return Math.max(380, Math.min(600, Math.floor(winW * 0.35)));
  });
  const [resizing, setResizing]   = useState(false);
  useEffect(() => { try { localStorage.setItem('folio.leftW.v2',     String(leftW)); }     catch (e) {} }, [leftW]);
  useEffect(() => { try { localStorage.setItem('folio.rightWCol.v2', String(rightWCol)); } catch (e) {} }, [rightWCol]);
  useEffect(() => { try { localStorage.setItem('folio.rightWExp.v2', String(rightWExp)); } catch (e) {} }, [rightWExp]);
  const rightW    = studioExpanded ? rightWExp : rightWCol;
  const setRightW = studioExpanded ? setRightWExp : setRightWCol;

  const resetSourceScope = () => {
    try { localStorage.removeItem(sourceScopeKey); } catch (e) {}
    const readyIds = sources.filter(s => s.status === 'ready').map(s => s.id);
    setIncludedIds(readyIds);
    sourceScopeInitializedRef.current = true;
  };

  // The notebook chat is always global ...derive a "primary" source for citation samples
  // and for legacy artifact panels (plan/quiz/etc.) from the first included source.
  const primarySource = sources.find(s => includedIds.includes(s.id)) || sources[0];
  const NOTEBOOK_SUBTITLE = 'Your personal study notebook';

  // Apply theme CSS variables (delegated to themes.jsx's applyFolioTheme).
  useLayoutEffect(() => {
    window.applyFolioTheme(t.theme);
    if (typeof onReady !== 'function' || readyNotifiedRef.current) return;
    const raf = window.requestAnimationFrame || ((cb) => window.setTimeout(cb, 0));
    const cancelRaf = window.cancelAnimationFrame || window.clearTimeout;
    const frame = raf(() => {
      readyNotifiedRef.current = true;
      onReady();
    });
    return () => cancelRaf(frame);
  }, [t.theme, onReady]);

  // Advance any processing (temp) sources with a placeholder progress bar.
  // Real completion comes from the upload job — never fake-mark "ready" here.
  useEffect(() => {
    if (!sources.some(s => s.status === 'processing')) return;
    const id = setInterval(() => {
      setSources(s => s.map(src => {
        if (src.status !== 'processing') return src;
        const next = Math.min(0.95, (src.progress || 0) + 0.03);
        const labels = [
          [0.2, 'Extracting text...'],
          [0.45, 'Analyzing structure...'],
          [0.7, 'Chunking content...'],
          [0.9, 'Creating embeddings...'],
        ];
        const label = labels.find(([cap]) => next < cap)?.[1] || 'Finalising index...';
        return { ...src, progress: next, progressLabel: label };
      }));
    }, 1200);
    return () => clearInterval(id);
  }, [sources]);

  // Shared helpers for handleSend and handleRegenerateTurn
  const buildCitations = (citationResults, srcs) => {
    const sourceById = {};
    for (const s of srcs) sourceById[s.id] = s.title || s.originalFilename || s.id;
    return (citationResults || []).map((c, i) => {
      const sourceType = c.source_type || 'book';
      const refLabel = c.ref_label || (sourceType === 'web' ? `W${i + 1}` : `S${i + 1}`);
      if (sourceType === 'web') {
        let host = c.site_name || '';
        try {
          if (!host && c.url) host = new URL(c.url).hostname.replace(/^www\./, '');
        } catch (e) {}
        return {
          src: c.title || host || c.url || 'Internet source',
          loc: c.retrieved_at ? `retrieved ${new Date(c.retrieved_at).toLocaleDateString()}` : '',
          section: host || 'web',
          chunk_id: c.chunk_id,
          text: c.text_snippet || '',
          source_type: 'web',
          ref_label: refLabel,
          url: c.url || '',
          site_name: host,
          published_date: c.published_date || '',
          retrieved_at: c.retrieved_at || '',
        };
      }
      let loc = '';
      if (c.page_start > 0 || c.page_end > 0) {
        loc = c.page_start === c.page_end ? `p. ${c.page_start}` : `pp. ${c.page_start}-${c.page_end}`;
      }
      return {
        src: sourceById[c.document_id] || c.document_id,
        loc,
        section: c.section_path || '',
        chunk_id: c.chunk_id,
        text: c.text_snippet || '',
        source_type: 'book',
        ref_label: refLabel,
      };
    });
  };

  const buildAssistantMessage = (result, documentIds, allowWeb, quizContext, citations) => {
    const answer = result.answer || 'No answer returned.';
    const isAgent = result.used_llm === true;
    return {
      id: 'a' + Date.now(),
      role: 'assistant',
      text: answer,
      citations: citations,
      plan: null,
      _intent: result.planner_task || '',
      _plannerTask: result.planner_task || '',
      _agent: isAgent,
      _toolTrace: result.tool_trace || [],
      _allowWeb: allowWeb,
      _webUsed: citations.some(c => c.source_type === 'web'),
      _quizContext: quizContext || null,
      _sourceDocumentIds: documentIds,
    };
  };

  const buildErrorMessage = (err) => {
    let errorMsg = err.message || 'Unknown error';
    let agentErrorCode = '';
    if (err.detail && err.detail.code) {
      const d = err.detail;
      if (d.code === 'llm_not_configured') {
        errorMsg = 'Agent unavailable. Enable an LLM provider to use conversational QA.\n\nSet LLM_REMOTE_ENABLED=true and LLM_API_KEY in backend/.env';
        agentErrorCode = 'llm_not_configured';
      } else if (d.code === 'llm_provider_error') {
        errorMsg = `LLM provider error: ${d.message || 'Unknown provider error'}`;
        agentErrorCode = 'llm_provider_error';
      } else {
        agentErrorCode = d.code;
      }
    }
    return { errorMsg, agentErrorCode };
  };

  // Query routing is handled by the backend /api/ask endpoint.
  // All intent classification, retrieval strategy, and answer generation
  // live in backend/app/services/qa/.
  const handleSend = async (text, options = {}) => {
    if (thinking) return; // prevent concurrent asks from interleaving answers
    const followupContext = continueFromPrev && typeof continueFromPrev === 'object' ? continueFromPrev : null;
    const documentIds = options.documentIdsOverride
      || (followupContext?.documentIds?.length ? followupContext.documentIds : includedIds);
    if (documentIds.length === 0) return;

    // Gate: require LLM to be configured for chat
    if (llmConfigured === false) {
      setSettingsOpen(true);
      return;
    }

    const uid = 'u' + Date.now();
    setChat(c => [...c, { id: uid, role: 'user', text, _quizContext: options.quizContext || null, _sourceDocumentIds: documentIds }]);
    setThinking(true);

    try {
      // Build conversation history when the next query is an explicit/automatic follow-up.
      // Quiz elaborate is self-contained (the prompt carries the question/answer), so
      // it sets follow_up:true to skip the evidence gate without pulling prior chat.
      const isQuiz = options.origin === 'quiz';
      let history = [];
      const shouldFollowUp = isQuiz
        || ((continueFromPrev || settings?.answer?.followupMode === 'automatic') && options.origin !== 'mindmap');
      if (shouldFollowUp && !isQuiz) {
        const lastUser = followupContext?.userText
          ? { text: followupContext.userText }
          : [...chat].reverse().find(m => m.role === 'user');
        const lastAssistant = followupContext?.assistantText
          ? { text: followupContext.assistantText, _intent: followupContext.intent || '' }
          : [...chat].reverse().find(m => m.role === 'assistant');
        if (lastUser && lastAssistant) {
          history = [
            { role: 'user', text: lastUser.text || '', intent: '' },
            { role: 'assistant', text: lastAssistant.text || '', intent: lastAssistant._intent || '' },
          ];
        }
      }
      if (continueFromPrev) setContinueFromPrev(false); // consume the flag

  // Query routing is handled by the backend /api/ask endpoint.
      const topK = Number.isFinite(options.topKOverride) ? options.topKOverride : 10;
      const allowWeb = Object.prototype.hasOwnProperty.call(options, 'allowWebOverride')
        ? !!options.allowWebOverride
        : !!internetEnabled;
      const result = await window.FOLIO_API.ask(text, notebookId, documentIds, topK, history, {
        allowWeb,
        preferences: buildAnswerPreferences(settings),
        followUp: !!shouldFollowUp,
      });

      const citations = buildCitations(result.citations, sources);

      const msg = buildAssistantMessage(result, documentIds, allowWeb, options.quizContext, citations);
      setChat(c => [...c, msg]);
    } catch (err) {
      console.error('Agent ask failed:', err);
      const { errorMsg, agentErrorCode } = buildErrorMessage(err);
      setChat(c => [...c, {
        id: 'a' + Date.now(),
        role: 'assistant',
        text: errorMsg,
        citations: [],
        plan: null,
        _agent: false,
        _agentError: agentErrorCode,
        _quizContext: options.quizContext || null,
        _sourceDocumentIds: documentIds,
      }]);
    } finally {
      setThinking(false);
    }
  };

  const handleDeleteTurn = (assistantMsgId) => {
    setChat(c => {
      const idx = c.findIndex(m => m.id === assistantMsgId);
      if (idx < 0) return c;
      if (idx > 0 && c[idx - 1].role === 'user') {
        return [...c.slice(0, idx - 1), ...c.slice(idx + 1)];
      }
      return [...c.slice(0, idx), ...c.slice(idx + 1)];
    });
  };

  const handleRegenerateTurn = async (assistantMsgId) => {
    if (thinking) return; // prevent a regenerate racing an in-flight ask
    // Find the assistant message and its paired user message
    const idx = chat.findIndex(m => m.id === assistantMsgId);
    if (idx <= 0) return;
    const userMsg = chat[idx - 1];
    if (userMsg.role !== 'user') return;

    const documentIds = userMsg._sourceDocumentIds || includedIds;
    if (documentIds.length === 0) return;
    if (llmConfigured === false) {
      setSettingsOpen(true);
      return;
    }

    // Build history from all messages before the user question
    const history = [];
    for (let i = 0; i < idx - 1; i++) {
      const m = chat[i];
      history.push({ role: m.role, text: m.text || '', intent: m._intent || '' });
    }

    // Remove old assistant message, keep user message
    setChat(c => c.filter(m => m.id !== assistantMsgId));
    setThinking(true);

    try {
      const topK = 10;
      const allowWeb = !!internetEnabled;
      // Re-generating a turn that came after a prior answer is itself a
      // follow-up; otherwise the rewrite re-triggers the "insufficient info" gate.
      const wasFollowUp = history.some(m => m.role === 'assistant');
      const result = await window.FOLIO_API.ask(userMsg.text, notebookId, documentIds, topK, history, {
        allowWeb,
        preferences: buildAnswerPreferences(settings),
        followUp: wasFollowUp,
      });

      const citations = buildCitations(result.citations, sources);
      const msg = buildAssistantMessage(result, documentIds, allowWeb, userMsg._quizContext, citations);

      // Insert new answer right after the user message
      setChat(c => {
        const userIdx = c.findIndex(m => m.id === userMsg.id);
        if (userIdx < 0) return [...c, msg];
        return [...c.slice(0, userIdx + 1), msg, ...c.slice(userIdx + 1)];
      });
    } catch (err) {
      console.error('Regenerate failed:', err);
      const { errorMsg, agentErrorCode } = buildErrorMessage(err);
      const errorMsgObj = {
        id: 'a' + Date.now(),
        role: 'assistant',
        text: errorMsg,
        citations: [],
        plan: null,
        _agent: false,
        _agentError: agentErrorCode,
        _quizContext: userMsg._quizContext || null,
        _sourceDocumentIds: documentIds,
      };
      setChat(c => {
        const userIdx = c.findIndex(m => m.id === userMsg.id);
        if (userIdx < 0) return [...c, errorMsgObj];
        return [...c.slice(0, userIdx + 1), errorMsgObj, ...c.slice(userIdx + 1)];
      });
    } finally {
      setThinking(false);
    }
  };

  const toggleInclude = (id) => {
    // Only ready sources can be included in the ask scope.
    const src = sources.find(s => s.id === id);
    if (!src || src.status !== 'ready') return;
    setIncludedIds(ids => ids.includes(id) ? ids.filter(x => x !== id) : [...ids, id]);
  };

  const selectAll = () => {
    const readyIds = sources.filter(s => s.status === 'ready').map(s => s.id);
    if (includedIds.length === readyIds.length) setIncludedIds([]);
    else setIncludedIds(readyIds);
  };

  const handleDeleteSource = async (id) => {
    try {
      await window.FOLIO_API.deleteDocument(id);
    } catch (err) {
      console.warn('Backend delete failed for', id, err);
      alert('Failed to delete source: ' + (err.message || 'Unknown error'));
      return;
    }
    // Only hide the source once the backend has confirmed the delete.
    setSources(prev => prev.filter(s => s.id !== id));
    setIncludedIds(prev => prev.filter(x => x !== id));
  };

  const handleDeleteArtifact = async (id) => {
    const removeLocalArtifact = () => {
      setSavedItems(prev => prev.filter(s => {
        const itemArtifactId = getArtifactId(s?._artifact) || s?.id;
        return s?.id !== id && itemArtifactId !== id;
      }));
      setReviewItems(prev => prev.filter(i => i.artifact_id !== id));
      if (expandedSavedId === id) setExpandedSavedId(null);
    };
    try {
      await window.FOLIO_API.deleteArtifact(id);
      removeLocalArtifact();
    } catch (err) {
      if (err?.status === 404) {
        removeLocalArtifact();
        return;
      }
      console.error('Failed to delete artifact:', err);
      alert(`Failed to delete saved item: ${err?.message || 'Unknown error'}`);
    }
  };

  const mergeReviewItem = (item) => {
    setReviewItems(prev => [item, ...prev.filter(i => i.id !== item.id)]);
    return item;
  };

  const handleSaveReviewItem = async (payload) => {
    if (!payload?.artifact_id) {
      throw new Error('This item is missing an artifact id and cannot be saved for review.');
    }
    const item = await window.FOLIO_API.saveReviewItem({
      ...payload,
      notebook_id: notebookId,
    });
    return mergeReviewItem(item);
  };

  const handleDeleteReviewItem = async (id) => {
    await window.FOLIO_API.deleteReviewItem(id);
    setReviewItems(prev => prev.filter(i => i.id !== id));
    if (expandedReviewItemId === id) setExpandedReviewItemId(null);
  };

  const handleOpenSavedItem = (id) => {
    setExpandedReviewItemId(null);
    setExpandedSavedId(id);
  };

  const handleOpenReviewItem = (id) => {
    setExpandedSavedId(null);
    setExpandedReviewItemId(id);
  };

  const handleCloseStudioExpanded = () => {
    setExpandedSavedId(null);
    setExpandedReviewItemId(null);
  };

  // Upload job management
  /** Called when a file is selected ...adds a temporary processing source to the sidebar. */
  const handleUploadStart = (tmpId, filename) => {
    setSources(prev => [{
      id: tmpId,
      title: cleanFileName(filename),
      originalFilename: filename,
      authors: '',
      edition: 'Processing',
      pages: 0,
      chunks: 0,
      status: 'processing',
      progress: 0,
      progressLabel: 'Uploading...',
      color: SOURCE_COLORS[0],
      summary: '',
      keyTopics: [],
      chapters: [],
      toc: false,
      frontMatter: false,
      addedRel: 'just now',
      _temp: true,
    }, ...prev]);
  };

  /** Called when upload completes (success or failure). */
  const handleUploadComplete = (result, tmpId, errorMsg) => {
    if (errorMsg || !result) {
      // Surface the failure, then drop the temporary row; a real (possibly
      // failed) document, with a real id that can be retried/deleted, is
      // surfaced by the refresh below.
      if (errorMsg) alert('Import failed: ' + errorMsg);
      setSources(prev => prev.filter(s => s.id !== tmpId));
      refreshSources();
      return;
    }
    if (result._async) {
      // Async ingestion finished — drop the temp row and refetch the real source list.
      setSources(prev => prev.filter(s => s.id !== tmpId));
      if (result.job_id) pendingJobsRef.current.push(result.job_id);
      if (result.document_id && settings?.sources?.autoIncludeUploads !== false) {
        setIncludedIds(prev => prev.includes(result.document_id) ? prev : [...prev, result.document_id]);
      }
      refreshSources();
      return;
    }
    // Replace temp source with real document
    _colorIdx = 0; // reset color cycling for consistency
    const mapped = mapDocumentToSource(result);
    setSources(prev => {
      const filtered = prev.filter(s => s.id !== tmpId);
      return [mapped, ...filtered];
    });
    if (settings?.sources?.autoIncludeUploads !== false) {
      setIncludedIds(prev => prev.includes(mapped.id) ? prev : [...prev, mapped.id]);
    }
  };

  // Studio: kick off a generation. Inline progress lives on the card; we don't navigate.
  const handleGenerate = (kind) => {
    if (includedIds.length === 0) return;
    if (usingFallback) {
      alert('The backend is not reachable. Please start the backend server and refresh the page.');
      return;
    }

    if (kind === 'summary') {
      setSummaryDocIds([...includedIds]);
      setSummaryModalOpen(true);
      setGeneratingKind('summary');
      return;
    }

    if (kind === 'flashcards') {
      setFlashcardsDocIds([...includedIds]);
      setFlashcardsModalOpen(true);
      setGeneratingKind('flashcards');
      return;
    }

    if (kind === 'quiz') {
      setQuizDocIds([...includedIds]);
      setQuizModalOpen(true);
      setGeneratingKind('quiz');
      return;
    }

    if (kind === 'mindmap') {
      setMindmapDocIds([...includedIds]);
      setMindmapModalOpen(true);
      setGeneratingKind('mindmap');
      return;
    }
  };

  // Handler for when artifact generation starts ...create optimistic placeholder
  const handleArtifactPendingStart = ({ kind, scope, documentId, sourceName, ...extras }) => {
    const requestId = extras.requestId || `${kind}_${documentId}_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    const tempId = '_pending_' + requestId;
    const sourcePart = sourceName ? `${sourceName} ...` : '';
    const scopePart = scope || (kind === 'flashcards' ? `${extras.count || 20} cards`
      : kind === 'quiz' ? `${extras.quiz_count || 10} questions`
      : 'whole document');
    const verb = kind === 'flashcards' ? 'flashcards'
      : kind === 'quiz' ? 'quiz'
      : kind === 'mindmap' ? 'mind map'
      : 'summary';
    const placeholderTitle = `Generating ${verb}: ${sourcePart}${scopePart}`;
    setSavedItems(prev => [{
      id: tempId,
      kind,
      title: placeholderTitle,
      scopeCount: extras.sourceCount || 1,
      when: 'generating...',
      _pending: true,
      _pendingDocId: documentId,
      _requestId: requestId,
    }, ...prev]);
  };

  // Handler for when artifact generation succeeds
  const handleArtifactGenerated = (result, documentId, requestId) => {
    const realId = getArtifactId(result) || ('g' + Date.now());
    const kind = result.kind || 'summary';
    const normalizedArtifact = normalizeArtifactIdentity({
      ...result,
      id: realId,
      artifact_id: result.artifact_id || realId,
    });
    const realItem = {
      id: realId,
      kind,
      title: result.title || (kind === 'flashcards' ? 'Flashcards' : 'Summary'),
      scopeCount: (result.source_document_ids || [documentId]).length,
      when: 'just now',
      _artifact: normalizedArtifact,
    };
    setSavedItems(prev => {
      const idx = prev.findIndex(s => s._pending && (
        requestId ? s._requestId === requestId : s._pendingDocId === documentId
      ));
      if (idx >= 0) {
        const next = [...prev];
        next[idx] = realItem;
        return next;
      }
      return [realItem, ...prev];
    });
    setJustAddedId(realId);
    if (settings?.studio?.autoExpandGenerated !== false) {
      setExpandedReviewItemId(null);
      setExpandedSavedId(realId);
    }
    setTimeout(() => setJustAddedId(curr => curr === realId ? null : curr), 2400);
  };

  const pushSavedArtifact = (artifact, when = 'just now') => {
    const normalizedArtifact = normalizeArtifactIdentity(artifact);
    const artifactId = getArtifactId(normalizedArtifact);
    const item = {
      id: artifactId,
      kind: normalizedArtifact.kind,
      title: normalizedArtifact.title || 'Saved note',
      scopeCount: (normalizedArtifact.source_document_ids || []).length,
      when,
      _artifact: normalizedArtifact,
    };
    setSavedItems(prev => [item, ...prev.filter(s => s.id !== item.id)]);
    setJustAddedId(item.id);
    setExpandedSavedId(item.id);
    setTimeout(() => setJustAddedId(curr => curr === item.id ? null : curr), 2400);
    return item;
  };

  const handleSaveTurn = async ({ title, messages, sourceDocumentIds = [], quizContext = null }) => {
    const content = { messages };
    if (quizContext) content.quiz_context = quizContext;
    const artifact = await window.FOLIO_API.createArtifact(
      'note',
      title || 'Saved chat turn',
      content,
      sourceDocumentIds,
      { origin: 'chat_save' },
      notebookId
    );
    pushSavedArtifact(artifact);
    return artifact;
  };

  const handleSaveNote = async (text) => {
    const title = text.split('\n')[0].slice(0, 60) || 'Margin note';
    const artifact = await window.FOLIO_API.createArtifact(
      'note',
      title,
      { text },
      [],
      { origin: 'margin_note' },
      notebookId
    );
    pushSavedArtifact(artifact);
    return artifact;
  };

  // Handler for when artifact generation fails for one source
  const handleArtifactFailed = (documentId, requestId) => {
    setSavedItems(prev => {
      const filtered = prev.filter(s => !(s._pending && (
        requestId ? s._requestId === requestId : s._pendingDocId === documentId
      )));
      return filtered;
    });
  };

  // Render center column ...chat is always the notebook chat, never source-specific.
  // Artifact views (plan/quiz/etc.) still draw on the primary included source for sample data.
  const buildQuizElaboratePayload = (question, answer = {}) => {
    const options = question?.options || [];
    const picked = options.find(o => o.id === answer?.picked);
    const correct = options.find(o => o.id === question?.correct);
    const selectedLabel = picked ? `${picked.id.toUpperCase()}: ${picked.text}` : '';
    const correctLabel = correct ? `${correct.id.toUpperCase()}: ${correct.text}` : (question?.correct || '');
    const choices = options.map(o => `${o.id.toUpperCase()}. ${o.text}`);
    const quizContext = {
      question: question?.prompt || question?.question || '',
      choices,
      selected_answer: selectedLabel,
      correct_answer: correctLabel,
      was_correct: typeof answer?.correct === 'boolean' ? answer.correct : null,
      explanation: question?.rationale || question?.explanation || '',
    };
    const prompt = [
      'Please elaborate on this quiz question and turn it into a study note.',
      '',
      `Question: ${quizContext.question}`,
      choices.length ? `Choices:\n${choices.join('\n')}` : '',
      selectedLabel ? `My answer: ${selectedLabel}` : '',
      correctLabel ? `Correct answer: ${correctLabel}` : '',
      typeof answer?.correct === 'boolean' ? `Result: ${answer.correct ? 'correct' : 'incorrect'}` : '',
      quizContext.explanation ? `Original explanation: ${quizContext.explanation}` : '',
      '',
      'Explain the underlying concept, why the correct answer works, why my answer was right or wrong, and what I should remember next time.',
    ].filter(Boolean).join('\n');
    return { prompt, quizContext };
  };

  const handleQuizElaborate = (question, answer = {}, item = null) => {
    const { prompt, quizContext } = buildQuizElaboratePayload(question, answer);
    const artifactDocIds = item?._artifact?.source_document_ids || [];
    setView('chat');
    handleSend(prompt, {
      documentIdsOverride: artifactDocIds.length ? artifactDocIds : includedIds,
      topKOverride: 14,
      origin: 'quiz',
      quizContext,
    });
  };

  const renderCenter = () => {
    if (view === 'quiz') return <QuizPanel source={primarySource} onBack={() => setView('chat')} onElaborate={handleQuizElaborate} />;
    if (view === 'flashcards') return <FlashcardsPanel source={primarySource} onBack={() => setView('chat')} />;
    if (view === 'summary') return <SummaryPanel source={primarySource} onBack={() => setView('chat')} />;
    if (view === 'mindmap') return <MindmapPanel source={primarySource} onBack={() => setView('chat')} />;
    return (
      <ChatCenter
        notebookTitle={currentNotebookName}
        notebookSubtitle={NOTEBOOK_SUBTITLE}
        includedCount={includedIds.length}
        totalSources={sources.length}
        prompts={usingFallback ? null : window.FOLIO_DATA.SUGGESTED_PROMPTS}
        usingFallback={usingFallback}
        chat={chat}
        setChat={setChat}
        thinking={thinking}
        setThinking={setThinking}
        onSend={handleSend}
        onSaveTurn={handleSaveTurn}
        onRegenerateTurn={handleRegenerateTurn}
        onDeleteTurn={handleDeleteTurn}
        sourceDocumentIds={includedIds}
        continueFromPrev={continueFromPrev}
        setContinueFromPrev={setContinueFromPrev}
        internetEnabled={internetEnabled}
        setInternetEnabled={setInternetEnabled}
        draftInput={chatDraft}
        setDraftInput={setChatDraft}
        enterToSend={settings?.answer?.enterToSend !== false}
        showAgentTrace={!!settings?.answer?.showAgentTrace}
        chatScrollKey={CHAT_SCROLL_KEY}
        onAddSource={() => setUploadOpen(true)}
        llmConfigured={llmConfigured}
        onRequireSettings={() => setSettingsOpen(true)}
      />
    );
  };

  const rightMaxCollapsed = Math.min(560, Math.floor(winW * 0.32));
  const leftMax = isNarrowLayout ? 0 : Math.min(760, Math.floor(winW * 0.40));
  const rightMax = studioExpanded
    ? (isNarrowLayout ? 0 : Math.min(1280, Math.floor(winW * 0.72)))
    : (isNarrowLayout ? 0 : rightMaxCollapsed);
  // Effective widths clamp the persisted sidebar widths to a share of the
  // viewport so the three columns shrink (instead of clipping) as the window
  // narrows; the 980px overlay switch is the final "hide" state.
  const leftWEff = Math.min(leftW, Math.max(240, Math.floor(winW * 0.30)));
  const rightWEff = Math.min(rightW, Math.max(220, Math.floor(winW * 0.28)));
  const layoutSettings = settings?.layout || {};
  const studioSettings = settings?.studio || {};
  const studioGenerationDefaults = {
    summary: { length: studioSettings.summaryLength || 'medium', style: studioSettings.summaryStyle || 'structured' },
    flashcards: { count: clampNumber(studioSettings.flashcardCount, 20, 10, 50) },
    quiz: { quiz_count: clampNumber(studioSettings.quizCount, 10, 5, 30), quiz_difficulty: studioSettings.quizDifficulty || 'medium' },
  };
  const fontScale = layoutSettings.fontScale || 'normal';
  const readingWidth = layoutSettings.readingWidth || 'normal';
  const workspaceClassName = cx(
    'h-full w-full overflow-hidden',
    `folio-font-${fontScale}`,
    `folio-reading-${readingWidth}`
  );

  return (
    <div
      className={workspaceClassName}
      style={{ color: 'var(--text-primary)', background: 'var(--surface-main)' }}
      data-screen-label="Folio Workspace"
      data-font-scale={fontScale}
      data-reading-width={readingWidth}
    >

      {/* Inline three-column layout (non-narrow) */}
      {!isNarrowLayout && (
        <div className="flex h-full">
          {/* Sources panel */}
          {t.showSidebar !== false && (
            <div
              className="shrink-0 h-full overflow-hidden"
              data-screen-label="Sources"
              style={{
                width: leftWEff,
                transition: resizing ? 'none' : 'width 280ms cubic-bezier(0.4, 0, 0.2, 1)',
              }}
            >
              <div style={{ width: leftWEff, minWidth: leftWEff }}>
                <ErrorBoundary>
                  <SourcesSidebar
                    sources={sources}
                    loading={sourcesLoading}
                    error={sourcesError}
                    includedIds={includedIds}
                    onToggleInclude={toggleInclude}
                    onSelectAll={selectAll}
                    onAdd={() => setUploadOpen(true)}
                    onUpload={() => setUploadOpen(true)}
                    onDeleteSource={handleDeleteSource}
                    notebookTitle={currentNotebookName}
                    onNotebookTitleChange={handleNotebookTitleChange}
                    onBack={onBack}
                    onOpenSettings={() => setSettingsOpen(true)}
                    onRetry={refreshSources}
                    sortMode={settings?.sources?.sort || 'recent'}
                  />
                </ErrorBoundary>
              </div>
            </div>
          )}

          {/* Left ColumnResizer */}
          {t.showSidebar !== false && (
            <ColumnResizer
              side="left"
              width={leftWEff}
              setWidth={setLeftW}
              min={240}
              max={leftMax}
              onDragStateChange={setResizing}
            />
          )}

          {/* Center ...the notebook chat */}
          <main
            className="flex-1 h-full"
            data-screen-label="Chat"
            style={{
              minWidth: 320,
              background: 'var(--surface-main)',
              transition: resizing ? 'none' : 'all 240ms ease',
            }}
          >
            <ErrorBoundary>{renderCenter()}</ErrorBoundary>
          </main>

          {/* Right ColumnResizer */}
          {effectiveShowStudio && (
            <ColumnResizer
              side="right"
              width={rightWEff}
              setWidth={setRightW}
              min={studioExpanded ? 320 : 240}
              max={rightMax}
              onDragStateChange={setResizing}
            />
          )}

          {/* Studio panel ...slides via width transition */}
          {effectiveShowStudio && (
            <div
              className="shrink-0 h-full overflow-hidden"
              data-screen-label="Studio"
              style={{
                width: rightWEff,
                transition: resizing ? 'none' : 'width 300ms ease-out',
              }}
            >
              <div style={{ width: rightWEff, minWidth: rightWEff, height: '100%', minHeight: 0 }}>
                <ErrorBoundary>
                  <StudioPanel
                    includedCount={includedIds.length}
                    savedItems={savedItems}
                    reviewItems={reviewItems}
                    generatingCounts={generatingCounts}
                    justAddedId={justAddedId}
                    expandedItem={expandedItem}
                    expandedReviewItem={expandedReviewItem}
                    onGenerate={handleGenerate}
                    onOpenSaved={handleOpenSavedItem}
                    onOpenReviewItem={handleOpenReviewItem}
                    onCloseExpanded={handleCloseStudioExpanded}
                    onAskFromMindmap={(prompt, documentIds) =>
                      handleSend(prompt, { documentIdsOverride: documentIds, origin: 'mindmap', topKOverride: 18 })
                    }
                    onQuizElaborate={handleQuizElaborate}
                    onSaveReviewItem={handleSaveReviewItem}
                    onDeleteReviewItem={handleDeleteReviewItem}
                    onDeleteArtifact={handleDeleteArtifact}
                    llmConfigured={llmConfigured}
                    generationDefaults={studioGenerationDefaults}
                    onSaveNote={handleSaveNote}
                  />
                </ErrorBoundary>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Overlay layout (narrow windows) */}
      {isNarrowLayout && (
        <div className="flex h-full relative overflow-hidden" style={{ isolation: 'isolate' }}>
          {/* Chat fills full width */}
          <main
            className="flex-1 h-full"
            data-screen-label="Chat"
            style={{ minWidth: 0, background: 'var(--surface-main)' }}
          >
            <ErrorBoundary>{renderCenter()}</ErrorBoundary>
          </main>

          {/* Left trigger tab ...Sources */}
          {t.showSidebar !== false && (
            <button
              onClick={() => { setSlideInLeft(true); setSlideInRight(false); }}
              onMouseEnter={() => { setSlideInLeft(true); setSlideInRight(false); }}
              className="slide-trigger left-edge"
              style={{
                width: 40, height: 104,
                opacity: slideInLeft ? 0 : 1,
                pointerEvents: slideInLeft ? 'none' : 'auto',
              }}
              title="Open Sources"
              aria-label="Open Sources panel"
            >
              <span style={{ writingMode: 'vertical-rl', fontSize: 'calc(10px * var(--folio-font-scale, 1))', letterSpacing: '0.14em',
                            textTransform: 'uppercase', fontFamily: 'Geist, ui-sans-serif' }}>
                Sources
              </span>
            </button>
          )}

          {/* Right trigger tab ...Studio */}
          {t.showStudio !== false && (
            <button
              onClick={() => { setSlideInRight(true); setSlideInLeft(false); }}
              onMouseEnter={() => { setSlideInRight(true); setSlideInLeft(false); }}
              className="slide-trigger right-edge"
              style={{
                width: 40, height: 104,
                opacity: slideInRight ? 0 : 1,
                pointerEvents: slideInRight ? 'none' : 'auto',
              }}
              title="Open Studio"
              aria-label="Open Studio panel"
            >
              <span style={{ writingMode: 'vertical-rl', fontSize: 'calc(10px * var(--folio-font-scale, 1))', letterSpacing: '0.14em',
                            textTransform: 'uppercase', fontFamily: 'Geist, ui-sans-serif' }}>
                Studio
              </span>
            </button>
          )}

          {/* Sources slide-in from left */}
          {t.showSidebar !== false && (
            <>
              {slideInLeft && (
                <div className="slide-backdrop" onClick={() => setSlideInLeft(false)} />
              )}
              <div className={`slide-panel-container left ${slideInLeft ? '' : 'closed'}`}
                   style={{ width: Math.min(380, winW - 48) }}
                   onMouseEnter={() => { if (hoverCloseRef.current) { clearTimeout(hoverCloseRef.current); hoverCloseRef.current = null; } }}
                   onMouseLeave={() => { hoverCloseRef.current = setTimeout(() => setSlideInLeft(false), 350); }}>
                <div className="h-full flex flex-col" style={{ background: 'var(--surface-left)' }}>
                  <div className="flex items-center justify-between px-5 py-3"
                       style={{ borderBottom: '1px solid var(--border-soft)' }}>
                    <span className="label">Sources</span>
                    <button onClick={() => setSlideInLeft(false)}
                            className="sans transition-colors"
                            style={{ fontSize: 'calc(13px * var(--folio-font-scale, 1))', color: 'var(--text-muted)' }}
                            title="Close">{'x'}</button>
                  </div>
                  <div className="flex-1 overflow-hidden">
                    <ErrorBoundary>
                      <SourcesSidebar
                        sources={sources}
                        loading={sourcesLoading}
                        error={sourcesError}
                        includedIds={includedIds}
                        onToggleInclude={toggleInclude}
                        onSelectAll={selectAll}
                        onAdd={() => { setUploadOpen(true); setSlideInLeft(false); }}
                        onUpload={() => { setUploadOpen(true); setSlideInLeft(false); }}
                        onDeleteSource={handleDeleteSource}
                        notebookTitle={notebookTitle}
                        onNotebookTitleChange={handleNotebookTitleChange}
                        onBack={onBack}
                        onOpenSettings={() => setSettingsOpen(true)}
                        onRetry={refreshSources}
                        sortMode={settings?.sources?.sort || 'recent'}
                      />
                    </ErrorBoundary>
                  </div>
                </div>
              </div>
            </>
          )}

          {/* Studio slide-in from right */}
          {t.showStudio !== false && (
            <>
              {slideInRight && (
                <div className="slide-backdrop" onClick={() => setSlideInRight(false)} />
              )}
              <div className={`slide-panel-container right ${slideInRight ? '' : 'closed'}`}
                   style={{ width: Math.min(studioExpanded ? 480 : 340, winW - 48) }}
                   onMouseEnter={() => { if (hoverCloseRef.current) { clearTimeout(hoverCloseRef.current); hoverCloseRef.current = null; } }}
                   onMouseLeave={() => { hoverCloseRef.current = setTimeout(() => setSlideInRight(false), 350); }}>
                <div className="h-full flex flex-col" style={{ background: 'var(--surface-right)' }}>
                  <div className="flex items-center justify-between px-5 py-3"
                       style={{ borderBottom: '1px solid var(--border-soft)' }}>
                    <span className="label">Studio</span>
                    <button onClick={() => setSlideInRight(false)}
                            className="sans transition-colors"
                            style={{ fontSize: 'calc(13px * var(--folio-font-scale, 1))', color: 'var(--text-muted)' }}
                            title="Close">{'x'}</button>
                  </div>
                  <div className="flex-1 overflow-hidden" style={{ minHeight: 0 }}>
                    <ErrorBoundary>
                      <StudioPanel
                        includedCount={includedIds.length}
                        savedItems={savedItems}
                        reviewItems={reviewItems}
                        generatingCounts={generatingCounts}
                        justAddedId={justAddedId}
                        expandedItem={expandedItem}
                        expandedReviewItem={expandedReviewItem}
                        onGenerate={handleGenerate}
                        onOpenSaved={handleOpenSavedItem}
                        onOpenReviewItem={handleOpenReviewItem}
                        onCloseExpanded={handleCloseStudioExpanded}
                        onAskFromMindmap={(prompt, documentIds) =>
                          handleSend(prompt, { documentIdsOverride: documentIds, origin: 'mindmap', topKOverride: 18 })
                        }
                        onQuizElaborate={handleQuizElaborate}
                        onSaveReviewItem={handleSaveReviewItem}
                        onDeleteReviewItem={handleDeleteReviewItem}
                        onDeleteArtifact={handleDeleteArtifact}
                        llmConfigured={llmConfigured}
                        generationDefaults={studioGenerationDefaults}
                        onSaveNote={handleSaveNote}
                      />
                    </ErrorBoundary>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* Upload modal */}
      <UploadModal
        open={uploadOpen}
        onClose={() => setUploadOpen(false)}
        onUploadStart={handleUploadStart}
        onComplete={handleUploadComplete}
        notebookId={notebookId}
      />

      {/* Summary config modal */}
      <window.SummaryConfigModal
        open={summaryModalOpen}
        onClose={() => { setSummaryModalOpen(false); setGeneratingKind(null); }}
        onSubmitted={() => { setSummaryModalOpen(false); /* keep generatingKind alive for Studio spinner */ }}
        documentIds={summaryDocIds}
        sourceNames={summaryDocIds.map(id => {
          const s = sources.find(x => x.id === id);
          return s ? (s.title || s.originalFilename || id) : id;
        })}
        onGenerate={handleArtifactGenerated}
        onPendingStart={handleArtifactPendingStart}
        onFailed={handleArtifactFailed}
        defaultSummaryLength={studioGenerationDefaults.summary.length}
        defaultSummaryStyle={studioGenerationDefaults.summary.style}
        notebookId={notebookId}
      />

      {/* Flashcards config modal */}
      <window.FlashcardsConfigModal
        open={flashcardsModalOpen}
        onClose={() => { setFlashcardsModalOpen(false); setGeneratingKind(null); }}
        onSubmitted={() => { setFlashcardsModalOpen(false); /* keep generatingKind alive for Studio spinner */ }}
        documentIds={flashcardsDocIds}
        sourceNames={flashcardsDocIds.map(id => {
          const s = sources.find(x => x.id === id);
          return s ? (s.title || s.originalFilename || id) : id;
        })}
        onGenerate={handleArtifactGenerated}
        onPendingStart={handleArtifactPendingStart}
        onFailed={handleArtifactFailed}
        defaultFlashcardCount={studioGenerationDefaults.flashcards.count}
        notebookId={notebookId}
      />

      {/* Quiz config modal */}
      <window.QuizConfigModal
        open={quizModalOpen}
        onClose={() => { setQuizModalOpen(false); setGeneratingKind(null); }}
        onSubmitted={() => { setQuizModalOpen(false); /* keep generatingKind alive */ }}
        documentIds={quizDocIds}
        sourceNames={quizDocIds.map(id => {
          const s = sources.find(x => x.id === id);
          return s ? (s.title || s.originalFilename || id) : id;
        })}
        onGenerate={handleArtifactGenerated}
        onPendingStart={handleArtifactPendingStart}
        onFailed={handleArtifactFailed}
        defaultQuizCount={studioGenerationDefaults.quiz.quiz_count}
        defaultQuizDifficulty={studioGenerationDefaults.quiz.quiz_difficulty}
        notebookId={notebookId}
      />

      {/* Mindmap config modal */}
      <window.MindmapConfigModal
        open={mindmapModalOpen}
        onClose={() => { setMindmapModalOpen(false); setGeneratingKind(null); }}
        onSubmitted={() => { setMindmapModalOpen(false); /* keep generatingKind alive */ }}
        documentIds={mindmapDocIds}
        sourceNames={mindmapDocIds.map(id => {
          const s = sources.find(x => x.id === id);
          return s ? (s.title || s.originalFilename || id) : id;
        })}
        onGenerate={handleArtifactGenerated}
        onPendingStart={handleArtifactPendingStart}
        onFailed={handleArtifactFailed}
        notebookId={notebookId}
      />

      {/* Tweaks */}
      {/* Settings modal */}
      <window.SettingsModal
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        theme={t.theme}
        onThemeChange={(v) => setTweak('theme', v)}
        themes={window.THEMES}
        onResetSourceScope={resetSourceScope}
        onConfigSaved={refreshLlmStatus}
      />
    </div>
  );
};

window.FolioApp = App;
