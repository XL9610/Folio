// Modals ...Upload, and the Plan/Quiz/Flashcards panels are in panels.jsx
const ModalShell = ({ open, onClose, closeable = true, children, width = 'max-w-[640px]', title }) => {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center fade-in">
      <div className="absolute inset-0 backdrop-blur-[2px]" style={{ background: 'rgba(0,0,0,0.62)' }} onClick={closeable ? onClose : undefined} />
      <div className={cx('relative bg-card border border-line rounded-lg shadow-soft w-[92vw]', width)}>
        {title && (
          <div className="flex items-center justify-between px-5 pt-4 pb-3 border-b border-line2">
            <div className="serif-display text-[18px]">{title}</div>
            {closeable && (
              <button onClick={onClose} className="text-muted hover:text-ink p-1 rounded-full"><Icon name="x" size={18} /></button>
            )}
          </div>
        )}
        {children}
      </div>
    </div>
  );
};

const UploadModal = ({ open, onClose, onComplete, onUploadStart, notebookId }) => {
  const [phase, setPhase] = React.useState('select'); // select | ingesting | done | error
  const [files, setFiles] = React.useState([]);       // supported files awaiting ingest
  const [rejected, setRejected] = React.useState([]); // unsupported files skipped
  const [jobs, setJobs] = React.useState([]);          // { name, status, error }
  const [summary, setSummary] = React.useState({ done: 0, failed: 0, background: 0, degraded: 0 });
  const [dragOver, setDragOver] = React.useState(false);
  const fileRef = React.useRef(null);

  const SUPPORTED_EXTENSIONS = ['.pdf', '.epub', '.pptx', '.docx', '.md', '.txt'];
  const FORMAT_LABELS = { '.pdf': 'PDF', '.epub': 'EPUB', '.pptx': 'PPTX', '.docx': 'DOCX', '.md': 'Markdown', '.txt': 'Text' };

  const extOf = (name) => '.' + String(name || '').split('.').pop().toLowerCase();
  const humanSize = (bytes) => (bytes >= 1048576 ? (bytes / 1048576).toFixed(1) + ' MB' : Math.max(1, Math.round(bytes / 1024)) + ' KB');

  const reset = () => { setPhase('select'); setFiles([]); setRejected([]); setJobs([]); setSummary({ done: 0, failed: 0, background: 0, degraded: 0 }); setDragOver(false); };
  const close = () => {
    onClose();
    // Keep ingestion running in the background; only reset when idle.
    if (phase !== 'ingesting') setTimeout(reset, 300);
  };

  const pickFiles = (list) => {
    const arr = Array.from(list || []);
    const supported = arr.filter((f) => SUPPORTED_EXTENSIONS.includes(extOf(f.name)));
    const skipped = arr.filter((f) => !SUPPORTED_EXTENSIONS.includes(extOf(f.name)));
    if (supported.length === 0) {
      setRejected(skipped);
      setPhase('error');
      return;
    }
    setFiles(supported);
    setRejected(skipped);
    setPhase('select');
  };

  const processFile = async (file, idx) => {
    const tid = `upload-${Date.now()}-${idx}`;
    onUploadStart?.(tid, file.name);
    setJobs((prev) => prev.map((j, i) => (i === idx ? { ...j, status: 'uploading' } : j)));
    try {
      const resp = await window.FOLIO_API.uploadDocument(file, notebookId || 'default');
      setJobs((prev) => prev.map((j, i) => (i === idx ? { ...j, status: 'processing' } : j)));
      let job = null;
      let finished = false;
      for (let k = 0; k < 600; k++) {  // up to ~10 min
        job = await window.FOLIO_API.getJob(resp.job_id);
        if (job.status === 'completed' || job.status === 'failed') { finished = true; break; }
        await new Promise((r) => setTimeout(r, 1000));
      }
      if (job && job.status === 'completed') {
        const doneStatus = job.stage === 'fts_only' ? 'keyword_only'
          : job.stage === 'indexed_partial' ? 'partial'
          : 'done';
        setJobs((prev) => prev.map((j, i) => (i === idx ? { ...j, status: doneStatus } : j)));
        onComplete?.({ _async: true, document_id: job.document_id, filename: resp.filename }, tid);
        return doneStatus;
      }
      if (!finished) {
        // Still running after ~10 min — report background, not failure; the
        // backend may complete later and a refresh surfaces the real source.
        setJobs((prev) => prev.map((j, i) => (i === idx ? { ...j, status: 'background' } : j)));
        onComplete?.({ _async: true, document_id: job?.document_id, filename: resp.filename, job_id: resp.job_id }, tid);
        return 'background';
      }
      throw new Error((job && job.error_message) || 'Ingestion failed');
    } catch (err) {
      setJobs((prev) => prev.map((j, i) => (i === idx ? { ...j, status: 'failed', error: err.message || 'Failed' } : j)));
      onComplete?.(null, tid, err.message);
      return 'failed';
    }
  };

  const startQueue = async () => {
    setPhase('ingesting');
    setJobs(files.map((f) => ({ name: f.name, status: 'queued', error: '' })));
    let done = 0;
    let failed = 0;
    let background = 0;
    let degraded = 0;
    for (let i = 0; i < files.length; i++) {
      const r = await processFile(files[i], i);
      if (r === 'done') done++;
      else if (r === 'failed') failed++;
      else if (r === 'background') background++;
      else degraded++; // 'keyword_only' | 'partial'
    }
    setSummary({ done, failed, background, degraded });
    setPhase('done');
  };

  const formatSummary = () => {
    const counts = {};
    files.forEach((f) => { const ext = extOf(f.name); counts[ext] = (counts[ext] || 0) + 1; });
    return Object.keys(counts).map((ext) => `${counts[ext]} ${FORMAT_LABELS[ext] || ext.slice(1).toUpperCase()}`).join(' · ');
  };

  const JOB_TONE = {
    queued: { color: 'var(--text-muted)', label: 'Queued' },
    uploading: { color: 'var(--warn)', label: 'Uploading' },
    processing: { color: 'var(--warn)', label: 'Processing' },
    background: { color: 'var(--text-muted)', label: 'Processing in background' },
    done: { color: 'var(--ok)', label: 'Indexed' },
    keyword_only: { color: 'var(--warn)', label: 'Keyword only' },
    partial: { color: 'var(--warn)', label: 'Indexed (partial)' },
    failed: { color: 'var(--warn)', label: 'Failed' },
  };

  return (
    <ModalShell open={open} onClose={close} closeable={true} title="Add sources">
      {phase === 'select' && (
        <div className="p-5">
          <input
            ref={fileRef}
            type="file"
            multiple
            accept=".pdf,.epub,.pptx,.docx,.md,.txt"
            className="hidden"
            onChange={(e) => { pickFiles(e.target.files); e.target.value = ''; }}
          />
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => { e.preventDefault(); setDragOver(false); pickFiles(e.dataTransfer.files); }}
            className={cx(
              'rounded-lg border border-dashed transition-all px-6 py-10 text-center',
              dragOver ? 'border-indigo-500 bg-indigo-50' : 'border-line bg-cream'
            )}
          >
            <div className="mx-auto w-12 h-12 rounded-full bg-card border border-line flex items-center justify-center mb-4">
              <Icon name="upload" size={20} className="text-indigo-500" />
            </div>
            <div className="serif-display text-[22px] text-ink">Drop files to begin</div>
            <div className="text-[13px] text-muted mt-1">PDF, EPUB, PPTX, DOCX, Markdown, TXT — one or more.</div>
            <Button variant="primary" size="md" className="mt-5" icon={<Icon name="file" size={14} />} onClick={() => fileRef.current?.click()}>
              Browse files
            </Button>
          </div>

          {files.length > 0 && (
            <div className="mt-5">
              <div className="flex items-center justify-between mb-2">
                <div className="text-[13px] font-medium text-ink">{files.length} file{files.length === 1 ? '' : 's'} selected</div>
                <div className="mono text-[10.5px] uppercase tracking-[0.14em] text-muted">{formatSummary()}</div>
              </div>
              <ul className="rounded-lg border border-line2 divide-y divide-line2 max-h-56 overflow-y-auto">
                {files.map((f, i) => (
                  <li key={i} className="flex items-center gap-3 px-3 py-2">
                    <span className="mono shrink-0 text-[10px] uppercase w-16 text-muted">{FORMAT_LABELS[extOf(f.name)] || extOf(f.name)}</span>
                    <span className="text-[13px] text-ink truncate flex-1">{f.name}</span>
                    <span className="mono text-[10.5px] text-muted shrink-0">{humanSize(f.size)}</span>
                    <button type="button" onClick={() => setFiles((p) => p.filter((_, j) => j !== i))} className="text-muted hover:text-ink p-1" aria-label={`Remove ${f.name}`}>
                      <Icon name="x" size={13} />
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {rejected.length > 0 && (
            <div className="mt-3 text-[12px]" style={{ color: 'var(--warn)' }}>
              Skipped {rejected.length} unsupported file{rejected.length === 1 ? '' : 's'}: {rejected.map((r) => r.name).join(', ')}
            </div>
          )}

          {files.length > 0 && (
            <div className="mt-5 flex justify-end">
              <Button variant="primary" size="md" iconRight={<Icon name="arrowRight" size={14} />} onClick={startQueue}>
                Ingest {files.length} file{files.length === 1 ? '' : 's'}
              </Button>
            </div>
          )}

          <div className="mt-5 mono text-[10.5px] uppercase tracking-[0.18em] text-muted">
            Private to you - processed in your workspace
          </div>
        </div>
      )}

      {phase === 'ingesting' && (
        <div className="p-5">
          <div className="flex items-center gap-3 px-3 py-3 rounded-lg bg-cream border border-line2 mb-4">
            <div className="w-8 h-8 rounded-full bg-card border border-line flex items-center justify-center shrink-0">
              <Icon name="file" size={15} />
            </div>
            <div className="min-w-0 flex-1">
              <div className="text-[13px] font-medium">Ingesting {jobs.filter((j) => j.status === 'done').length}/{jobs.length} files</div>
              <div className="mono text-[10.5px] text-muted">processing one at a time — you can close this and it keeps going</div>
            </div>
          </div>
          <ul className="rounded-lg border border-line2 divide-y divide-line2 max-h-72 overflow-y-auto">
            {jobs.map((j, i) => {
              const tone = JOB_TONE[j.status] || JOB_TONE.queued;
              return (
                <li key={i} className="flex items-center gap-3 px-3 py-2.5">
                  <span className="text-[13px] text-ink truncate flex-1">{j.name}</span>
                  {j.status === 'failed' && <span className="text-[11px] truncate max-w-[40%]" style={{ color: 'var(--warn)' }}>{j.error}</span>}
                  <span className="mono text-[10px] uppercase tracking-wide shrink-0" style={{ color: tone.color }}>{tone.label}</span>
                </li>
              );
            })}
          </ul>
          <div className="mt-4 text-right">
            <Button variant="ghost" size="md" onClick={close}>Run in background</Button>
          </div>
        </div>
      )}

      {phase === 'done' && (
        <div className="p-6 text-center">
          <div className="mx-auto w-12 h-12 rounded-full bg-ok/15 text-ok flex items-center justify-center mb-4">
            <Icon name="check" size={22} stroke={2.4} />
          </div>
          <div className="serif-display text-[24px] text-ink">
            {(() => {
              const parts = [];
              if (summary.done > 0) parts.push(`${summary.done} indexed`);
              if (summary.degraded > 0) parts.push(`${summary.degraded} with limitations`);
              if (summary.failed > 0) parts.push(`${summary.failed} failed`);
              if (summary.background > 0) parts.push(`${summary.background} still processing`);
              return parts.length ? parts.join(' · ') : 'No files processed';
            })()}
          </div>
          <div className="text-[13px] text-muted mt-1">Your sources appear in the sidebar as they finish indexing.</div>
          <div className="mt-5 flex items-center justify-center gap-2">
            <Button variant="primary" size="md" onClick={close}>Done</Button>
          </div>
        </div>
      )}

      {phase === 'error' && (
        <div className="p-6 text-center">
          <div className="mx-auto w-12 h-12 rounded-full bg-warn/15 text-warn flex items-center justify-center mb-4">
            <Icon name="x" size={22} stroke={2.4} />
          </div>
          <div className="serif-display text-[20px] text-ink">No supported files</div>
          <div className="text-[13px] text-muted mt-2 max-w-sm mx-auto">
            Supported: PDF, EPUB, PPTX, DOCX, Markdown, TXT.
            {rejected.length > 0 && <span style={{ color: 'var(--warn)' }}> Skipped: {rejected.map((r) => r.name).join(', ')}</span>}
          </div>
          <div className="mt-5 flex items-center justify-center gap-2">
            <Button variant="secondary" size="md" onClick={close}>Close</Button>
            <Button variant="primary" size="md" onClick={() => reset()}>Try again</Button>
          </div>
        </div>
      )}
    </ModalShell>
  );
};

// Shared modal building blocks

const ModalLabel = ({ children }) => (
  <div className="sans" style={{ fontSize: 'calc(10.5px * var(--folio-font-scale, 1))', color: 'var(--text-secondary)',
       letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 6 }}>
    {children}
  </div>
);

const SourcesChipList = ({ activeIds, idToName, onRemove, loading }) => {
  if (activeIds.length === 0) return (
    <div className="text-[12.5px]" style={{ color: 'var(--warn)' }}>No sources selected.</div>
  );
  return (
    <div className="flex flex-col gap-1.5">
      {activeIds.map(id => (
        <div key={id} className="flex items-center justify-between px-2 py-1.5 rounded"
             style={{ background: 'var(--surface-card)' }}>
          <span className="text-[13px] text-ink truncate flex-1" title={idToName[id]}>
            {idToName[id]}
          </span>
          <button type="button"
                  onClick={() => onRemove(id)}
                  disabled={loading || activeIds.length <= 1}
                  aria-label={`Remove ${idToName[id] || 'source'}`}
                  className="ml-2 inline-flex h-6 w-6 items-center justify-center rounded text-[12px]"
                  style={{ color: 'var(--text-secondary)',
                           opacity: activeIds.length <= 1 ? 0.3 : 0.6,
                           cursor: activeIds.length <= 1 ? 'not-allowed' : 'pointer' }}
                  title={activeIds.length <= 1 ? 'At least one source required' : 'Remove'}>
            <Icon name="x" size={13} />
          </button>
        </div>
      ))}
    </div>
  );
};

const ScopeInput = ({ value, onChange, disabled, placeholder }) => (
  <input
    type="text"
    value={value}
    onChange={e => onChange(e.target.value)}
    placeholder={placeholder || 'e.g. Chapter 3 - leave blank for whole doc'}
    disabled={disabled}
    className="w-full bg-transparent border-b py-2 outline-none serif-display"
    style={{ fontSize: 'calc(16px * var(--folio-font-scale, 1))', color: 'var(--text-primary)', borderColor: 'var(--border-medium)' }}
  />
);

const PillRow = ({ options, value, onChange, disabled }) => (
  <div className="flex flex-wrap gap-2">
    {options.map(opt => (
      <button key={opt} onClick={() => onChange(opt)} disabled={disabled}
              aria-pressed={value === opt}
              className={cx('px-3 py-1.5 rounded-full text-[11.5px] sans transition-colors',
                value === opt ? 'bg-[var(--accent)] text-[var(--surface-main)]'
                              : 'bg-[var(--surface-card)] text-[var(--text-secondary)] hover:bg-[var(--border-soft)]')}>
        {opt}
      </button>
    ))}
  </div>
);

const MODAL_DEFAULT_OPTIONS = {
  summaryLength: ['short', 'medium', 'detailed'],
  summaryStyle: ['prose', 'bullets', 'structured'],
  flashcardCount: [10, 20, 30, 50],
  quizCount: [5, 10, 15, 20, 25, 30],
  quizDifficulty: ['basic', 'medium', 'hard'],
};

const pickDefined = (...values) => values.find(v => v !== undefined && v !== null && v !== '');

const normalizeChoiceDefault = (value, fallback, options) => {
  const picked = pickDefined(value);
  return options.includes(picked) ? picked : fallback;
};

const normalizeCountDefault = (value, fallback) => {
  const picked = pickDefined(value);
  const n = typeof picked === 'number' ? picked : Number.parseInt(picked, 10);
  return Number.isFinite(n) && n > 0 ? Math.round(n) : fallback;
};

const numericOptionsWithDefault = (options, value) => (
  options.includes(value) ? options : [...options, value].sort((a, b) => a - b)
);

const resolveGenerateModalDefaults = (kind, defaults = {}) => {
  const d = defaults || {};
  if (kind === 'summary') {
    return {
      length: normalizeChoiceDefault(d.length, 'medium', MODAL_DEFAULT_OPTIONS.summaryLength),
      style: normalizeChoiceDefault(d.style, 'structured', MODAL_DEFAULT_OPTIONS.summaryStyle),
    };
  }
  if (kind === 'flashcards') {
    return { count: normalizeCountDefault(d.count, 20) };
  }
  if (kind === 'quiz') {
    return {
      quiz_count: normalizeCountDefault(d.quiz_count ?? d.count, 10),
      quiz_difficulty: normalizeChoiceDefault(d.quiz_difficulty ?? d.difficulty, 'medium', MODAL_DEFAULT_OPTIONS.quizDifficulty),
    };
  }
  return {};
};

const ModalActions = ({ onCancel, onSubmit, loading, count, disabled }) => (
  <div className="flex items-center justify-end gap-3 pt-3">
    <button onClick={onCancel} disabled={loading}
            className="sans text-[11.5px] uppercase tracking-wider"
            style={{ color: 'var(--text-muted)' }}>Cancel</button>
    <button onClick={onSubmit} disabled={loading || disabled}
            className="sans inline-flex items-center gap-1.5 text-[11.5px] uppercase tracking-wider px-4 py-2 rounded-full transition-colors"
            style={{ color: 'var(--surface-main)', background: loading ? 'var(--border-medium)' : 'var(--accent)' }}>
      {loading && <InlineSpinner size={12} color="var(--surface-main)" />}
      {loading ? 'Generating...' : count <= 1 ? 'Generate' : `Generate ${count}`}
    </button>
  </div>
);

// Shared hook: manages activeIds + scope + handleSubmit framework for any generate modal
const useGenerateModalState = ({ open, documentIds, sourceNames, kind, buildParams,
                                 onPendingStart, onGenerate, onFailed, onClose, onSubmitted,
                                 notebookId }) => {
  const nbId = notebookId || 'default';
  const [activeIds, setActiveIds] = React.useState([]);
  const [scope, setScope] = React.useState('');
  const [loading, setLoading] = React.useState(false);

  React.useEffect(() => {
    if (open) {
      setActiveIds([...(documentIds || [])]);
      setScope('');
    }
  }, [open, documentIds]);

  const idToName = React.useMemo(() => {
    const m = {};
    (documentIds || []).forEach((id, i) => { m[id] = (sourceNames || [])[i] || id; });
    return m;
  }, [documentIds, sourceNames]);

  const handleRemove = (id) => setActiveIds(prev => prev.filter(x => x !== id));

  const handleSubmit = async () => {
    if (activeIds.length === 0 || !window.FOLIO_API?.generate) return;
    setLoading(true);
    const trimmedScope = scope.trim();
    const params = buildParams({ scope: trimmedScope });

    // Pre-flight: verify document IDs still exist in the notebook before calling generate
    let validIds = new Set(activeIds);
    try {
      const docs = await window.FOLIO_API.listDocuments(nbId);
      const backendIds = new Set(docs.map(d => d.document_id));
      validIds = new Set(activeIds.filter(id => backendIds.has(id)));
    } catch (listErr) {
      // If listDocuments fails (e.g. backend down), proceed with what we have
      // and let the generate call surface any real errors.
      console.warn('Pre-flight document list failed, proceeding with active IDs:', listErr);
    }

    if (validIds.size === 0) {
      setLoading(false);
      alert('Selected documents are no longer available. The document list will be refreshed.');
      return;
    }

    const requests = Array.from(validIds).map(docId => ({
      docId,
      requestId: `${kind}_${docId}_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    }));

    if (onPendingStart) {
      requests.forEach(({ docId, requestId }) => onPendingStart({
        kind, scope: trimmedScope, documentId: docId, requestId,
        sourceName: idToName[docId] || docId, ...params,
      }));
    }
    // Close modal immediately so user sees Studio pending state; onSubmitted keeps
    // generatingKind alive, onClose would clear it for cancel scenarios.
    if (onSubmitted) { onSubmitted(); } else { onClose(); }

    requests.forEach(async ({ docId, requestId }) => {
      try {
        const result = await window.FOLIO_API.generate(kind, [docId],
          { scope: trimmedScope, ...params }, nbId);
        onGenerate && onGenerate(result, docId, requestId);
      } catch (err) {
        console.error(`${kind} gen failed for`, docId, err);
        onFailed && onFailed(docId, requestId);
        const msg = err?.message || 'Generation failed';
        const label = kind === 'flashcards' ? 'Flashcards'
          : kind === 'quiz' ? 'Quiz'
          : kind === 'mindmap' ? 'Mind map'
          : 'Summary';
        // Surface a cleaner message — strip UUIDs when the cause is a missing document.
        const clean = msg.includes('not found in notebook')
          ? 'The selected document is no longer available. Please refresh your source list.'
          : msg;
        alert(`${label} generation failed: ${clean}`);
      }
    });
    setLoading(false);
  };

  return { activeIds, scope, setScope, loading, idToName, handleRemove, handleSubmit };
};

// Summary Config Modal

const SummaryConfigModal = ({
  open, onClose, documentIds, sourceNames, onGenerate, onPendingStart, onFailed, onSubmitted,
  defaults, defaultLength, defaultStyle, defaultSummaryLength, defaultSummaryStyle,
  notebookId,
}) => {
  const summaryDefaults = resolveGenerateModalDefaults('summary', {
    length: pickDefined(defaultSummaryLength, defaultLength, defaults?.summary?.length, defaults?.length),
    style: pickDefined(defaultSummaryStyle, defaultStyle, defaults?.summary?.style, defaults?.style),
  });
  const [length, setLength] = React.useState(summaryDefaults.length);
  const [style, setStyle] = React.useState(summaryDefaults.style);

  const { activeIds, scope, setScope, loading, idToName, handleRemove, handleSubmit } =
    useGenerateModalState({
      open, documentIds, sourceNames, kind: 'summary',
      buildParams: () => ({ length, style }),
      onPendingStart, onGenerate, onFailed, onClose, onSubmitted,
      notebookId,
    });

  React.useEffect(() => {
    if (!open) return;
    setLength(summaryDefaults.length);
    setStyle(summaryDefaults.style);
  }, [open, summaryDefaults.length, summaryDefaults.style]);

  const reset = () => { setScope(''); setLength(summaryDefaults.length); setStyle(summaryDefaults.style); };

  return (
    <ModalShell open={open} onClose={() => { onClose(); setTimeout(reset, 300); }} title="New summary">
      <div className="p-5 space-y-4">
        <div>
          <ModalLabel>Sources ({activeIds.length})</ModalLabel>
          <SourcesChipList activeIds={activeIds} idToName={idToName} onRemove={handleRemove} loading={loading} />
          {activeIds.length > 1 && (
            <div className="sans mt-1.5" style={{ fontSize: 'calc(11px * var(--folio-font-scale, 1))', color: 'var(--text-secondary)' }}>
              One summary will be generated per source.
            </div>
          )}
        </div>
        <div>
          <ModalLabel>Scope</ModalLabel>
          <ScopeInput value={scope} onChange={setScope} disabled={loading}
            placeholder="e.g. Chapter 3, methodology, what the book says about X ...leave blank for whole doc" />
        </div>
        <div>
          <ModalLabel>Length</ModalLabel>
          <PillRow options={MODAL_DEFAULT_OPTIONS.summaryLength} value={length} onChange={setLength} disabled={loading} />
        </div>
        <div>
          <ModalLabel>Style</ModalLabel>
          <PillRow options={MODAL_DEFAULT_OPTIONS.summaryStyle} value={style} onChange={setStyle} disabled={loading} />
        </div>
        <ModalActions onCancel={() => { onClose(); setTimeout(reset, 300); }}
                      onSubmit={handleSubmit} loading={loading} count={activeIds.length}
                      disabled={activeIds.length === 0} />
      </div>
    </ModalShell>
  );
};

// Flashcards Config Modal

const FlashcardsConfigModal = ({
  open, onClose, documentIds, sourceNames, onGenerate, onPendingStart, onFailed, onSubmitted,
  defaults, defaultCount, defaultFlashcardCount,
  notebookId,
}) => {
  const flashcardDefaults = resolveGenerateModalDefaults('flashcards', {
    count: pickDefined(defaultFlashcardCount, defaultCount, defaults?.flashcards?.count, defaults?.count),
  });
  const [count, setCount] = React.useState(flashcardDefaults.count);

  const { activeIds, scope, setScope, loading, idToName, handleRemove, handleSubmit } =
    useGenerateModalState({
      open, documentIds, sourceNames, kind: 'flashcards',
      buildParams: () => ({ count }),
      onPendingStart, onGenerate, onFailed, onClose, onSubmitted,
      notebookId,
    });

  React.useEffect(() => {
    if (!open) return;
    setCount(flashcardDefaults.count);
  }, [open, flashcardDefaults.count]);

  const reset = () => { setScope(''); setCount(flashcardDefaults.count); };

  return (
    <ModalShell open={open} onClose={() => { onClose(); setTimeout(reset, 300); }} title="New flashcards">
      <div className="p-5 space-y-4">
        <div>
          <ModalLabel>Sources ({activeIds.length})</ModalLabel>
          <SourcesChipList activeIds={activeIds} idToName={idToName} onRemove={handleRemove} loading={loading} />
          {activeIds.length > 1 && (
            <div className="sans mt-1.5" style={{ fontSize: 'calc(11px * var(--folio-font-scale, 1))', color: 'var(--text-secondary)' }}>
              One deck will be generated per source.
            </div>
          )}
        </div>
        <div>
          <ModalLabel>Scope</ModalLabel>
          <ScopeInput value={scope} onChange={setScope} disabled={loading}
            placeholder="e.g. Chapter 3 ...leave blank for whole doc" />
        </div>
        <div>
          <ModalLabel>Number of cards</ModalLabel>
          <PillRow options={numericOptionsWithDefault(MODAL_DEFAULT_OPTIONS.flashcardCount, flashcardDefaults.count)} value={count} onChange={setCount} disabled={loading} />
        </div>
        <ModalActions onCancel={() => { onClose(); setTimeout(reset, 300); }}
                      onSubmit={handleSubmit} loading={loading} count={activeIds.length}
                      disabled={activeIds.length === 0} />
      </div>
    </ModalShell>
  );
};

// Quiz Config Modal

const QuizConfigModal = ({
  open, onClose, documentIds, sourceNames, onGenerate, onPendingStart, onFailed, onSubmitted,
  defaults, defaultCount, defaultDifficulty, defaultQuizCount, defaultQuizDifficulty,
  notebookId,
}) => {
  const quizDefaults = resolveGenerateModalDefaults('quiz', {
    quiz_count: pickDefined(defaultQuizCount, defaultCount, defaults?.quiz?.quiz_count, defaults?.quiz?.count, defaults?.quiz_count, defaults?.count),
    quiz_difficulty: pickDefined(defaultQuizDifficulty, defaultDifficulty, defaults?.quiz?.quiz_difficulty, defaults?.quiz?.difficulty, defaults?.quiz_difficulty, defaults?.difficulty),
  });
  const [quizCount, setQuizCount] = React.useState(quizDefaults.quiz_count);
  const [quizDifficulty, setQuizDifficulty] = React.useState(quizDefaults.quiz_difficulty);
  const [quizRequirements, setQuizRequirements] = React.useState('');

  const { activeIds, scope, setScope, loading, idToName, handleRemove, handleSubmit } =
    useGenerateModalState({
      open, documentIds, sourceNames, kind: 'quiz',
      buildParams: () => ({ quiz_count: quizCount, quiz_difficulty: quizDifficulty, quiz_requirements: quizRequirements }),
      onPendingStart, onGenerate, onFailed, onClose, onSubmitted,
      notebookId,
    });

  React.useEffect(() => {
    if (!open) return;
    setQuizCount(quizDefaults.quiz_count);
    setQuizDifficulty(quizDefaults.quiz_difficulty);
    setQuizRequirements('');
  }, [open, quizDefaults.quiz_count, quizDefaults.quiz_difficulty]);

  const reset = () => {
    setScope('');
    setQuizCount(quizDefaults.quiz_count);
    setQuizDifficulty(quizDefaults.quiz_difficulty);
    setQuizRequirements('');
  };

  return (
    <ModalShell open={open} onClose={() => { onClose(); setTimeout(reset, 300); }} title="New quiz">
      <div className="p-5 space-y-4">
        <div>
          <ModalLabel>Sources ({activeIds.length})</ModalLabel>
          <SourcesChipList activeIds={activeIds} idToName={idToName} onRemove={handleRemove} loading={loading} />
          {activeIds.length > 1 && (
            <div className="sans mt-1.5" style={{ fontSize: 'calc(11px * var(--folio-font-scale, 1))', color: 'var(--text-secondary)' }}>
              One quiz will be generated per source.
            </div>
          )}
        </div>
        <div>
          <ModalLabel>Scope</ModalLabel>
          <ScopeInput value={scope} onChange={setScope} disabled={loading}
            placeholder="e.g. Chapter 3 ...leave blank for whole doc" />
        </div>
        <div>
          <ModalLabel>Number of questions</ModalLabel>
          <PillRow options={numericOptionsWithDefault(MODAL_DEFAULT_OPTIONS.quizCount, quizDefaults.quiz_count)} value={quizCount} onChange={setQuizCount} disabled={loading} />
        </div>
        <div>
          <ModalLabel>Difficulty</ModalLabel>
          <PillRow options={MODAL_DEFAULT_OPTIONS.quizDifficulty} value={quizDifficulty} onChange={setQuizDifficulty} disabled={loading} />
        </div>
        <div>
          <ModalLabel>Special Instructions (optional)</ModalLabel>
          <textarea
            value={quizRequirements}
            onChange={e => setQuizRequirements(e.target.value)}
            placeholder="e.g. Focus on experimental methods, include questions about key dates, emphasize formulas..."
            disabled={loading}
            rows={3}
            className="w-full bg-transparent border rounded-lg px-3 py-2 outline-none serif-display resize-y"
            style={{ fontSize: 'calc(14px * var(--folio-font-scale, 1))', borderColor: 'var(--border-medium)', color: 'var(--text-primary)', minHeight: 54 }}
          />
        </div>
        <ModalActions onCancel={() => { onClose(); setTimeout(reset, 300); }}
                      onSubmit={handleSubmit} loading={loading} count={activeIds.length}
                      disabled={activeIds.length === 0} />
      </div>
    </ModalShell>
  );
};

// Mindmap Config Modal

const MindmapConfigModal = ({ open, onClose, documentIds, sourceNames, onGenerate, onPendingStart, onFailed, onSubmitted, notebookId }) => {
  // Mindmap is single-source only ...always uses just the first selected document.
  const singleId = (documentIds || [])[0] || '';
  const [scope, setScope] = React.useState('');
  const [loading, setLoading] = React.useState(false);

  const idToName = React.useMemo(() => {
    const m = {};
    (documentIds || []).forEach((id, i) => { m[id] = (sourceNames || [])[i] || id; });
    return m;
  }, [documentIds, sourceNames]);

  const reset = () => { setScope(''); };

  const handleSubmit = async () => {
    if (!singleId || !window.FOLIO_API?.generate) return;
    setLoading(true);
    const trimmedScope = scope.trim();
    const requestId = `mindmap_${singleId}_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

    if (onPendingStart) {
      onPendingStart({
        kind: 'mindmap', scope: trimmedScope, documentId: singleId, requestId,
        sourceName: idToName[singleId] || singleId,
        sourceCount: 1,
      });
    }
    if (onSubmitted) onSubmitted();
    setLoading(false);

    try {
      const nbId = notebookId || 'default';
      const result = await window.FOLIO_API.generate('mindmap', [singleId], { scope: trimmedScope }, nbId);
      onGenerate && onGenerate(result, singleId, requestId);
    } catch (err) {
      console.error('Mindmap generation failed:', err);
      onFailed && onFailed(singleId, requestId);
      const msg = err?.message || 'Generation failed';
      const clean = (msg || '').includes('not found in notebook')
        ? 'The selected document is no longer available. Please refresh your source list.'
        : msg;
      alert('Mind map generation failed: ' + clean);
    }
  };

  return (
    <ModalShell open={open} onClose={() => { onClose(); setTimeout(reset, 300); }} title="New mind map">
      <div className="p-5 space-y-4">
        <div>
          <ModalLabel>Source</ModalLabel>
          <SourcesChipList activeIds={singleId ? [singleId] : []} idToName={idToName} onRemove={() => {}} loading={loading} />
          <div className="sans mt-1.5" style={{ fontSize: 'calc(11px * var(--folio-font-scale, 1))', color: 'var(--text-secondary)' }}>
            Mind maps are generated from a single source. To use a different source, select it in the sidebar.
          </div>
        </div>
        <div>
          <ModalLabel>Scope (optional)</ModalLabel>
          <ScopeInput value={scope} onChange={setScope} disabled={loading}
            placeholder="e.g. Chapter 3, methodology, key argument ...leave blank for whole doc" />
        </div>
        <ModalActions onCancel={() => { onClose(); setTimeout(reset, 300); }}
                      onSubmit={handleSubmit} loading={loading} count={1}
                      disabled={!singleId} />
      </div>
    </ModalShell>
  );
};

window.UploadModal = UploadModal;
window.SummaryConfigModal = SummaryConfigModal;
window.FlashcardsConfigModal = FlashcardsConfigModal;
window.QuizConfigModal = QuizConfigModal;
window.MindmapConfigModal = MindmapConfigModal;
