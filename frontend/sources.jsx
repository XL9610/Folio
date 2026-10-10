// Left sidebar ...Sources as a scope selector with explicit checkboxes.
// Checkbox and title clicks both toggle inclusion in the active scope.

const Checkbox = ({ checked, indeterminate, onClick, size = 14, disabled = false }) => (
  <button
    onClick={disabled ? undefined : onClick}
    disabled={disabled}
    className="shrink-0 inline-flex items-center justify-center transition-colors"
    style={{
      width: size,
      height: size,
      borderRadius: 3,
      border: `1px solid ${checked || indeterminate ? 'var(--accent)' : 'var(--border-medium)'}`,
      background: checked || indeterminate ? 'var(--accent)' : 'transparent',
      opacity: disabled ? 0.4 : 1,
      cursor: disabled ? 'not-allowed' : 'pointer',
    }}
    aria-checked={!!checked}
    aria-disabled={disabled}
    role="checkbox"
  >
    {checked && (
      <svg width={size - 4} height={size - 4} viewBox="0 0 10 10" fill="none">
        <path d="M2 5.2L4.2 7.4L8.5 2.5" stroke="var(--surface-main)" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    )}
    {indeterminate && !checked && (
      <span style={{ width: size - 6, height: 1.5, background: 'var(--surface-main)', borderRadius: 1 }} />
    )}
  </button>
);

const SOURCE_SORT_MODES = ['recent', 'title', 'status'];
const SOURCE_STATUS_ORDER = { processing: 0, failed: 1, ready: 2 };

const normalizeSourceSortMode = (mode) => (
  SOURCE_SORT_MODES.includes(mode) ? mode : 'recent'
);

const sourceTimestamp = (source) => {
  const raw = source?.created_at || source?.createdAt || source?.added_at || source?.addedAt || source?.addedRel || '';
  const parsed = Date.parse(raw);
  return Number.isFinite(parsed) ? parsed : 0;
};

const sourceTitle = (source) => String(source?.title || source?.originalFilename || '').trim().toLocaleLowerCase();

const sortSourcesForDisplay = (sources, sortMode = 'recent') => {
  const mode = normalizeSourceSortMode(sortMode);
  const rows = [...(sources || [])].map((source, index) => ({ source, index }));
  rows.sort((a, b) => {
    if (mode === 'title') {
      const byTitle = sourceTitle(a.source).localeCompare(sourceTitle(b.source));
      return byTitle || a.index - b.index;
    }
    if (mode === 'status') {
      const aRank = SOURCE_STATUS_ORDER[a.source?.status] ?? 99;
      const bRank = SOURCE_STATUS_ORDER[b.source?.status] ?? 99;
      if (aRank !== bRank) return aRank - bRank;
      const byTitle = sourceTitle(a.source).localeCompare(sourceTitle(b.source));
      return byTitle || a.index - b.index;
    }
    const aTime = sourceTimestamp(a.source);
    const bTime = sourceTimestamp(b.source);
    if (aTime !== bTime) return bTime - aTime;
    return a.index - b.index;
  });
  return rows.map(row => row.source);
};

const SourceItem = ({ source, included, onToggleInclude, onDelete }) => {
  const { title, authors, edition, pages, chunks, status, progress, progressLabel, originalFilename } = source;
  const isProcessing = status === 'processing';
  const isReady = status === 'ready';
  // EPUB / text sources have no physical pages — show chunk count instead
  const hasPages = Number.isFinite(pages) && pages > 0;

  return (
    <div
      className="group relative"
      style={{
        background: 'transparent',
        borderTop: '1px solid var(--border-soft)',
      }}
    >
      <div className="px-5 py-3.5">
        {/* Title row: checkbox + title text, delete button absolute on hover */}
        <div className="flex items-start gap-3">
          <div className="pt-0.5 shrink-0">
            <Checkbox checked={included} onClick={onToggleInclude} disabled={!isReady} />
          </div>

          <button
            onClick={onToggleInclude}
            className="flex-1 min-w-0 text-left block"
            style={{ opacity: included ? 1 : 0.5, transition: 'opacity 200ms ease' }}
            aria-pressed={included}
          >
            <div
              className="serif-display leading-snug"
              style={{
                fontSize: 'calc(16.5px * var(--folio-font-scale, 1))',
                fontWeight: 400,
                color: 'var(--text-primary)',
                letterSpacing: '-0.005em',
                overflowWrap: 'break-word',
                wordBreak: 'normal',
                hyphens: 'auto',
                lineHeight: 1.25,
                maxWidth: '100%',
                paddingRight: 32,
              }}
              title={originalFilename || title}
            >
              {title}
            </div>
            <div
              className="serif-display mt-0.5"
              style={{ fontSize: 'calc(12.5px * var(--folio-font-scale, 1))', color: 'var(--text-secondary)', fontStyle: 'italic', fontWeight: 400 }}
            >
              {authors}
            </div>
            <div className="mt-1 flex items-center gap-2 sans" style={{ fontSize: 'calc(11px * var(--folio-font-scale, 1))', color: 'var(--text-muted)', letterSpacing: '0.06em' }}>
              <span style={{ textTransform: 'uppercase', letterSpacing: '0.14em' }}>{edition}</span>
              <span style={{ color: 'var(--border-medium)' }}>-</span>
              <span>{hasPages ? `${pages.toLocaleString()} pp` : `${(chunks || 0).toLocaleString()} chunks`}</span>
              {status === 'failed' && (
                  <button
                      onClick={(e) => { e.stopPropagation(); onDelete?.(source.id); }}
                      style={{
                          fontSize: 'calc(10px * var(--folio-font-scale, 1))', color: 'var(--danger)', cursor: 'pointer',
                          textTransform: 'uppercase', letterSpacing: '0.06em',
                          marginLeft: 8, padding: '2px 6px', borderRadius: 4,
                          border: '1px solid var(--danger)',
                      }}
                      title="Remove this failed source"
                  >Remove</button>
              )}
            </div>

            {isProcessing && (
              <div className="mt-2">
                <div className="h-[1.5px] rounded-full" style={{ background: 'var(--border-soft)' }}>
                  <div
                    className="h-full rounded-full pulse-bar"
                    style={{ width: `${(progress || 0.4) * 100}%`, background: 'var(--accent)' }}
                  />
                </div>
                <div className="mt-1 sans inline-flex items-center gap-1.5" style={{ fontSize: 'calc(10px * var(--folio-font-scale, 1))', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                  <InlineSpinner size={10} />
                  {progressLabel || 'Processing...'}
                </div>
              </div>
            )}
          </button>
        </div>

        {/* Delete button ...positioned at top-right, visible on hover */}
        <div className="absolute right-3 top-3.5 opacity-0 group-hover:opacity-100 transition-opacity">
          <window.ConfirmDeleteButton onDelete={() => onDelete?.(source.id)} />
        </div>
      </div>
    </div>
  );
};

const SourcesSidebar = ({ sources, loading, error, includedIds, onToggleInclude, onSelectAll, onAdd, onUpload, onDeleteSource, notebookTitle, onNotebookTitleChange, onCollapse, onBack, onOpenSettings, onRetry, sortMode = 'recent' }) => {
  const displaySources = React.useMemo(() => sortSourcesForDisplay(sources, sortMode), [sources, sortMode]);
  const readySources = sources.filter(s => s.status === 'ready');
  const ready = readySources.length;
  const includedCount = includedIds.length;
  const allIncluded = ready > 0 && readySources.every(s => includedIds.includes(s.id));
  const someIncluded = !allIncluded && readySources.some(s => includedIds.includes(s.id));
  const [editingTitle, setEditingTitle] = React.useState(false);
  const [draftTitle, setDraftTitle] = React.useState('');

  const handleStartEdit = () => {
    setDraftTitle(notebookTitle || 'Study Companion');
    setEditingTitle(true);
  };
  const handleSave = () => {
    onNotebookTitleChange(draftTitle);
    setEditingTitle(false);
  };
  const handleCancel = () => { setEditingTitle(false); };
  const handleKeyDown = (e) => {
    if (e.key === 'Enter') handleSave();
    if (e.key === 'Escape') handleCancel();
  };

  return (
    <aside
      className="h-full flex flex-col"
      style={{ background: 'var(--surface-left)', borderRight: '1px solid var(--border-soft)' }}
    >
      {/* Notebook header */}
      <div className="px-5 pt-10 pb-6 group">
        {onBack && (
          <button
            onClick={onBack}
            className="sans mb-3 inline-flex items-center gap-1"
            style={{ fontSize: 'calc(12px * var(--folio-font-scale, 1))', color: 'var(--text-muted)', letterSpacing: '0.02em' }}
            onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--accent)'; }}
            onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-muted)'; }}
          >
            <span style={{ fontSize: 'calc(13px * var(--folio-font-scale, 1))', lineHeight: 1 }}>←</span> All notebooks
          </button>
        )}
        <div className="flex items-center justify-between">
          <div className="label">Notebook</div>
          <div className="flex items-center gap-1.5">
            {onOpenSettings && (
              <button
                onClick={onOpenSettings}
                className="opacity-0 group-hover:opacity-100 focus-visible:opacity-100 transition-opacity"
                style={{ color: 'var(--text-muted)', padding: 2, borderRadius: 4, display: 'inline-flex', alignItems: 'center' }}
                onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--text-primary)'; }}
                onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-muted)'; }}
                title="Settings"
                aria-label="Settings"
              >
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
              </button>
            )}
            {onCollapse && (
              <button
                onClick={onCollapse}
                className="opacity-0 group-hover:opacity-100 focus-visible:opacity-100 transition-opacity"
                style={{ color: 'var(--text-muted)', padding: 2, borderRadius: 4 }}
                onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--text-primary)'; }}
                onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-muted)'; }}
                title="Collapse sidebar"
                aria-label="Collapse sidebar"
              >
                <span style={{ fontSize: 'calc(14px * var(--folio-font-scale, 1))', lineHeight: 1 }}>{'<'}</span>
              </button>
            )}
          </div>
        </div>
        <div className="mt-1.5 flex items-center gap-2">
          {editingTitle ? (
            <>
              <input
                value={draftTitle}
                onChange={e => setDraftTitle(e.target.value)}
                onKeyDown={handleKeyDown}
                autoFocus
                className="serif-display flex-1 min-w-0"
                style={{
                  fontSize: 'calc(22px * var(--folio-font-scale, 1))', lineHeight: 1.15, color: 'var(--text-primary)',
                  letterSpacing: 0, fontWeight: 400,
                  background: 'var(--surface-card)', border: '1px solid var(--accent)',
                  borderRadius: 6, padding: '4px 8px', outline: 'none',
                }}
              />
              <button onClick={handleSave} style={{ color: 'var(--accent)', padding: 2 }}>
                <Icon name="check" size={18} />
              </button>
              <button onClick={handleCancel} style={{ color: 'var(--text-muted)', padding: 2 }}>
                <Icon name="x" size={16} />
              </button>
            </>
          ) : (
            <>
              <div className="serif-display" style={{ fontSize: 'calc(22px * var(--folio-font-scale, 1))', lineHeight: 1.15, color: 'var(--text-primary)', letterSpacing: 0, fontWeight: 400 }}>
                {notebookTitle}
              </div>
              <button onClick={handleStartEdit} className="opacity-0 group-hover:opacity-100 hover:opacity-100 transition-opacity"
                style={{ color: 'var(--text-muted)', padding: 2, marginLeft: -2 }}>
                <Icon name="pen" size={13} />
              </button>
            </>
          )}
        </div>
        <div className="serif-display mt-0.5" style={{ fontSize: 'calc(13px * var(--folio-font-scale, 1))', fontStyle: 'italic', color: 'var(--text-muted)' }}>
          Your personal study notebook
        </div>
      </div>

      {/* Sources in scope label */}
      <div className="px-5 pb-3 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <Checkbox
            checked={allIncluded}
            indeterminate={someIncluded}
            onClick={onSelectAll}
            size={13}
          />
          <span className="label" style={{ color: 'var(--text-secondary)' }}>
            Sources in scope - {includedCount} of {sources.length}
          </span>
        </div>
      </div>

      {/* List */}
      <div className="flex-1 overflow-y-auto scroll-thin">
        {loading ? (
          <div className="px-5 py-3 space-y-3">
            {[1,2,3,4].map(i => (
              <div key={i} className="flex items-start gap-3">
                <div className="shrink-0 mt-0.5 w-[14px] h-[14px] rounded-sm" style={{ background: 'var(--border-soft)', animation: 'skeletonPulse 1.6s ease-in-out infinite' }} />
                <div className="flex-1 space-y-2">
                  <div className="h-4 rounded w-3/4" style={{ background: 'var(--border-soft)', animation: 'skeletonPulse 1.6s ease-in-out infinite' }} />
                  <div className="h-3 rounded w-1/2" style={{ background: 'var(--border-soft)', animation: 'skeletonPulse 1.6s ease-in-out infinite', animationDelay: '0.15s' }} />
                </div>
              </div>
            ))}
          </div>
        ) : (
          <>
            {error && (
              <div className="px-5 py-3">
                <div className="p-3 rounded-lg" style={{ background: 'var(--surface-card)', border: '1px solid var(--warn)' }}>
                  <div className="sans" style={{ fontSize: 'calc(11px * var(--folio-font-scale, 1))', color: 'var(--warn)', marginBottom: 6 }}>Failed to load sources</div>
                  <div className="sans" style={{ fontSize: 'calc(10.5px * var(--folio-font-scale, 1))', color: 'var(--text-muted)', lineHeight: 1.4 }}>{error}</div>
                  {onRetry && (
                    <button type="button" onClick={onRetry} className="sans mt-2" style={{ fontSize: 'calc(11px * var(--folio-font-scale, 1))', color: 'var(--accent)', cursor: 'pointer' }}>Retry</button>
                  )}
                </div>
              </div>
            )}
            {displaySources.map(s => (
              <SourceItem
                key={s.id}
                source={s}
                included={includedIds.includes(s.id)}
                onToggleInclude={() => onToggleInclude(s.id)}
                onDelete={onDeleteSource}
              />
            ))}

            {/* Add ...quiet text link, not a button */}
            <button
              onClick={onAdd}
              className="serif-display block px-5 py-3 text-left w-full transition-colors"
              style={{ fontSize: 'calc(13.5px * var(--folio-font-scale, 1))', color: 'var(--text-muted)', fontStyle: 'italic', borderTop: '1px solid var(--border-soft)' }}
              onMouseEnter={(e) => e.currentTarget.style.color = 'var(--text-primary)'}
              onMouseLeave={(e) => e.currentTarget.style.color = 'var(--text-muted)'}
            >
              + add a source...
            </button>
          </>
        )}
      </div>

      {/* Footer */}
      <div className="px-5 pt-4 pb-6">
        <div className="hairline mb-3" />
        <div className="sans" style={{ fontSize: 'calc(10.5px * var(--folio-font-scale, 1))', color: 'var(--text-muted)' }}>
          {loading ? 'Loading...' : `${ready} of ${sources.length} ready`}
        </div>
      </div>
    </aside>
  );
};

window.SourcesSidebar = SourcesSidebar;
