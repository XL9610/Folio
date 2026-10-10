// Right panel ...Studio. Two distinct sections:
//   1. Generate cards (action-feeling, generative, inline progress state)
//   2. Saved items (quiet document list)
// Clicking a saved item EXPANDS this panel; the center chat stays visible.

const TypeKind = (k) => ({
  quiz: 'Quiz',
  flashcards: 'Flashcards',
  summary: 'Chapter summary',
  mindmap: 'Mind Map',
  note: 'Note',
}[k] || k);

// Per-kind visual styling. Summary inherits theme accent; others use fixed tones.
const KIND_VISUALS = {
  summary:    { kindVar: '--kind-summary' },
  flashcards: { kindVar: '--kind-flashcards' },
  quiz:       { kindVar: '--kind-quiz' },
  mindmap:    { kindVar: '--kind-mindmap' },
  note:       { kindVar: '--kind-flashcards' },
};
const kindVar = (k) => {
  const v = KIND_VISUALS[k] || KIND_VISUALS.summary;
  return `var(${v.kindVar})`;
};

const STUDIO_GENERATION_DEFAULTS = {
  summary: { length: 'medium', style: 'structured' },
  flashcards: { count: 20 },
  quiz: { quiz_count: 10, quiz_difficulty: 'medium' },
  mindmap: {},
};

const resolveStudioGenerationDefaults = (defaults = {}) => {
  const d = defaults || {};
  return {
    summary: { ...STUDIO_GENERATION_DEFAULTS.summary, ...(d.summary || {}) },
    flashcards: { ...STUDIO_GENERATION_DEFAULTS.flashcards, ...(d.flashcards || {}) },
    quiz: { ...STUDIO_GENERATION_DEFAULTS.quiz, ...(d.quiz || {}) },
    mindmap: { ...STUDIO_GENERATION_DEFAULTS.mindmap, ...(d.mindmap || {}) },
  };
};

// Cleans messy titles persisted from older versions, mirrors backend clean_filename.
// Safe to apply to already-clean titles (idempotent).
const cleanDisplayTitle = (title, kind) => {
  if (!title) return 'Untitled';
  let s = String(title);

  // Drop redundant kind prefix ("Flashcards: ..." when kind=flashcards)
  s = s.replace(/^(Flashcards|Quiz|Summary):\s*/i, '');

  // Strip piracy / library markers (mirror Python clean_filename)
  s = s.replace(/_z[_-]?library[_-]?sk[_,.\s]*/gi, '');
  s = s.replace(/_1lib[_-]?sk[_,.\s]*/gi, '');
  s = s.replace(/_?libgen[_-]?[a-z]*[_,.\s]*/gi, '');
  s = s.replace(/\(z[_-]?lib[^)]*\)/gi, '');
  s = s.replace(/\[z[_-]?lib[^\]]*\]/gi, '');

  // Strip file extension anywhere in string
  s = s.replace(/\.(pdf|epub|mobi|djvu|txt|docx?)(?=\s|[-_,.]|$)/gi, '');

  // Underscores ->spaces
  s = s.replace(/_/g, ' ');

  // Collapse leading / trailing punctuation runs
  s = s.replace(/^[,;\-\s]+/, '').replace(/[,;\-\s]+$/, '');

  // Collapse multiple spaces
  s = s.replace(/\s{2,}/g, ' ').trim();

  return s || 'Untitled';
};

const ScopeLabel = ({ count, danger }) => (
  <span
    className="sans"
    style={{
      fontSize: 'calc(10px * var(--folio-font-scale, 1))',
      color: danger ? 'var(--accent)' : 'var(--text-muted)',
      letterSpacing: '0.14em',
      textTransform: 'uppercase',
    }}
  >
    {count === 0
      ? 'Select sources to generate'
      : `Uses ${count} selected ${count === 1 ? 'source' : 'sources'}`}
  </span>
);

// Generate card
const GenerateCard = ({ kind, title, desc, includedCount, generating, generatingCount = 0, onClick, llmConfigured, onSetupApi }) => {
  const disabled = includedCount === 0 || generating;
  const needsSetup = llmConfigured === false; // explicitly false (not null=unknown)
  const [showTooltip, setShowTooltip] = React.useState(false);
  const kVar = kindVar(kind);
  const tooltipId = `generate-${kind}-tooltip`;
  const sourceLabel = `${includedCount} selected ${includedCount === 1 ? 'source' : 'sources'}`;
  const actionLabel = needsSetup
    ? `Set up API to generate ${title}`
    : includedCount === 0
      ? `Select sources to generate ${title}`
      : `Generate ${title} from ${sourceLabel}`;
  const statusLabel = generating
    ? 'Generating...'
    : needsSetup
      ? 'Setup needed'
      : includedCount === 0
        ? 'Needs source'
        : 'Ready';

  return (
    <div
      className="relative"
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
      style={{ zIndex: showTooltip ? 20 : 0 }}
    >
      <button
        onFocus={() => setShowTooltip(true)}
        onBlur={() => setShowTooltip(false)}
        onClick={disabled ? undefined : (needsSetup && onSetupApi ? onSetupApi : onClick)}
        disabled={disabled}
        aria-label={actionLabel}
        aria-describedby={showTooltip ? tooltipId : undefined}
        title={desc}
        className="text-left rounded-lg px-3 py-3 transition-all w-full relative"
        style={{
          background: showTooltip && !disabled
            ? `color-mix(in srgb, ${kVar} 8%, var(--surface-card))`
            : 'var(--surface-card)',
          border: '1px solid var(--border-soft)',
          borderLeft: `3px solid ${kVar}`,
          cursor: disabled ? (generating ? 'wait' : 'not-allowed') : 'pointer',
          opacity: disabled && !generating ? 0.55 : 1,
        }}
      >
        <div className="flex items-start gap-2.5">
          <span aria-hidden="true"
                style={{ display: 'inline-flex', width: 20, height: 20, minWidth: 20, color: kVar }}>
            <Icon name={kind} size={20} />
          </span>
          <div style={{ minWidth: 0 }}>
            <div
              className="serif-display"
              style={{
                fontSize: 'calc(14px * var(--folio-font-scale, 1))',
                fontWeight: 400,
                color: 'var(--text-primary)',
                lineHeight: 1.15,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              {title}
            </div>
            <div
              className="sans mt-1 inline-flex items-center gap-1.5"
              style={{
                fontSize: 'calc(9.5px * var(--folio-font-scale, 1))',
                color: needsSetup ? '#D4850A' : 'var(--text-muted)',
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                lineHeight: 1.2,
              }}
            >
              {generating && <InlineSpinner size={10} color={kVar} />}
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{statusLabel}</span>
            </div>
          </div>
        </div>
      </button>
      {showTooltip && (
        <div
          id={tooltipId}
          role="tooltip"
          className="sans absolute left-0 right-0 top-full mt-1.5 rounded-md px-3 py-2 transition-all"
          style={{
            background: 'var(--surface-popover, var(--surface-card-hover))',
            border: '1px solid var(--border-medium)',
            boxShadow: '0 12px 26px rgba(0, 0, 0, 0.16)',
            color: 'var(--text-secondary)',
            fontSize: 'calc(11px * var(--folio-font-scale, 1))',
            lineHeight: 1.35,
            pointerEvents: 'none',
          }}
        >
          {desc}
        </div>
      )}
    </div>
  );
};

// Saved item row: quiet document list, not an action card.
const SavedRow = ({ item, justAdded, isOpen, onOpen, onDelete }) => (
  <div
    className="group relative transition-colors"
    style={{
      background: justAdded
        ? 'color-mix(in srgb, var(--accent) 12%, var(--surface-card))'
        : isOpen
          ? 'var(--surface-selected)'
          : 'transparent',
      borderBottom: '1px solid var(--border-soft)',
      opacity: item._pending ? 0.7 : 1,
    }}
  >
    {isOpen && (
      <span
        className="absolute left-0 top-0 bottom-0 w-[2px]"
        style={{ background: 'var(--accent)' }}
      />
    )}
    <button
      onClick={item._pending ? undefined : onOpen}
      className="block w-full text-left px-5 py-3 transition-colors"
      style={{ cursor: item._pending ? 'default' : undefined }}
      onMouseEnter={(e) => { if (!isOpen && !justAdded && !item._pending) e.currentTarget.parentElement.style.background = 'color-mix(in srgb, var(--accent) 8%, var(--surface-card))'; }}
      onMouseLeave={(e) => { if (!isOpen && !justAdded && !item._pending) e.currentTarget.parentElement.style.background = 'transparent'; }}
    >
      <div
        className="serif-display flex items-baseline gap-2"
        style={{ fontSize: 'calc(14px * var(--folio-font-scale, 1))', color: 'var(--text-primary)', fontWeight: 400, lineHeight: 1.35 }}
      >
        <span
          aria-hidden="true"
          style={{
            display: 'inline-flex',
            width: 18,
            height: 18,
            minWidth: 18,
            color: kindVar(item.kind),
            transform: 'translateY(3px)',
            flexShrink: 0,
          }}
        >
          <Icon name={item.kind} size={18} />
        </span>
        <span className="flex-1">{cleanDisplayTitle(item.title, item.kind)}</span>
        {item._pending && (
          <span className="inline-flex items-center ml-2" style={{ verticalAlign: 'middle' }}>
            <InlineSpinner size={12} />
          </span>
        )}
      </div>
      <div
        className="sans mt-1 flex items-center gap-1.5 flex-wrap"
        style={{ fontSize: 'calc(10.5px * var(--folio-font-scale, 1))', color: 'var(--text-secondary)', letterSpacing: '0.04em' }}
      >
        <span style={{
          color: kindVar(item.kind),
          textTransform: 'uppercase',
          letterSpacing: '0.16em',
          fontSize: 'calc(9.5px * var(--folio-font-scale, 1))',
          fontWeight: 400,
        }}>
          {TypeKind(item.kind)}
        </span>
        <span style={{ color: 'var(--border-medium)' }}>-</span>
        <span>
          {item.kind === 'note'
            ? 'saved turn'
            : `${item.scopeCount} selected ${item.scopeCount === 1 ? 'source' : 'sources'}`}
        </span>
        <span style={{ color: 'var(--border-medium)' }}>-</span>
        <span>{item.when}</span>
      </div>
    </button>
    {/* Delete button ...positioned top-right, visible on group hover */}
    {onDelete && !item._pending && (
      <div className="absolute right-3 top-3 opacity-0 group-hover:opacity-100 transition-opacity">
        <window.ConfirmDeleteButton label="Del" onDelete={() => onDelete(item?._artifact?.artifact_id || item?._artifact?.id || item.id)} />
      </div>
    )}
  </div>
);

// List view: default narrow Studio.
const ReviewTypeLabel = (item) => (
  item.item_type === 'quiz_question' ? 'Quiz question' : 'Flashcard'
);

const ReviewItemRow = ({ item, onOpen, onDelete }) => (
  <div className="group relative transition-colors" style={{ borderBottom: '1px solid var(--border-soft)' }}>
    <button
      onClick={onOpen}
      className="block w-full text-left px-5 py-3 transition-colors"
      onMouseEnter={(e) => { e.currentTarget.parentElement.style.background = 'color-mix(in srgb, var(--accent) 8%, var(--surface-card))'; }}
      onMouseLeave={(e) => { e.currentTarget.parentElement.style.background = 'transparent'; }}
    >
      <div className="serif-display flex items-baseline gap-2" style={{ fontSize: 'calc(14px * var(--folio-font-scale, 1))', color: 'var(--text-primary)', lineHeight: 1.35 }}>
        <span aria-hidden="true" style={{ display: 'inline-flex', width: 18, height: 18, minWidth: 18, color: item.is_mistake ? 'var(--warn)' : kindVar(item.item_type === 'quiz_question' ? 'quiz' : 'flashcards'), transform: 'translateY(3px)', flexShrink: 0 }}>
          <Icon name={item.item_type === 'quiz_question' ? 'target' : 'cards'} size={18} />
        </span>
        <span className="flex-1">{item.prompt || 'Untitled review item'}</span>
        {item.is_starred && (
          <span aria-hidden="true" style={{ color: '#D4850A', transform: 'translateY(3px)' }}>
            <Icon name="star" size={14} />
          </span>
        )}
      </div>
      <div className="sans mt-1 flex items-center gap-1.5 flex-wrap" style={{ fontSize: 'calc(10.5px * var(--folio-font-scale, 1))', color: 'var(--text-secondary)', letterSpacing: '0.04em' }}>
        <span style={{ color: item.is_mistake ? 'var(--warn)' : 'var(--accent)', textTransform: 'uppercase', letterSpacing: '0.16em', fontSize: 'calc(9.5px * var(--folio-font-scale, 1))' }}>
          {item.is_mistake ? 'Mistake' : item.is_starred ? 'Starred' : 'Saved'}
        </span>
        <span style={{ color: 'var(--border-medium)' }}>-</span>
        <span>{ReviewTypeLabel(item)}</span>
        {item.page_start > 0 && (
          <>
            <span style={{ color: 'var(--border-medium)' }}>-</span>
            <span>p. {item.page_start}</span>
          </>
        )}
      </div>
    </button>
    {onDelete && (
      <div className="absolute right-3 top-3 opacity-0 group-hover:opacity-100 transition-opacity">
        <window.ConfirmDeleteButton label="Del" onDelete={() => onDelete(item.id)} />
      </div>
    )}
  </div>
);

const ReviewFilterButton = ({ active, children, onClick }) => (
  <button
    onClick={onClick}
    className="sans rounded-full px-2.5 py-1 transition-colors"
    style={{
      fontSize: 'calc(10.5px * var(--folio-font-scale, 1))',
      color: active ? 'var(--surface-main)' : 'var(--text-secondary)',
      background: active ? 'var(--accent)' : 'var(--surface-card)',
      border: '1px solid var(--border-soft)',
      letterSpacing: '0.04em',
    }}
  >
    {children}
  </button>
);

const studioSectionStyle = {
  background: 'color-mix(in srgb, var(--surface-card) 88%, var(--surface-main))',
  border: '1px solid var(--border-soft)',
  boxShadow: '0 1px 0 rgba(0, 0, 0, 0.02)',
};

const StudioDropdownSection = ({ title, count, defaultOpen = false, children, rightContent = null }) => {
  const [open, setOpen] = React.useState(defaultOpen);
  const contentId = React.useId ? React.useId() : `${title.toLowerCase().replace(/\s+/g, '-')}-section`;
  const itemLabel = `${count} ${count === 1 ? 'item' : 'items'}`;

  return (
    <section className="mx-4 mt-4 rounded-xl overflow-hidden" style={studioSectionStyle}>
      <button
        type="button"
        onClick={() => setOpen(v => !v)}
        aria-expanded={open}
        aria-controls={contentId}
        aria-label={`${title}, ${itemLabel}`}
        className="w-full px-4 py-3 flex items-center justify-between gap-3 text-left transition-colors"
        style={{ color: 'var(--text-secondary)' }}
      >
        <span className="inline-flex items-center gap-2" style={{ minWidth: 0 }}>
          <span aria-hidden="true" style={{ color: 'var(--text-muted)', flexShrink: 0 }}>
            <Icon name={open ? 'chevronDown' : 'chevronRight'} size={14} />
          </span>
          <span className="label" style={{ color: 'var(--text-secondary)' }}>{title} - {count}</span>
        </span>
        {rightContent}
      </button>
      {open && (
        <div id={contentId} style={{ borderTop: '1px solid var(--border-soft)' }}>
          {children}
        </div>
      )}
    </section>
  );
};

const StudioListView = ({
  includedCount, savedItems, reviewItems = [], generatingCounts = {}, justAddedId,
  onGenerate, onOpenSaved, onDeleteSaved, onOpenReviewItem, onDeleteReviewItem,
  llmConfigured, onCollapse, generationDefaults, onSaveNote,
}) => {
  const [reviewFilter, setReviewFilter] = React.useState('all');
  const [noteOpen, setNoteOpen] = React.useState(false);
  const [noteText, setNoteText] = React.useState('');
  const [noteSaving, setNoteSaving] = React.useState(false);
  const resolvedGenerationDefaults = React.useMemo(
    () => resolveStudioGenerationDefaults(generationDefaults),
    [generationDefaults]
  );
  const filteredReviewItems = React.useMemo(() => {
    if (reviewFilter === 'mistakes') return reviewItems.filter(i => i.is_mistake);
    if (reviewFilter === 'starred') return reviewItems.filter(i => i.is_starred);
    return reviewItems;
  }, [reviewItems, reviewFilter]);
  const generateItems = [
    { kind: 'quiz', title: 'A quiz', desc: 'Multiple choice and short answer, with citations' },
    { kind: 'flashcards', title: 'Flashcards', desc: 'Term and definition, grouped by chapter' },
    { kind: 'mindmap', title: 'A mind map', desc: 'Hierarchical map of key concepts' },
    { kind: 'summary', title: 'A summary', desc: 'Whole document, a section, or a topic' },
  ];

  return (
    <>
      {/* Header ...slightly translucent overlay so it reads as panel chrome */}
      <div className="studio-header px-5 pt-10 pb-5">
        <div className="flex items-center justify-between">
          <div className="label">Studio</div>
          {onCollapse && (
            <button
              onClick={onCollapse}
              style={{
                color: 'var(--text-muted)', padding: 2, borderRadius: 4,
                opacity: 0.55, transition: 'opacity 150ms, color 150ms',
              }}
              onMouseEnter={(e) => { e.currentTarget.style.opacity = '1'; e.currentTarget.style.color = 'var(--text-primary)'; }}
              onMouseLeave={(e) => { e.currentTarget.style.opacity = '0.55'; e.currentTarget.style.color = 'var(--text-muted)'; }}
              title="Collapse studio"
              aria-label="Collapse studio panel"
            >
              <span style={{ fontSize: 'calc(14px * var(--folio-font-scale, 1))', lineHeight: 1 }}>{'>'}</span>
            </button>
          )}
        </div>
        <div
          className="serif-display mt-1.5"
          style={{ fontSize: 'calc(19px * var(--folio-font-scale, 1))', lineHeight: 1.2, color: 'var(--text-primary)', fontWeight: 400, letterSpacing: 0 }}
        >
          Create from selected sources
        </div>
        <div
          className="serif-display mt-1"
          style={{ fontSize: 'calc(13px * var(--folio-font-scale, 1))', color: 'var(--text-secondary)', fontStyle: 'italic' }}
        >
          {includedCount === 0
            ? 'Pick at least one source on the left.'
            : `Working from ${includedCount} of your sources.`}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto scroll-thin pb-8" style={{ minHeight: 0 }}>
        {/* Generate */}
        <section className="mx-4 mt-4 rounded-xl overflow-visible" style={studioSectionStyle}>
          <div className="px-4 py-3 flex items-baseline justify-between gap-3">
            <span className="label" style={{ color: 'var(--text-secondary)' }}>Generate</span>
            <ScopeLabel count={includedCount} danger={includedCount === 0} />
          </div>
          <div className="px-4 pb-4 grid grid-cols-2 gap-2">
            {generateItems.map(g => {
              const pendingCount = generatingCounts[g.kind] || 0;
              return (
                <GenerateCard
                  key={g.kind}
                  kind={g.kind}
                  title={g.title}
                  desc={g.desc}
                  includedCount={includedCount}
                  generating={pendingCount > 0}
                  generatingCount={pendingCount}
                  onClick={() => onGenerate(g.kind, resolvedGenerationDefaults[g.kind] || {})}
                  llmConfigured={llmConfigured}
                  onSetupApi={() => {
                    if (typeof window !== 'undefined' && window._openSettings) window._openSettings();
                  }}
                />
              );
            })}
          </div>
        </section>

        {/* History ...generated artifacts, visually quieter, no card chrome */}
        <StudioDropdownSection title="History" count={savedItems.length} defaultOpen={true}>
          <div>
            {savedItems.length === 0 && (
              <div
                className="serif-display px-5 py-5 text-center"
                style={{ fontSize: 'calc(13px * var(--folio-font-scale, 1))', color: 'var(--text-muted)', fontStyle: 'italic' }}
              >
                Nothing saved yet. Generate something to fill this drawer.
              </div>
            )}
            {savedItems.map(item => (
              <SavedRow
                key={item.id}
                item={item}
                justAdded={item.id === justAddedId}
                isOpen={false}
                onOpen={() => onOpenSaved(item.id)}
                onDelete={onDeleteSaved}
              />
            ))}
          </div>
        </StudioDropdownSection>

        {/* Saved - generated cards/questions saved for review */}
        <StudioDropdownSection title="Saved" count={reviewItems.length} defaultOpen={false}>
          <div className="px-5 pb-3 flex items-center gap-2">
            <ReviewFilterButton active={reviewFilter === 'all'} onClick={() => setReviewFilter('all')}>All</ReviewFilterButton>
            <ReviewFilterButton active={reviewFilter === 'mistakes'} onClick={() => setReviewFilter('mistakes')}>Mistakes</ReviewFilterButton>
            <ReviewFilterButton active={reviewFilter === 'starred'} onClick={() => setReviewFilter('starred')}>Starred</ReviewFilterButton>
          </div>
          <div>
            {filteredReviewItems.length === 0 && (
              <div className="serif-display px-5 py-5 text-center" style={{ fontSize: 'calc(13px * var(--folio-font-scale, 1))', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                {reviewItems.length === 0 ? 'Save cards or miss quiz questions to fill this basket.' : 'No items match this filter.'}
              </div>
            )}
            {filteredReviewItems.map(item => (
              <ReviewItemRow
                key={item.id}
                item={item}
                onOpen={() => onOpenReviewItem && onOpenReviewItem(item.id)}
                onDelete={onDeleteReviewItem}
              />
            ))}
          </div>
        </StudioDropdownSection>

        <section className="mx-4 mt-4 rounded-xl px-4 py-4" style={studioSectionStyle}>
          <div className="label mb-3" style={{ color: 'var(--text-secondary)' }}>Notes</div>
          {!noteOpen ? (
            <button
              type="button"
              className="serif-display text-left transition-colors w-full"
              style={{ fontSize: 'calc(14px * var(--folio-font-scale, 1))', color: 'var(--accent)', fontStyle: 'italic' }}
              onClick={() => setNoteOpen(true)}
            >
              + write a margin note...
            </button>
          ) : (
            <div className="space-y-2">
              <textarea
                autoFocus
                value={noteText}
                onChange={(e) => setNoteText(e.target.value)}
                placeholder="Write a note about what you're reading..."
                rows={3}
                className="w-full serif-display rounded-lg px-3 py-2 outline-none"
                style={{ fontSize: 'calc(14px * var(--folio-font-scale, 1))', color: 'var(--text-primary)', background: 'var(--surface-card)', border: '1px solid var(--border-soft)', resize: 'vertical' }}
              />
              <div className="flex items-center justify-end gap-2">
                <button type="button" onClick={() => { setNoteOpen(false); setNoteText(''); }}
                  className="sans text-[11px] uppercase tracking-wider" style={{ color: 'var(--text-muted)', padding: '4px 8px' }}>
                  Cancel
                </button>
                <button type="button" disabled={!noteText.trim() || noteSaving} onClick={async () => {
                  if (!noteText.trim() || noteSaving || !onSaveNote) return;
                  setNoteSaving(true);
                  try { await onSaveNote(noteText.trim()); setNoteText(''); setNoteOpen(false); }
                  finally { setNoteSaving(false); }
                }}
                  className="sans text-[11px] uppercase tracking-wider px-3 py-1.5 rounded-full"
                  style={{ color: 'var(--surface-main)', background: noteText.trim() && !noteSaving ? 'var(--accent)' : 'var(--border-medium)' }}>
                  {noteSaving ? 'Saving…' : 'Save note'}
                </button>
              </div>
            </div>
          )}
        </section>
      </div>
    </>
  );
};

// Saved-item content renderers.
const SavedQuizDoc = ({ item }) => {
  const artifact = item._artifact;
  const questions = artifact?.content_json?.questions || artifact?.questions || [];
  // Fallback to mock data if no artifact
  const displayQuestions = questions.length > 0
    ? questions
    : (item.body?.questions || []);

  return (
    <div className="space-y-7">
      {displayQuestions.slice(0, 3).map((q, i) => (
        <div key={q.id || i}>
          <div className="sans mb-1.5" style={{ fontSize: 'calc(10.5px * var(--folio-font-scale, 1))', color: 'var(--text-muted)', letterSpacing: '0.14em', textTransform: 'uppercase' }}>
            Question {i + 1} - {q.choices ? 'Multiple choice' : (q.type === 'short' ? 'Short answer' : 'Question')}
          </div>
          <div className="serif-display" style={{ fontSize: 'calc(16px * var(--folio-font-scale, 1))', color: 'var(--text-primary)', lineHeight: 1.45, fontWeight: 500 }}>
            {q.question || q.prompt}
          </div>
          {q.choices && (
            <ol className="mt-3 space-y-1.5">
              {q.choices.map((choice, j) => {
                const label = String.fromCharCode(97 + j);
                const choiceText = typeof choice === 'string' ? choice : (choice.text || choice);
                return (
                  <li key={j} className="flex items-baseline gap-3">
                    <span className="serif-display" style={{ fontSize: 'calc(13px * var(--folio-font-scale, 1))', color: choiceText === q.answer ? 'var(--ok)' : 'var(--accent)', minWidth: 16 }}>
                      {label}.
                    </span>
                    <span className="serif-display" style={{ fontSize: 'calc(14.5px * var(--folio-font-scale, 1))', color: 'var(--text-primary)', lineHeight: 1.5, opacity: 0.85 }}>
                      {choiceText}
                    </span>
                  </li>
                );
              })}
            </ol>
          )}
          {(q.section_path || q.page_start > 0 || q.citation) && (
            <div className="mt-3 sans" style={{ fontSize: 'calc(10.5px * var(--folio-font-scale, 1))', color: 'var(--text-muted)', letterSpacing: '0.04em', fontStyle: 'italic' }}>
              From {q.section_path || q.citation?.section || 'source'}
              {q.page_start > 0 ? ` - p. ${q.page_start}` : q.citation?.loc ? ` - ${q.citation.loc}` : ''}
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

const SavedSummaryDoc = ({ item }) => {
  const artifact = item._artifact;
  const content = artifact?.content_json || artifact || {};
  const markdown = content.markdown || '';
  const stats = content.stats || artifact?.stats || {};

  if (!markdown) {
    // Fallback for legacy fake items without real data
    return (
      <div className="space-y-4 serif-display" style={{ fontSize: 'calc(15px * var(--folio-font-scale, 1))', lineHeight: 1.75, color: 'var(--text-primary)', opacity: 0.92 }}>
        <p>This summary was generated before the artifact backend was available.</p>
      </div>
    );
  }

  const hasMarked = typeof window !== 'undefined' && window.marked;
  const hasPurify = typeof window !== 'undefined' && window.DOMPurify;
  if (hasMarked && hasPurify) {
    const raw = window.marked.parse(markdown, { gfm: true, breaks: false });
    const clean = window.DOMPurify.sanitize(raw);
    return (
      <div>
        <div
          className="markdown-body serif-display"
          style={{ fontSize: 'calc(15px * var(--folio-font-scale, 1))', lineHeight: 1.75, color: 'var(--text-primary)' }}
          dangerouslySetInnerHTML={{ __html: clean }}
        />
        {/* Stats footer */}
        {(stats.chunk_count != null || stats.scope_layer) && (
          <div className="mt-6 pt-4 border-t" style={{ borderColor: 'var(--border-soft)' }}>
            <div className="sans flex flex-wrap gap-3" style={{ fontSize: 'calc(10.5px * var(--folio-font-scale, 1))', color: 'var(--text-muted)', letterSpacing: '0.04em' }}>
              {stats.chunk_count != null && (
                <span>{stats.chunk_count} chunks</span>
              )}
              {stats.scope_layer && (
                <span>scope: {stats.scope_layer}</span>
              )}
              {stats.page_start > 0 && (
                <span>pp. {stats.page_start}-{stats.page_end}</span>
              )}
            </div>
          </div>
        )}
      </div>
    );
  }

  // Fallback: plain text
  return (
    <div className="serif-display whitespace-pre-wrap" style={{ fontSize: 'calc(15px * var(--folio-font-scale, 1))', lineHeight: 1.75, color: 'var(--text-primary)', opacity: 0.92 }}>
      {markdown}
    </div>
  );
};

const SavedNoteAnswer = ({ text }) => {
  if (!text) return null;
  const hasMarked = typeof window !== 'undefined' && window.marked;
  const hasPurify = typeof window !== 'undefined' && window.DOMPurify;
  if (hasMarked && hasPurify) {
    const raw = window.marked.parse(text, { gfm: true, breaks: false });
    const clean = window.DOMPurify.sanitize(raw);
    return (
      <div>
        <div className="sans mb-1" style={{ fontSize: 'calc(10px * var(--folio-font-scale, 1))', color: 'var(--text-muted)', letterSpacing: '0.16em', textTransform: 'uppercase' }}>Answer</div>
        <div
          className="markdown-body serif-display"
          style={{ fontSize: 'calc(15px * var(--folio-font-scale, 1))', lineHeight: 1.7, color: 'var(--text-primary)' }}
          dangerouslySetInnerHTML={{ __html: clean }}
        />
      </div>
    );
  }
  return (
    <div>
      <div className="sans mb-1" style={{ fontSize: 'calc(10px * var(--folio-font-scale, 1))', color: 'var(--text-muted)', letterSpacing: '0.16em', textTransform: 'uppercase' }}>Answer</div>
      <div className="serif-display whitespace-pre-wrap" style={{ fontSize: 'calc(15px * var(--folio-font-scale, 1))', lineHeight: 1.7, color: 'var(--text-primary)' }}>
        {text}
      </div>
    </div>
  );
};

// Detail view.
const SavedNoteDoc = ({ item }) => {
  const artifact = item._artifact || {};
  const content = artifact.content_json || artifact || {};
  const messages = Array.isArray(content.messages) ? content.messages : [];
  const quiz = content.quiz_context || null;
  const user = messages.find(m => m.role === 'user') || {};
  const assistant = [...messages].reverse().find(m => m.role === 'assistant') || {};
  const freeText = typeof content.text === 'string' ? content.text : '';

  return (
    <div className="space-y-5">
      {freeText && (
        <div className="serif-display whitespace-pre-wrap" style={{ fontSize: 'calc(15px * var(--folio-font-scale, 1))', lineHeight: 1.65, color: 'var(--text-primary)' }}>
          {freeText}
        </div>
      )}
      {quiz && quiz.question && (
        <div className="p-4" style={{ background: 'var(--surface-card)', border: '1px solid var(--border-soft)', borderRadius: 8 }}>
          <div className="sans mb-1" style={{ fontSize: 'calc(10px * var(--folio-font-scale, 1))', color: 'var(--accent)', letterSpacing: '0.16em', textTransform: 'uppercase' }}>
            Quiz context
          </div>
          <div className="serif-display" style={{ fontSize: 'calc(16px * var(--folio-font-scale, 1))', color: 'var(--text-primary)', lineHeight: 1.45 }}>
            {quiz.question}
          </div>
          {(quiz.selected_answer || quiz.correct_answer) && (
            <div className="sans mt-2" style={{ fontSize: 'calc(11px * var(--folio-font-scale, 1))', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              {quiz.selected_answer && <span>Your answer: {quiz.selected_answer}</span>}
              {quiz.selected_answer && quiz.correct_answer && <span> - </span>}
              {quiz.correct_answer && <span>Correct: {quiz.correct_answer}</span>}
              {typeof quiz.was_correct === 'boolean' && (
                <span style={{ color: quiz.was_correct ? 'var(--ok)' : 'var(--warn)' }}>
                  {' - '}{quiz.was_correct ? 'correct' : 'incorrect'}
                </span>
              )}
            </div>
          )}
        </div>
      )}

      {user.text && (
        <div>
          <div className="sans mb-1" style={{ fontSize: 'calc(10px * var(--folio-font-scale, 1))', color: 'var(--text-muted)', letterSpacing: '0.16em', textTransform: 'uppercase' }}>Question</div>
          <div className="serif-display whitespace-pre-wrap" style={{ fontSize: 'calc(15px * var(--folio-font-scale, 1))', lineHeight: 1.65, color: 'var(--text-primary)' }}>
            {user.text}
          </div>
        </div>
      )}
      {assistant.text && <SavedNoteAnswer text={assistant.text} />}
    </div>
  );
};

const ReviewItemDetailView = ({ item, onClose, onDelete }) => {
  const choices = Array.isArray(item.choices) ? item.choices : [];
  return (
    <>
      <div className="studio-header px-7 pt-7 pb-5 flex items-baseline justify-between gap-4">
        <div className="min-w-0">
          <div className="sans" style={{ fontSize: 'calc(10.5px * var(--folio-font-scale, 1))', color: 'var(--text-secondary)', letterSpacing: '0.14em', textTransform: 'uppercase' }}>
            Studio <span style={{ color: 'var(--border-medium)' }}>&gt;</span> Saved <span style={{ color: 'var(--border-medium)' }}>&gt;</span>{' '}
            <span style={{ color: item.is_mistake ? 'var(--warn)' : 'var(--accent)' }}>{item.is_mistake ? 'Mistake' : ReviewTypeLabel(item)}</span>
          </div>
          <h2 className="serif-display mt-2" style={{ fontSize: 'calc(28px * var(--folio-font-scale, 1))', color: 'var(--text-primary)', lineHeight: 1.15, fontWeight: 500 }}>
            {item.prompt}
          </h2>
          <div className="sans mt-1.5" style={{ fontSize: 'calc(11px * var(--folio-font-scale, 1))', color: 'var(--text-secondary)', letterSpacing: '0.04em' }}>
            {ReviewTypeLabel(item)}
            {item.page_start > 0 ? ` - p. ${item.page_start}` : ''}
          </div>
        </div>
        <div className="flex items-center gap-4 sans shrink-0" style={{ fontSize: 'calc(10.5px * var(--folio-font-scale, 1))', color: 'var(--text-secondary)', letterSpacing: '0.14em', textTransform: 'uppercase' }}>
          <button className="transition-colors" onClick={() => onDelete && onDelete(item.id)} onMouseEnter={(e) => e.currentTarget.style.color = 'var(--accent)'} onMouseLeave={(e) => e.currentTarget.style.color = 'var(--text-secondary)'}>remove</button>
          <button onClick={onClose} className="transition-colors" onMouseEnter={(e) => e.currentTarget.style.color = 'var(--text-primary)'} onMouseLeave={(e) => e.currentTarget.style.color = 'var(--text-secondary)'} style={{ paddingLeft: 12, borderLeft: '1px solid var(--border-soft)' }}>
            close x
          </button>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto scroll-thin" style={{ minHeight: 0 }}>
        <div className="px-7 py-8">
          <div className="mx-auto" style={{ maxWidth: 720, background: 'var(--surface-card-hover)', border: '1px solid var(--border-soft)', borderRadius: 10, padding: '32px 40px' }}>
            <div className="space-y-6">
              {choices.length > 0 && (
                <div>
                  <div className="sans mb-2" style={{ fontSize: 'calc(10px * var(--folio-font-scale, 1))', color: 'var(--text-muted)', letterSpacing: '0.16em', textTransform: 'uppercase' }}>Choices</div>
                  <ol className="space-y-1.5">
                    {choices.map((choice, i) => (
                      <li key={i} className="serif-display" style={{ fontSize: 'calc(14px * var(--folio-font-scale, 1))', color: 'var(--text-primary)', lineHeight: 1.5, opacity: 0.88 }}>
                        {choice}
                      </li>
                    ))}
                  </ol>
                </div>
              )}
              {item.last_user_answer && (
                <div>
                  <div className="sans mb-1" style={{ fontSize: 'calc(10px * var(--folio-font-scale, 1))', color: 'var(--text-muted)', letterSpacing: '0.16em', textTransform: 'uppercase' }}>Your answer</div>
                  <div className="serif-display" style={{ fontSize: 'calc(15px * var(--folio-font-scale, 1))', color: item.is_mistake ? 'var(--warn)' : 'var(--text-primary)', lineHeight: 1.55 }}>{item.last_user_answer}</div>
                </div>
              )}
              <div>
                <div className="sans mb-1" style={{ fontSize: 'calc(10px * var(--folio-font-scale, 1))', color: 'var(--text-muted)', letterSpacing: '0.16em', textTransform: 'uppercase' }}>
                  {item.item_type === 'flashcard' ? 'Definition' : 'Correct answer'}
                </div>
                <div className="serif-display" style={{ fontSize: 'calc(17px * var(--folio-font-scale, 1))', color: 'var(--text-primary)', lineHeight: 1.55 }}>{item.answer || 'No answer saved.'}</div>
              </div>
              {item.explanation && (
                <div>
                  <div className="sans mb-1" style={{ fontSize: 'calc(10px * var(--folio-font-scale, 1))', color: 'var(--text-muted)', letterSpacing: '0.16em', textTransform: 'uppercase' }}>Explanation</div>
                  <div className="serif-display" style={{ fontSize: 'calc(14.5px * var(--folio-font-scale, 1))', color: 'var(--text-secondary)', lineHeight: 1.65 }}>{item.explanation}</div>
                </div>
              )}
              {(item.section_path || item.page_start > 0) && (
                <div className="pt-4" style={{ borderTop: '1px solid var(--border-soft)' }}>
                  <div className="sans" style={{ fontSize: 'calc(10.5px * var(--folio-font-scale, 1))', color: 'var(--text-muted)', letterSpacing: '0.04em', fontStyle: 'italic' }}>
                    From {item.section_path || 'source'}{item.page_start > 0 ? ` - p. ${item.page_start}` : ''}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

const StudioDetailView = ({ item, onClose, onDelete, onAskFromMindmap, onQuizElaborate, onSaveReviewItem, reviewItems }) => {
  let body = null;
  const isMindmap = item.kind === 'mindmap' || item.kind === 'quiz';

  // Local inline renderers bypass window.X globals to avoid
  // "element type is invalid" crashes when panels.jsx loads late.
  // -- flashcard inline renderer --
  const renderFlashcards = (cards, title) => {
    if (!Array.isArray(cards) || cards.length === 0) {
      return <div className="serif-display text-[16px] text-muted italic">No cards in this deck.</div>;
    }
    return (
      <div className="space-y-4">
        {cards.slice(0, 20).map((c, i) => (
          <div key={i} className="p-4" style={{ background: 'var(--surface-card)', border: '1px solid var(--border-soft)', borderRadius: 8 }}>
            <div className="sans mb-1" style={{ fontSize: 'calc(10px * var(--folio-font-scale, 1))', color: 'var(--text-muted)', letterSpacing: '0.16em', textTransform: 'uppercase' }}>
              Card {i + 1}
            </div>
            <div className="serif-display" style={{ fontSize: 'calc(17px * var(--folio-font-scale, 1))', color: 'var(--text-primary)', fontWeight: 500 }}>
              {c.term || '(no term)'}
            </div>
            <div className="serif-display mt-1.5" style={{ fontSize: 'calc(13.5px * var(--folio-font-scale, 1))', color: 'var(--text-secondary)', lineHeight: 1.55 }}>
              {c.definition || ''}
            </div>
            {(c.section_path || c.page_start > 0) && (
              <div className="sans mt-2" style={{ fontSize: 'calc(10px * var(--folio-font-scale, 1))', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                {c.section_path}{c.page_start > 0 ? ` - p. ${c.page_start}` : ''}
              </div>
            )}
          </div>
        ))}
      </div>
    );
  };

  // -- mindmap inline renderer --
  const renderMindmap = (markdown, title) => {
    if (!markdown || typeof markdown !== 'string' || !markdown.trim()) {
      return <div className="serif-display text-[16px] text-muted italic">No mind map data available.</div>;
    }
    // Render as markdown using marked+purify (same as SavedSummaryDoc)
    if (typeof window !== 'undefined' && window.marked && window.DOMPurify) {
      try {
        const raw = window.marked.parse(markdown, { gfm: true, breaks: false });
        const clean = window.DOMPurify.sanitize(raw);
        return (
          <div className="markdown-body serif-display" style={{ fontSize: 'calc(15px * var(--folio-font-scale, 1))', lineHeight: 1.75, color: 'var(--text-primary)' }}
               dangerouslySetInnerHTML={{ __html: clean }} />
        );
      } catch (e) {
        console.warn('Markdown rendering failed:', e);
      }
    }
    // Ultimate fallback: plain text
    return (
      <pre className="serif-display whitespace-pre-wrap" style={{ fontSize: 'calc(14px * var(--folio-font-scale, 1))', lineHeight: 1.6, color: 'var(--text-primary)' }}>
        {String(markdown)}
      </pre>
    );
  };

  // Dispatch expanded artifact rendering by kind.

  if (item.kind === 'quiz') {
    if (typeof window !== 'undefined' && window.QuizViewer) {
      body = (
        <window.QuizViewer
          artifact={item._artifact}
          title={item.title}
          compact
          reviewItems={reviewItems}
          onSaveReviewItem={onSaveReviewItem}
          onElaborate={(question, answer) => onQuizElaborate && onQuizElaborate(question, answer, item)}
        />
      );
    } else {
      body = <SavedQuizDoc item={item} />;
    }
  }
  else if (item.kind === 'note') body = <SavedNoteDoc item={item} />;
  else if (item.kind === 'flashcards') {
    const artifact = item._artifact;
    const cards = artifact?.content_json?.cards || artifact?.cards || [];
    // Use full interactive viewer when available, fall back to inline list
    if (typeof window !== 'undefined' && window.FlashcardsViewer) {
      body = <window.FlashcardsViewer cards={cards} title={item.title} compact artifact={item._artifact} reviewItems={reviewItems} onSaveReviewItem={onSaveReviewItem} />;
    } else {
      body = renderFlashcards(cards, item.title);
    }
  }
  else if (item.kind === 'mindmap') {
    const artifact = item._artifact;
    const tree = artifact?.content_json?.tree || artifact?.tree || null;
    const markdown = artifact?.content_json?.markdown || artifact?.markdown || '';
    const artifactDocIds = artifact?.source_document_ids || [];
    if (typeof window !== 'undefined' && window.MindmapViewer) {
      body = (
        <window.MindmapViewer
          tree={tree}
          markdown={markdown}
          title={item.title}
          compact
          sourceDocumentIds={artifactDocIds}
          onAskNode={(node) => {
            const docs = Array.isArray(node?.source_document_ids) && node.source_document_ids.length
              ? node.source_document_ids
              : artifactDocIds;
            const prompt = window.buildMindmapPrompt
              ? window.buildMindmapPrompt(node, tree?.label || item.title, item.title)
              : `Discuss what these sources say about ${node?.label || 'this concept'}.`;
            onAskFromMindmap && onAskFromMindmap(prompt, docs);
          }}
        />
      );
    } else {
      body = renderMindmap(markdown, item.title);
    }
  }
  else body = <SavedSummaryDoc item={item} />;

  return (
    <>
      {/* Top utility bar */}
      <div
        className="studio-header px-7 pt-7 pb-5 flex items-baseline justify-between gap-4"
      >
        <div className="min-w-0">
          <div
            className="sans"
            style={{ fontSize: 'calc(10.5px * var(--folio-font-scale, 1))', color: 'var(--text-secondary)', letterSpacing: '0.14em', textTransform: 'uppercase' }}
          >
            Studio <span style={{ color: 'var(--border-medium)' }}>&gt;</span> History <span style={{ color: 'var(--border-medium)' }}>&gt;</span>{' '}
            <span style={{ color: 'var(--accent)' }}>{TypeKind(item.kind)}</span>
          </div>
          <h2
            className="serif-display mt-2"
            style={{ fontSize: 'calc(28px * var(--folio-font-scale, 1))', color: 'var(--text-primary)', lineHeight: 1.15, fontWeight: 500, letterSpacing: '-0.012em' }}
          >
            {item.title}
          </h2>
          <div className="sans mt-1.5" style={{ fontSize: 'calc(11px * var(--folio-font-scale, 1))', color: 'var(--text-secondary)', letterSpacing: '0.04em' }}>
            {TypeKind(item.kind)} - {item.scopeCount} selected {item.scopeCount === 1 ? 'source' : 'sources'} - {item.when}
          </div>
        </div>

        <div className="flex items-center gap-4 sans shrink-0" style={{ fontSize: 'calc(10.5px * var(--folio-font-scale, 1))', color: 'var(--text-secondary)', letterSpacing: '0.14em', textTransform: 'uppercase' }}>
          <button className="transition-colors" onClick={() => onDelete && onDelete(item.id)} onMouseEnter={(e) => e.currentTarget.style.color = 'var(--accent)'} onMouseLeave={(e) => e.currentTarget.style.color = 'var(--text-secondary)'}>delete</button>
          <button
            onClick={onClose}
            className="transition-colors"
            onMouseEnter={(e) => e.currentTarget.style.color = 'var(--text-primary)'}
            onMouseLeave={(e) => e.currentTarget.style.color = 'var(--text-secondary)'}
            style={{ paddingLeft: 12, borderLeft: '1px solid var(--border-soft)' }}
          >
            close x
          </button>
        </div>
      </div>

      {/* Content scroll */}
      <div className="flex-1 overflow-y-auto scroll-thin" style={{ minHeight: 0 }}>
        <div className="px-7 py-8">
          <div
            className="mx-auto"
            style={{
              maxWidth: isMindmap ? 'none' : 720,
              background: isMindmap ? 'transparent' : 'var(--surface-card-hover)',
              border: isMindmap ? 'none' : '1px solid var(--border-soft)',
              borderRadius: isMindmap ? 0 : 10,
              padding: isMindmap ? 0 : '36px 44px',
            }}
          >
            {body}
          </div>
        </div>
      </div>
    </>
  );
};

// Container
const StudioPanel = ({
  includedCount, savedItems, reviewItems = [], generatingCounts = {}, justAddedId,
  expandedItem, expandedReviewItem, onGenerate, onOpenSaved, onOpenReviewItem, onCloseExpanded,
  onDeleteArtifact, onDeleteReviewItem, onAskFromMindmap, onQuizElaborate, onSaveReviewItem,
  llmConfigured, onCollapse, generationDefaults, onSaveNote,
}) => {
  return (
    <aside
      className="h-full flex flex-col"
      style={{
        background: 'var(--surface-right)',
        borderLeft: '1px solid var(--border-medium)',
        minHeight: 0,
        overflow: 'hidden',
      }}
    >
      {expandedReviewItem ? (
        <ReviewItemDetailView
          item={expandedReviewItem}
          onClose={onCloseExpanded}
          onDelete={onDeleteReviewItem}
        />
      ) : expandedItem ? (
        <StudioDetailView
          item={expandedItem}
          onClose={onCloseExpanded}
          onDelete={onDeleteArtifact}
          onAskFromMindmap={onAskFromMindmap}
          onQuizElaborate={onQuizElaborate}
          onSaveReviewItem={onSaveReviewItem}
          reviewItems={reviewItems}
        />
      ) : (
        <StudioListView
          includedCount={includedCount}
          savedItems={savedItems}
          reviewItems={reviewItems}
          generatingCounts={generatingCounts}
          justAddedId={justAddedId}
          onGenerate={onGenerate}
          onOpenSaved={onOpenSaved}
          onOpenReviewItem={onOpenReviewItem}
          onDeleteSaved={onDeleteArtifact}
          onDeleteReviewItem={onDeleteReviewItem}
          llmConfigured={llmConfigured}
          onCollapse={onCollapse}
          generationDefaults={generationDefaults}
          onSaveNote={onSaveNote}
        />
      )}
    </aside>
  );
};

window.FOLIO_STUDIO_GENERATION_DEFAULTS = STUDIO_GENERATION_DEFAULTS;
window.resolveStudioGenerationDefaults = resolveStudioGenerationDefaults;
window.StudioPanel = StudioPanel;
