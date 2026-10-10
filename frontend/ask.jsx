// Center panel ...Chat presented as a scholarly reading surface.
// Document-centric: large serif title, footnote-style citations, generous rhythm.

const CITATION_REF_RE = /\[(S|W)(\d+)\]/g;
const CITATION_SKIP_TAGS = new Set(['A', 'CODE', 'PRE', 'SCRIPT', 'STYLE', 'TEXTAREA']);

const citationClassName = (prefix) => (
  `fn-num citation-ref ${prefix === 'W' ? 'citation-web' : 'citation-book'}`
);

const citationTitle = (prefix, number) => (
  `${prefix === 'W' ? 'Web source' : 'Book source'} ${prefix}${number}`
);

const shouldSkipCitationDecoration = (node, root) => {
  let el = node.parentElement;
  while (el && el !== root) {
    if (CITATION_SKIP_TAGS.has(el.tagName)) return true;
    el = el.parentElement;
  }
  return false;
};

const decorateCitationRefsInHtml = (html) => {
  if (!html || typeof window === 'undefined' || !window.DOMParser) return html;
  const doc = new window.DOMParser().parseFromString(`<div>${html}</div>`, 'text/html');
  const root = doc.body.firstElementChild;
  if (!root) return html;
  const walker = doc.createTreeWalker(root, window.NodeFilter?.SHOW_TEXT || 4);
  const nodes = [];
  let node = walker.nextNode();
  while (node) {
    if (!shouldSkipCitationDecoration(node, root) && CITATION_REF_RE.test(node.nodeValue || '')) {
      nodes.push(node);
    }
    CITATION_REF_RE.lastIndex = 0;
    node = walker.nextNode();
  }

  nodes.forEach(textNode => {
    const text = textNode.nodeValue || '';
    CITATION_REF_RE.lastIndex = 0;
    const fragment = doc.createDocumentFragment();
    let lastIndex = 0;
    let match;
    while ((match = CITATION_REF_RE.exec(text)) !== null) {
      if (match.index > lastIndex) {
        fragment.appendChild(doc.createTextNode(text.slice(lastIndex, match.index)));
      }
      const prefix = match[1];
      const number = match[2];
      const span = doc.createElement('span');
      span.className = citationClassName(prefix);
      span.title = citationTitle(prefix, number);
      span.dataset.refLabel = `${prefix}${number}`;
      span.textContent = `[${prefix}${number}]`;
      fragment.appendChild(span);
      lastIndex = match.index + match[0].length;
    }
    if (lastIndex < text.length) {
      fragment.appendChild(doc.createTextNode(text.slice(lastIndex)));
    }
    textNode.parentNode?.replaceChild(fragment, textNode);
  });

  return root.innerHTML;
};

/** Full Markdown renderer using CDN marked + DOMPurify.
 *  Falls back to MarkdownLite if CDN scripts are unavailable.
 */
const MarkdownRich = ({ text, citations = [] }) => {
  const [tip, setTip] = React.useState(null);
  if (!text) return null;

  const hasMarked = typeof window !== 'undefined' && window.marked;
  const hasPurify = typeof window !== 'undefined' && window.DOMPurify;

  const citationByLabel = React.useMemo(() => {
    const m = {};
    (citations || []).forEach((c, i) => {
      const label = c.ref_label || ((c.source_type || 'book') === 'web' ? `W${i + 1}` : `S${i + 1}`);
      m[label] = c;
    });
    return m;
  }, [citations]);

  if (!hasMarked || !hasPurify) {
    return <pre className="serif-display whitespace-pre-wrap" style={{ fontSize: 'calc(15px * var(--folio-font-scale, 1))', lineHeight: 1.7, color: 'var(--text-primary)' }}>{text}</pre>;
  }

  const raw = window.marked.parse(text, {
    gfm: true,
    breaks: false,
  });

  const clean = window.DOMPurify.sanitize(raw);
  const html = decorateCitationRefsInHtml(clean);

  const handleOver = (e) => {
    const el = e.target && e.target.closest ? e.target.closest('.citation-ref') : null;
    if (!el) { setTip(null); return; }
    const c = citationByLabel[el.dataset.refLabel];
    if (!c) { setTip(null); return; }
    const rect = el.getBoundingClientRect();
    setTip({ c, left: rect.left + rect.width / 2, top: rect.top });
  };

  const tooltip = tip && typeof window !== 'undefined' && window.ReactDOM && window.ReactDOM.createPortal
    ? window.ReactDOM.createPortal(
        <div
          style={{
            position: 'fixed',
            left: tip.left,
            top: tip.top - 10,
            transform: 'translate(-50%, -100%)',
            zIndex: 10000,
            background: 'var(--surface-card)',
            border: '1px solid var(--border-soft)',
            borderRadius: 8,
            padding: '8px 12px',
            maxWidth: 340,
            boxShadow: '0 10px 30px rgba(20,23,28,0.22)',
            pointerEvents: 'none',
            color: 'var(--text-primary)',
          }}
        >
          <div className="sans" style={{ fontSize: 11, letterSpacing: '0.04em', textTransform: 'uppercase', color: 'var(--text-muted)' }}>
            {(tip.c.source_type || 'book') === 'web' ? 'Internet' : 'Book'}
          </div>
          <div className="serif-display" style={{ fontSize: 13, marginTop: 3, fontWeight: 500 }}>{tip.c.src}</div>
          {tip.c.section && <div className="serif-display" style={{ fontSize: 12.5, fontStyle: 'italic', color: 'var(--text-secondary)', marginTop: 2 }}>{tip.c.section}</div>}
          {tip.c.loc && <div className="mono" style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>{tip.c.loc}</div>}
        </div>,
        document.body,
      )
    : null;

  return (
    <React.Fragment>
      <div
        className="markdown-body"
        dangerouslySetInnerHTML={{ __html: html }}
        onMouseOver={handleOver}
        onMouseLeave={() => setTip(null)}
      />
      {tooltip}
    </React.Fragment>
  );
};

const Citations = ({ citations }) => {
  const [expanded, setExpanded] = React.useState(false);
  const count = citations.length;

  return (
    <div className="mt-8">
      <div className="hairline mb-4" style={{ width: 80 }} />

      <button
        onClick={() => setExpanded(e => !e)}
        className="sans inline-flex items-center gap-1.5 hover:opacity-100 transition-opacity"
        style={{
          fontSize: 'calc(11.5px * var(--folio-font-scale, 1))',
          color: 'var(--text-secondary)',
          letterSpacing: '0.06em',
          textTransform: 'uppercase',
          opacity: 0.75,
          cursor: 'pointer',
        }}
        aria-expanded={expanded}
      >
        <span
          style={{
            display: 'inline-block',
            transition: 'transform 150ms',
            transform: expanded ? 'rotate(90deg)' : 'rotate(0deg)',
            fontSize: 'calc(9px * var(--folio-font-scale, 1))',
          }}
        >
          >
        </span>
        <span>
          {count} {count === 1 ? 'citation' : 'citations'}
        </span>
      </button>

      {expanded && (
        <ol className="space-y-2.5 mt-4 fade-in">
          {citations.map((c, i) => (
            <li key={i} className="flex gap-3 items-baseline">
              <span className="serif-display shrink-0" style={{ fontSize: 'calc(12px * var(--folio-font-scale, 1))', color: 'var(--accent)', minWidth: 18 }}>
                [{c.ref_label || ((c.source_type || 'book') === 'web' ? `W${i + 1}` : `S${i + 1}`)}]
              </span>
              <div className="serif-display" style={{ fontSize: 'calc(13.5px * var(--folio-font-scale, 1))', color: 'var(--text-secondary)', lineHeight: 1.55 }}>
                <span className="sans mr-2" style={{
                  fontSize: 'calc(9.5px * var(--folio-font-scale, 1))',
                  letterSpacing: '0.06em',
                  textTransform: 'uppercase',
                  color: (c.source_type || 'book') === 'web' ? '#3f7665' : 'var(--accent)',
                  border: '1px solid var(--border-soft)',
                  padding: '1px 5px',
                  borderRadius: 4,
                }}>{(c.source_type || 'book') === 'web' ? 'Internet' : 'Book'}</span>
                {(c.source_type || 'book') === 'web' && c.url ? (
                  <a
                    href={c.url}
                    target="_blank"
                    rel="noreferrer"
                    style={{ color: 'var(--text-primary)', textDecoration: 'underline', textUnderlineOffset: 2 }}
                  >
                    {c.src}
                  </a>
                ) : (
                  <span style={{ color: 'var(--text-primary)' }}>{c.src}</span>
                )}
                <span style={{ color: 'var(--text-muted)' }}> - </span>
                <span style={{ fontStyle: 'italic' }}>{c.section}</span>
                <span style={{ color: 'var(--text-muted)' }}> - </span>
                <span>{c.loc}</span>
                {(c.source_type || 'book') === 'web' && c.url && (
                  <div className="sans" style={{ fontSize: 'calc(11px * var(--folio-font-scale, 1))', color: 'var(--text-muted)', lineHeight: 1.4 }}>
                    {c.url}
                  </div>
                )}
              </div>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
};

const PlanTable = ({ plan }) => (
  <div className="mt-7">
    <div className="hairline mb-5" style={{ width: 80 }} />
    {plan.map((p, i) => (
      <div key={i} className="grid grid-cols-12 py-3" style={{
        borderTop: i === 0 ? 'none' : '1px solid var(--border-soft)',
      }}>
        <div className="col-span-2 serif-display" style={{ fontSize: 'calc(14px * var(--folio-font-scale, 1))', color: 'var(--text-primary)', fontWeight: 500 }}>{p.w}</div>
        <div className="col-span-7 serif-display" style={{ fontSize: 'calc(14.5px * var(--folio-font-scale, 1))', color: 'var(--text-primary)', lineHeight: 1.55 }}>{p.focus}</div>
        <div className="col-span-3 serif-display text-right" style={{ fontSize: 'calc(12.5px * var(--folio-font-scale, 1))', color: 'var(--text-muted)', fontStyle: 'italic' }}>
          {p.read} - {p.listen}
        </div>
      </div>
    ))}
  </div>
);

const UserMessage = ({ msg }) => (
  <div className="fade-up flex justify-end">
    <div
      className="ml-auto max-w-[92%] md:max-w-[68%]"
      style={{
        background: 'color-mix(in srgb, var(--accent) 11%, var(--surface-card))',
        border: '1px solid color-mix(in srgb, var(--accent) 24%, var(--border-soft))',
        borderRadius: '18px 18px 5px 18px',
        padding: '12px 16px 14px',
        boxShadow: '0 8px 22px rgba(0, 0, 0, 0.05)',
      }}
    >
      <div
        className="label mb-2"
        style={{ color: 'var(--accent)', letterSpacing: '0.18em', textAlign: 'right' }}
      >
        You asked
      </div>
      <div
        className="serif-display"
        style={{
          fontSize: 'calc(18px * var(--folio-font-scale, 1))',
          color: 'var(--text-primary)',
          lineHeight: 1.42,
          fontWeight: 500,
          letterSpacing: 0,
          whiteSpace: 'pre-wrap',
          overflowWrap: 'anywhere',
        }}
      >
        {msg.text}
      </div>
    </div>
  </div>
);

/** Shorten tool names for display. */
const formatToolName = (name = '') => {
  const map = {
    planner: 'planner',
    get_source_outline: 'outline',
    get_source_metadata: 'metadata',
    retrieve_chunks: 'retrieve',
    web_search: 'web',
  };
  return map[name] || name.replace(/^get_source_/, '').replace(/_/g, ' ');
};

/** Lightweight agent trace - shows planner task and tool chain. */
const AgentTrace = ({ msg, show = false }) => {
  // Error path ...no tool trace expected
  if (msg._agentError) {
    return (
      <div className="mt-5 sans" style={{ fontSize: 'calc(11px * var(--folio-font-scale, 1))', color: 'var(--warn)', letterSpacing: '0.06em' }}>
        Agent unavailable - {msg._agentError}
      </div>
    );
  }

  // Not an agent message, or no trace data (historical / saved messages)
  if (!msg._agent || !msg._toolTrace?.length) {
    return null;
  }

  if (!show) {
    return null;
  }

  // Normal agent path
  const task = msg._plannerTask || msg._intent || '';
  const toolNames = msg._toolTrace.map(t => formatToolName(t.name));

  // Cap at 5 display names + overflow count
  let displayTools = toolNames;
  let overflow = 0;
  if (toolNames.length > 5) {
    displayTools = toolNames.slice(0, 5);
    overflow = toolNames.length - 5;
  }

  const toolList = displayTools.join(', ');
  const suffix = overflow > 0 ? ` +${overflow} more` : '';

  return (
    <div className="mt-5 sans" style={{ fontSize: 'calc(11px * var(--folio-font-scale, 1))', color: 'var(--text-muted)', letterSpacing: '0.06em' }}>
      Agent{task ? ` - Task: ${task}` : ''} - Tools: {toolList}{suffix}
      {/* Surfaces only failed tool calls ...ok status is silent */}
      {msg._toolTrace.filter(t => t.status && t.status !== 'ok').map((t, i) => (
        <span key={i} style={{ color: 'var(--warn)' }}>
          {' - '}{formatToolName(t.name)}: {t.error || t.status}
        </span>
      ))}
    </div>
  );
};

const AssistantMessage = ({ msg, onFollowUp, onSaveMessage, onRegenerate, onDelete, showAgentTrace = false }) => {
  // Label is shown only for error states; the normal "Study Companion" /
  // "From your sources" label is noise and removed.
  let label = null;
  let labelColor = 'var(--warn)';
  if (msg._agentError) {
    const code = msg._agentError;
    if (code === 'llm_not_configured') {
      label = 'Agent unavailable';
    } else if (code === 'llm_provider_error') {
      label = 'LLM provider error';
    } else {
      label = 'Error';
    }
  }

  return (
  <div className="fade-up">
    <div className="reading-col">
      {label && (
        <div
          className="label mb-4"
          style={{ color: labelColor, letterSpacing: '0.24em' }}
        >
          {label}
        </div>
      )}
      <MarkdownRich text={msg.text} citations={msg.citations} />

      {msg.plan && <PlanTable plan={msg.plan} />}

      {msg.citations && msg.citations.length > 0 && <Citations citations={msg.citations} />}

      <AgentTrace msg={msg} show={showAgentTrace} />

      <div className="mt-6 flex items-center gap-5 sans" style={{ fontSize: 'calc(11px * var(--folio-font-scale, 1))', color: 'var(--text-muted)', letterSpacing: '0.08em' }}>
        {onFollowUp && (
          <button
            className="inline-flex items-center rounded-full px-3 py-1.5 transition-colors uppercase tracking-wider"
            onClick={() => onFollowUp(msg)}
            title="Ask a follow-up about this answer"
            style={{
              color: 'var(--surface-main)',
              background: 'var(--accent)',
              fontWeight: 500,
            }}
          >
            Follow up
          </button>
        )}
        <button
          className="hover:text-[var(--text-primary)] transition-colors uppercase tracking-wider"
          onClick={() => onSaveMessage && onSaveMessage(msg)}
          title="Save this response"
        >Save</button>
        {onRegenerate && (
          <button
            className="hover:text-[var(--accent)] transition-colors uppercase tracking-wider"
            onClick={() => onRegenerate(msg)}
            title="Regenerate this answer"
          >Regen</button>
        )}
        {onDelete && (
          <button
            className="hover:text-[var(--warn)] transition-colors uppercase tracking-wider"
            onClick={() => onDelete(msg)}
            title="Delete this Q&A turn"
          >Delete</button>
        )}
      </div>
    </div>
  </div>
  );
};

const Message = ({ msg, onFollowUp, onSaveMessage, onRegenerate, onDelete, showAgentTrace = false }) =>
  msg.role === 'user' ? <UserMessage msg={msg} /> : <AssistantMessage msg={msg} onFollowUp={onFollowUp} onSaveMessage={onSaveMessage} onRegenerate={onRegenerate} onDelete={onDelete} showAgentTrace={showAgentTrace} />;

const TypingIndicator = () => (
  <div className="fade-in py-2">
    <div className="reading-col">
      <div className="label" style={{ color: 'var(--text-muted)' }}>Reading sources</div>
      <div className="mt-3 flex gap-2 items-center">
        {[0, 1, 2].map(i => (
          <span
            key={i}
            className="inline-block rounded-full pulse-bar"
            style={{
              width: 5,
              height: 5,
              background: 'var(--accent)',
              opacity: 0.6,
              animationDelay: `${i * 180}ms`,
            }}
          />
        ))}
      </div>
    </div>
  </div>
);

// Notebook hero ...title is the NOTEBOOK, not whatever source is currently previewed.
// If a preview is active, it is shown as a quiet secondary slip below the hero, never as the title.
const NotebookHero = ({ notebookTitle, notebookSubtitle, includedCount, totalSources, onPrompt, prompts, hasMessages, onAddSource }) => {
  return (
    <header className="reading-col mx-auto pt-4 pb-3">
      <div className="label mb-2" style={{ color: 'var(--text-muted)', letterSpacing: '0.24em' }}>
        {totalSources === 0 ? (
          <span style={{ color: 'var(--accent)' }}>Step 2 · Add your first book</span>
        ) : includedCount === 0 ? (
          <span style={{ color: 'var(--accent)' }}>No sources selected - pick at least one to ask</span>
        ) : (
          <span>Ask across your notebook</span>
        )}
      </div>

      <h1
        className="serif-display"
        style={{
          fontSize: 'clamp(calc(22px * var(--folio-font-scale, 1)), 2.8vw, calc(30px * var(--folio-font-scale, 1)))',
          lineHeight: 1.1,
          letterSpacing: 0,
          color: 'var(--text-primary)',
          fontWeight: 400,
        }}
      >
        {notebookTitle}
      </h1>

      <div className="serif-display mt-1" style={{ fontSize: 'calc(14px * var(--folio-font-scale, 1))', color: 'var(--text-muted)', fontStyle: 'italic', fontWeight: 400 }}>
        {notebookSubtitle}
      </div>

      {!hasMessages && totalSources === 0 && (
        <div className="mt-6 rounded-xl px-5 py-6" style={{ background: 'var(--surface-card)', border: '1px solid var(--border-soft)' }}>
          <div className="label mb-2" style={{ color: 'var(--text-muted)' }}>Step 2 of 3</div>
          <div className="serif-display" style={{ fontSize: 'calc(18px * var(--folio-font-scale, 1))', color: 'var(--text-primary)' }}>Add your first book</div>
          <div className="sans mt-1" style={{ fontSize: 'calc(13px * var(--folio-font-scale, 1))', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            Drop in PDFs, EPUBs, slides, notes and more — you can add several at once.
          </div>
          {onAddSource && (
            <button
              onClick={onAddSource}
              className="sans mt-4 rounded-lg px-4 py-2"
              style={{ background: '#3D49C7', color: '#fff', border: 'none', cursor: 'pointer', fontSize: 'calc(13px * var(--folio-font-scale, 1))' }}
            >
              Add sources
            </button>
          )}
        </div>
      )}

      {!hasMessages && totalSources > 0 && prompts && (
        <div className="mt-5">
          <div className="label mb-3" style={{ color: 'var(--text-muted)' }}>Step 3 of 3 · Ask your first question</div>
          <div className="flex flex-col">
            {prompts.slice(0, 4).map((p, i) => (
              <button
                key={i}
                onClick={() => onPrompt(p)}
                className="serif-display text-left py-2 transition-colors group"
                style={{
                  fontSize: 'calc(15px * var(--folio-font-scale, 1))',
                  color: 'var(--text-secondary)',
                  borderTop: i === 0 ? '1px solid var(--border-soft)' : 'none',
                  borderBottom: '1px solid var(--border-soft)',
                  lineHeight: 1.55,
                  fontStyle: 'italic',
                  fontWeight: 400,
                }}
              >
                <span className="group-hover:text-[var(--text-primary)] transition-colors">{p}</span>
                <span
                  className="ml-2 opacity-0 group-hover:opacity-100 transition-opacity sans not-italic"
                  style={{ fontSize: 'calc(11px * var(--folio-font-scale, 1))', color: 'var(--accent)', letterSpacing: '0.1em' }}
                >
                  ->
                </span>
              </button>
            ))}
          </div>
        </div>
      )}

      {hasMessages && (
        <div className="hairline mt-5" />
      )}
    </header>
  );
};

const ChatCenter = ({ notebookTitle, notebookSubtitle, includedCount, totalSources, prompts, usingFallback, chat, setChat, thinking, setThinking, onSend, onSaveTurn, onRegenerateTurn, onDeleteTurn, sourceDocumentIds = [], continueFromPrev, setContinueFromPrev, internetEnabled = false, setInternetEnabled, draftInput, setDraftInput, enterToSend, showAgentTrace, chatScrollKey = '', onAddSource, llmConfigured, onRequireSettings }) => {
  const [localInput, setLocalInput] = React.useState('');
  const input = draftInput ?? localInput;
  const updateInput = setDraftInput || setLocalInput;
  const scrollRef = React.useRef(null);
  const inputRef = React.useRef(null);
  const restoredScrollRef = React.useRef(false);
  const stickToBottomRef = React.useRef(true);
  const saveScrollTimerRef = React.useRef(null);
  const chatPreferences = { enterToSend, showAgentTrace };

  const isNearBottom = React.useCallback((el) => {
    if (!el) return true;
    return el.scrollHeight - el.scrollTop - el.clientHeight < 120;
  }, []);

  const saveScrollPosition = React.useCallback(() => {
    if (!chatScrollKey || !scrollRef.current) return;
    try {
      window.localStorage?.setItem(chatScrollKey, String(Math.max(0, Math.round(scrollRef.current.scrollTop))));
    } catch (e) {}
  }, [chatScrollKey]);

  const scheduleSaveScrollPosition = React.useCallback(() => {
    if (!chatScrollKey || !restoredScrollRef.current) return;
    if (saveScrollTimerRef.current) window.clearTimeout(saveScrollTimerRef.current);
    saveScrollTimerRef.current = window.setTimeout(() => {
      saveScrollTimerRef.current = null;
      saveScrollPosition();
    }, 160);
  }, [chatScrollKey, saveScrollPosition]);

  const resizeInput = React.useCallback(() => {
    const el = inputRef.current;
    if (!el) return;
    const maxHeight = 180;
    el.style.height = 'auto';
    const nextHeight = Math.min(el.scrollHeight, maxHeight);
    el.style.height = `${nextHeight}px`;
    el.style.overflowY = el.scrollHeight > maxHeight ? 'auto' : 'hidden';
  }, []);

  React.useLayoutEffect(() => {
    const el = scrollRef.current;
    if (!el || restoredScrollRef.current || chat.length === 0) return;
    const raw = chatScrollKey ? window.localStorage?.getItem(chatScrollKey) : null;
    const saved = raw ? Number(raw) : NaN;
    window.requestAnimationFrame(() => {
      if (Number.isFinite(saved)) {
        el.scrollTop = Math.min(saved, Math.max(0, el.scrollHeight - el.clientHeight));
        stickToBottomRef.current = isNearBottom(el);
      } else {
        el.scrollTop = el.scrollHeight;
        stickToBottomRef.current = true;
      }
      restoredScrollRef.current = true;
    });
  }, [chat.length, chatScrollKey, isNearBottom]);

  React.useEffect(() => {
    const el = scrollRef.current;
    if (!el || !restoredScrollRef.current) return;
    if (stickToBottomRef.current) {
      window.requestAnimationFrame(() => {
        el.scrollTop = el.scrollHeight;
        saveScrollPosition();
      });
    }
  }, [chat.length, thinking, saveScrollPosition]);

  React.useEffect(() => {
    return () => {
      if (saveScrollTimerRef.current) window.clearTimeout(saveScrollTimerRef.current);
      saveScrollPosition();
    };
  }, [saveScrollPosition]);

  React.useLayoutEffect(() => {
    resizeInput();
  }, [input, resizeInput]);

  const send = (text) => {
    const t = (text ?? input).trim();
    if (!t || includedCount === 0 || thinking) return;
    if (llmConfigured === false) {
      onRequireSettings?.();
      return; // keep the draft — don't clear the user's question
    }
    restoredScrollRef.current = true;
    stickToBottomRef.current = true;
    updateInput('');
    onSend(t, { allowWebOverride: internetEnabled });
  };

  const disabled = includedCount === 0 || thinking;
  const sendShortcut = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform)
    ? 'Cmd+Enter'
    : 'Ctrl+Enter';
  const inputHint = chatPreferences.enterToSend
    ? 'Enter to send / Shift+Enter for newline'
    : `Enter for newline / ${sendShortcut} to send`;

  const handleInputKeyDown = (e) => {
    if (e.key !== 'Enter' || disabled) return;
    if (chatPreferences.enterToSend) {
      if (!e.shiftKey && !e.ctrlKey && !e.metaKey && !e.altKey) {
        e.preventDefault();
        send();
      }
      return;
    }
    if ((e.ctrlKey || e.metaKey) && !e.shiftKey && !e.altKey) {
      e.preventDefault();
      send();
    }
  };

  const handleSaveMessage = async (msg) => {
    const idx = chat.findIndex(m => m.id === msg.id);
    const userMsg = idx > 0 ? chat[idx - 1] : null;
    const messages = [];
    if (userMsg && userMsg.role === 'user') {
      messages.push({ role: 'user', text: userMsg.text || '' });
    }
    messages.push({ role: 'assistant', text: msg.text || '', citations: msg.citations || [] });
    const baseTitle = (userMsg?.text || msg.text || 'Saved answer').slice(0, 80).replace(/\n/g, ' ');
    const title = baseTitle + (baseTitle.length >= 80 ? '...' : '');
    try {
      if (onSaveTurn) {
        await onSaveTurn({
          title,
          messages,
          sourceDocumentIds: msg._sourceDocumentIds || userMsg?._sourceDocumentIds || sourceDocumentIds,
          quizContext: msg._quizContext || userMsg?._quizContext || null,
        });
      } else {
        await window.FOLIO_API.createArtifact('note', title, { messages, quiz_context: msg._quizContext || null }, msg._sourceDocumentIds || sourceDocumentIds, { origin: 'chat_save' });
      }
    } catch (err) {
      alert('Failed to save: ' + (err.message || 'Unknown error'));
    }
  };

  const handleFollowUp = (msg) => {
    const idx = chat.findIndex(m => m.id === msg.id);
    const userMsg = idx > 0 ? chat[idx - 1] : null;
    setContinueFromPrev({
      assistantId: msg.id,
      userText: userMsg?.role === 'user' ? (userMsg.text || '') : '',
      assistantText: msg.text || '',
      intent: msg._intent || '',
      documentIds: msg._sourceDocumentIds || userMsg?._sourceDocumentIds || sourceDocumentIds,
    });
    window.requestAnimationFrame(() => inputRef.current?.focus());
  };

  const handleRegenerate = (msg) => {
    if (onRegenerateTurn) onRegenerateTurn(msg.id);
  };

  const handleDelete = (msg) => {
    if (onDeleteTurn) onDeleteTurn(msg.id);
  };

  return (
    <div className="h-full flex flex-col" style={{ background: 'var(--surface-main)' }}>
      {/* Reading area ...no top toolbar, the page IS the surface */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto scroll-thin"
        onScroll={() => {
          const el = scrollRef.current;
          stickToBottomRef.current = isNearBottom(el);
          scheduleSaveScrollPosition();
        }}
      >
        <div className="px-12">
          <NotebookHero
            notebookTitle={notebookTitle}
            notebookSubtitle={notebookSubtitle}
            includedCount={includedCount}
            totalSources={totalSources}
            prompts={prompts}
            hasMessages={chat.length > 0}
            onPrompt={(p) => send(p)}
            onAddSource={onAddSource}
          />

          {/* Fallback banner ...shown when backend is unreachable */}
          {usingFallback && (
            <div className="reading-col mx-auto mt-6 px-4 py-3 rounded-lg" style={{ background: 'var(--surface-card)', border: '1px solid var(--border-soft)' }}>
              <div className="sans" style={{ fontSize: 'calc(11px * var(--folio-font-scale, 1))', color: 'var(--warn)', letterSpacing: '0.04em', fontWeight: 400 }}>
                Backend unavailable - showing your last loaded data.
              </div>
              <div className="serif-display mt-1" style={{ fontSize: 'calc(13px * var(--folio-font-scale, 1))', color: 'var(--text-secondary)', fontStyle: 'italic' }}>
                Restart the app or retry to reconnect.
              </div>
            </div>
          )}

          {chat.length > 0 && (
            <div className="reading-col mx-auto pb-12 space-y-10">
              {chat.map(m => <Message key={m.id} msg={m} onFollowUp={handleFollowUp} onSaveMessage={handleSaveMessage} onRegenerate={onRegenerateTurn ? handleRegenerate : null} onDelete={onDeleteTurn ? handleDelete : null} showAgentTrace={chatPreferences.showAgentTrace} />)}
              {thinking && <TypingIndicator />}
            </div>
          )}
        </div>
      </div>

      {/* Composer ...a single line, not a card */}
      <div className="px-12 pt-2 pb-4" style={{ background: 'var(--surface-main)' }}>
        <form
          onSubmit={(e) => { e.preventDefault(); send(); }}
          className="reading-col mx-auto"
        >
          {/* Continue chip ...visible when next query will carry previous context */}
          {continueFromPrev && (
            <div className="mb-2 flex items-center gap-2">
              <span
                className="sans inline-flex items-center gap-1 px-2 py-0.5 rounded"
                style={{
                  fontSize: 'calc(10.5px * var(--folio-font-scale, 1))',
                  color: 'var(--accent)',
                  background: 'var(--accent-bg, #EEF0FB)',
                  letterSpacing: '0.04em',
                  fontWeight: 500,
                }}
              >
                Following up
                <button
                  type="button"
                  onClick={() => setContinueFromPrev(false)}
                  className="ml-1 hover:opacity-70"
                  style={{ fontSize: 'calc(13px * var(--folio-font-scale, 1))', lineHeight: 1 }}
                  aria-label="Cancel follow-up"
                  title="Cancel follow-up"
                >
                  <Icon name="x" size={12} />
                </button>
              </span>
            </div>
          )}
          <div
            className="flex items-end gap-3 py-3"
            style={{
              borderBottom: `1px solid ${disabled ? 'var(--border-soft)' : 'var(--border-medium)'}`,
              opacity: disabled ? 0.55 : 1,
            }}
          >
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => updateInput(e.target.value)}
              onKeyDown={handleInputKeyDown}
              placeholder={disabled ? 'Select a source to ask...' : continueFromPrev ? 'Follow up...' : 'Ask across selected sources...'}
              rows={1}
              disabled={disabled}
              className="flex-1 bg-transparent outline-none resize-none serif-display"
              style={{
                fontSize: 'var(--reading-font-size,18px)',
                lineHeight: 1.55,
                color: 'var(--text-primary)',
                fontWeight: 400,
                minHeight: 28,
                width: '100%',
                overflowY: 'hidden',
              }}
            />
            <button
              type="submit"
              disabled={!input.trim() || disabled}
              className="sans transition-opacity"
              style={{
                fontSize: 'calc(11px * var(--folio-font-scale, 1))',
                letterSpacing: '0.18em',
                textTransform: 'uppercase',
                color: input.trim() && !disabled ? 'var(--surface-main)' : 'var(--text-muted)',
                background: input.trim() && !disabled ? 'var(--accent)' : 'transparent',
                border: `1px solid ${input.trim() && !disabled ? 'var(--accent)' : 'var(--border-soft)'}`,
                borderRadius: 'var(--radius-button)',
                padding: '6px 12px',
              }}
            >
              Ask ->
            </button>
          </div>
          <div className="flex justify-between mt-2 sans" style={{ fontSize: 'calc(10.5px * var(--folio-font-scale, 1))', color: 'var(--text-muted)', letterSpacing: '0.04em' }}>
            <span>
              {includedCount} {includedCount === 1 ? 'source' : 'sources'} in scope
              <button
                type="button"
                onClick={() => setInternetEnabled && setInternetEnabled(v => !v)}
                aria-pressed={internetEnabled}
                className="ml-3 sans"
                style={{
                  fontSize: 'calc(10.5px * var(--folio-font-scale, 1))',
                  letterSpacing: '0.06em',
                  textTransform: 'uppercase',
                  color: internetEnabled ? 'var(--ok)' : 'var(--text-muted)',
                  border: `1px solid ${internetEnabled ? 'var(--ok)' : 'var(--border-soft)'}`,
                  borderRadius: 'var(--radius-button)',
                  padding: '2px 7px',
                  background: internetEnabled ? 'color-mix(in srgb, var(--ok) 12%, transparent)' : 'transparent',
                }}
                title="Allow Internet research for this question"
              >
                Internet {internetEnabled ? 'on' : 'off'}
              </button>
            </span>
            <span>{inputHint}</span>
          </div>
        </form>
    </div>
  </div>
  );
};

window.ChatCenter = ChatCenter;
