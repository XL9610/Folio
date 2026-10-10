// Settings modal: preferences, themes, LLM API, and internet research.

const SETTINGS_API_BASE = (typeof window !== 'undefined' && window.FOLIO_API_BASE) || 'http://127.0.0.1:8000';

const LLM_PROVIDER_PRESETS = {
  openai:    { label: 'OpenAI',    base_url: 'https://api.openai.com/v1', model: 'gpt-4o', api_format: 'openai-compatible' },
  anthropic: { label: 'Anthropic', base_url: 'https://api.anthropic.com/v1', model: 'claude-sonnet-5', api_format: 'anthropic' },
  google:    { label: 'Google',    base_url: 'https://generativelanguage.googleapis.com/v1beta', model: 'gemini-2.5-flash', api_format: 'google' },
  deepseek:  { label: 'DeepSeek',  base_url: 'https://api.deepseek.com/v1', model: 'deepseek-chat', api_format: 'openai-compatible' },
  custom:    { label: 'Custom',    base_url: '', model: '', api_format: 'openai-compatible' },
  local:     { label: 'Local',     base_url: 'http://localhost:11434/v1', model: 'llama3', api_format: 'openai-compatible' },
};

const PROVIDER_MODELS = {
  openai:    ['gpt-4o', 'gpt-4o-mini', 'gpt-4.1', 'o4-mini'],
  anthropic: ['claude-opus-5', 'claude-sonnet-5', 'claude-fable-5-1', 'claude-haiku-4-5-20251001'],
  google:    ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.0-flash'],
  deepseek:  ['deepseek-v4-flash', 'deepseek-v4-pro'],
  local:     ['llama3', 'llama3:70b', 'mistral', 'qwen2.5', 'gemma3'],
};

const labelize = (value) => String(value || '')
  .split('-')
  .map((part) => part ? part[0].toUpperCase() + part.slice(1) : part)
  .join(' ');

const optionLabel = (options, value) => {
  const found = options.find((option) => {
    const normalized = typeof option === 'object' ? option : { value: option };
    return normalized.value === value;
  });
  if (!found) return labelize(value);
  return typeof found === 'object' ? (found.label || labelize(found.value)) : labelize(found);
};

const SettingsSection = ({ label, summary, defaultOpen = false, children }) => {
  const [open, setOpen] = React.useState(defaultOpen);
  return (
    <section
      style={{
        marginBottom: 10,
        background: 'var(--surface-card)',
        border: `1px solid ${open ? 'var(--border-medium)' : 'var(--border-soft)'}`,
        borderRadius: 8,
        overflow: 'hidden',
      }}
    >
      <button
        type="button"
        onClick={() => setOpen(v => !v)}
        aria-expanded={open}
        className="w-full flex items-center gap-3 text-left px-4 py-3 transition-colors"
        style={{ color: 'var(--text-primary)' }}
      >
        <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>
          <Icon name={open ? 'chevronDown' : 'chevronRight'} size={15} />
        </span>
        <span className="label" style={{ color: 'var(--text-primary)', flexShrink: 0 }}>{label}</span>
        {summary && (
          <span
            className="sans"
            style={{
              marginLeft: 'auto',
              minWidth: 0,
              color: 'var(--text-secondary)',
              fontSize: 'calc(11.5px * var(--folio-font-scale, 1))',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {summary}
          </span>
        )}
      </button>
      {open && (
        <div
          className="px-4 pt-3 pb-4"
          style={{ borderTop: '1px solid var(--border-soft)', background: 'var(--surface-main)' }}
        >
          {children}
        </div>
      )}
    </section>
  );
};

const SettingsInnerSection = ({ label, summary, defaultOpen = false, buttonLabel, children }) => {
  const [open, setOpen] = React.useState(defaultOpen);
  return (
    <div style={{ borderTop: '1px solid var(--border-soft)' }}>
      <button
        type="button"
        onClick={() => setOpen(v => !v)}
        aria-expanded={open}
        aria-label={buttonLabel || undefined}
        className="w-full flex items-center gap-2 text-left py-3 transition-colors"
        style={{ color: 'var(--text-primary)' }}
      >
        <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>
          <Icon name={open ? 'chevronDown' : 'chevronRight'} size={14} />
        </span>
        <span className="sans" style={{ fontSize: 'calc(12px * var(--folio-font-scale, 1))', color: 'var(--text-primary)', fontWeight: 600 }}>{label}</span>
        {summary && (
          <span
            className="sans"
            style={{
              marginLeft: 'auto',
              minWidth: 0,
              color: 'var(--text-secondary)',
              fontSize: 'calc(11px * var(--folio-font-scale, 1))',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {summary}
          </span>
        )}
      </button>
      {open && <div className="pb-3">{children}</div>}
    </div>
  );
};

const SettingsSubhead = ({ children, detail }) => (
  <div style={{ marginTop: 4, marginBottom: 12 }}>
    <div className="sans" style={{ fontSize: 'calc(11px * var(--folio-font-scale, 1))', color: 'var(--text-muted)', letterSpacing: '0.14em', textTransform: 'uppercase' }}>
      {children}
    </div>
    {detail && (
      <div className="sans mt-1" style={{ fontSize: 'calc(11.5px * var(--folio-font-scale, 1))', color: 'var(--text-secondary)', lineHeight: 1.45 }}>
        {detail}
      </div>
    )}
  </div>
);

const SettingsField = ({ label, hint, children }) => (
  <div style={{ marginBottom: 14 }}>
    {label && (
      <div className="sans" style={{ fontSize: 'calc(11.5px * var(--folio-font-scale, 1))', color: 'var(--text-primary)', letterSpacing: '0.04em', fontWeight: 600, marginBottom: 5 }}>
        {label}
      </div>
    )}
    {children}
    {hint && (
      <div className="sans mt-1" style={{ fontSize: 'calc(10.5px * var(--folio-font-scale, 1))', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
        {hint}
      </div>
    )}
  </div>
);

const SettingsNote = ({ children, tone = 'neutral' }) => (
  <div
    className="mb-3 p-3 rounded-lg sans"
    style={{
      fontSize: 'calc(11px * var(--folio-font-scale, 1))',
      color: tone === 'warn' ? 'var(--warn)' : 'var(--text-secondary)',
      background: 'var(--surface-card)',
      border: `1px solid ${tone === 'warn' ? 'var(--warn)' : 'var(--border-soft)'}`,
      lineHeight: 1.5,
    }}
  >
    {children}
  </div>
);

const SettingsButton = ({ children, onClick, disabled, variant = 'secondary' }) => (
  <button
    type="button"
    onClick={onClick}
    disabled={disabled}
    className="sans rounded-full px-4 py-2 text-[12px] transition-colors inline-flex items-center justify-center gap-1.5"
    style={{
      background: variant === 'primary' ? 'var(--accent)' : (disabled ? 'var(--border-soft)' : 'var(--surface-card)'),
      border: `1px solid ${variant === 'primary' ? 'var(--accent)' : 'var(--border-medium)'}`,
      color: variant === 'primary' ? 'var(--surface-main)' : 'var(--text-primary)',
      opacity: disabled ? 0.6 : 1,
      cursor: disabled ? 'not-allowed' : 'pointer',
    }}
  >
    {children}
  </button>
);

const ToggleControl = ({ checked, onChange, label, description }) => (
  <button
    type="button"
    role="switch"
    aria-checked={!!checked}
    onClick={() => onChange(!checked)}
    className="w-full flex items-center gap-3 text-left rounded-lg border px-3 py-2.5 transition-colors"
    style={{
      borderColor: checked ? 'var(--accent)' : 'var(--border-soft)',
      background: checked ? 'var(--surface-selected)' : 'var(--surface-card)',
      color: 'var(--text-primary)',
    }}
  >
    <span
      aria-hidden="true"
      style={{
        width: 32,
        height: 18,
        borderRadius: 999,
        padding: 2,
        background: checked ? 'var(--accent)' : 'var(--border-medium)',
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: checked ? 'flex-end' : 'flex-start',
        flexShrink: 0,
      }}
    >
      <span style={{ width: 14, height: 14, borderRadius: '50%', background: 'var(--surface-main)', display: 'block' }} />
    </span>
    <span className="min-w-0">
      <span className="sans block" style={{ fontSize: 'calc(12.5px * var(--folio-font-scale, 1))', color: 'var(--text-primary)' }}>{label}</span>
      {description && (
        <span className="sans block mt-0.5" style={{ fontSize: 'calc(11px * var(--folio-font-scale, 1))', color: 'var(--text-secondary)', lineHeight: 1.35 }}>
          {description}
        </span>
      )}
    </span>
  </button>
);

const SegmentedControl = ({ value, onChange, options }) => {
  const normalized = options.map((option) =>
    typeof option === 'object' ? option : { value: option, label: labelize(option) }
  );
  return (
    <div className="flex flex-wrap gap-1.5">
      {normalized.map((option) => {
        const active = value === option.value;
        return (
          <button
            key={String(option.value)}
            type="button"
            onClick={() => onChange(option.value)}
            className="rounded-full border transition-all px-2.5 py-1.5 sans text-[11px]"
            style={{
              borderColor: active ? 'var(--accent)' : 'var(--border-soft)',
              background: active ? 'var(--surface-selected)' : 'var(--surface-card)',
              color: 'var(--text-primary)',
            }}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
};

const SelectControl = ({ value, onChange, options }) => (
  <select
    value={value}
    onChange={(event) => onChange(event.target.value)}
    className="w-full rounded-lg border px-3 py-2 sans text-[13px]"
    style={{
      background: 'var(--surface-card)',
      borderColor: 'var(--border-medium)',
      color: 'var(--text-primary)',
    }}
  >
    {options.map((option) => {
      const normalized = typeof option === 'object' ? option : { value: option, label: labelize(option) };
      return <option key={String(normalized.value)} value={normalized.value}>{normalized.label}</option>;
    })}
  </select>
);

const NumberControl = ({ value, onChange, min, max, step = 1 }) => {
  const rawValue = Number.parseInt(value, 10);
  const boundedValue = Number.isFinite(rawValue) ? Math.max(min, Math.min(max, rawValue)) : min;
  const commit = (next) => {
    const parsed = Number.parseInt(next, 10);
    if (!Number.isFinite(parsed)) return;
    onChange(Math.max(min, Math.min(max, parsed)));
  };
  return (
    <div className="inline-flex items-center rounded-full border overflow-hidden" style={{ borderColor: 'var(--border-medium)', background: 'var(--surface-card)' }}>
      <button
        type="button"
        aria-label="Decrease"
        onClick={() => commit(String(boundedValue - step))}
        className="px-3 py-2"
        style={{ color: 'var(--text-primary)', opacity: boundedValue <= min ? 0.4 : 1 }}
        disabled={boundedValue <= min}
      >
        -
      </button>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={boundedValue}
        onChange={(event) => commit(event.target.value)}
        className="sans text-center outline-none"
        style={{
          width: 56,
          fontSize: 'calc(12px * var(--folio-font-scale, 1))',
          color: 'var(--text-primary)',
          background: 'transparent',
          borderLeft: '1px solid var(--border-soft)',
          borderRight: '1px solid var(--border-soft)',
          padding: '7px 4px',
        }}
      />
      <button
        type="button"
        aria-label="Increase"
        onClick={() => commit(String(boundedValue + step))}
        className="px-3 py-2"
        style={{ color: 'var(--text-primary)', opacity: boundedValue >= max ? 0.4 : 1 }}
        disabled={boundedValue >= max}
      >
        +
      </button>
    </div>
  );
};

const useFolioSettings = () => {
  const [settings, setSettingsState] = React.useState(() => window.getSettings());

  React.useEffect(() => {
    return window.subscribeSettings((next) => setSettingsState(next));
  }, []);

  const updateSettings = React.useCallback((patch) => {
    const next = window.setSettings(patch);
    setSettingsState(next);
  }, []);

  const resetAllSettings = React.useCallback(() => {
    const next = window.resetSettings();
    setSettingsState(next);
  }, []);

  return [settings, updateSettings, resetAllSettings];
};

const SettingsModal = ({
  open,
  onClose,
  theme,
  onThemeChange,
  themes,
  onResetSourceScope,
  onConfigSaved,
}) => {
  const [settings, updateSettings, resetAllSettings] = useFolioSettings();
  const llmCfg = (settings && settings._llm) || {};
  const webCfg = (settings && settings._webSearch) || {};
  const savedProvider = React.useMemo(() => localStorage.getItem('folio.llmProvider') || '', []);
  const [provider, setProvider] = React.useState(llmCfg.provider || savedProvider || '');
  const [apiKey, setApiKey] = React.useState('');
  const [apiFormat, setApiFormat] = React.useState(llmCfg.api_format || 'openai-compatible');
  const [baseUrl, setBaseUrl] = React.useState(llmCfg.base_url || '');
  const [model, setModel] = React.useState(llmCfg.model || '');
  const [isCustomModel, setIsCustomModel] = React.useState(false);
  const [testStatus, setTestStatus] = React.useState(null);
  const [saving, setSaving] = React.useState(false);
  const [llmStatus, setLlmStatus] = React.useState(llmCfg.configured ? llmCfg : null);
  const [webEnabled, setWebEnabled] = React.useState(!!(webCfg.configured || webCfg.provider));
  const [webProvider, setWebProvider] = React.useState(webCfg.provider === 'custom' ? 'custom' : 'tavily');
  const [webApiKey, setWebApiKey] = React.useState('');
  const [webBaseUrl, setWebBaseUrl] = React.useState(webCfg.base_url || '');
  const [webStatus, setWebStatus] = React.useState(webCfg.configured ? webCfg : null);
  const [webTestStatus, setWebTestStatus] = React.useState(null);
  const [webSaving, setWebSaving] = React.useState(false);

  React.useEffect(() => {
    if (!open) return;
    (async () => {
      try {
        const res = await fetch(`${SETTINGS_API_BASE}/api/config/llm-status`);
        const data = await res.json();
        setLlmStatus(data);
        if (data.provider) setProvider(data.provider);
        if (data.api_format) setApiFormat(data.api_format);
        if (data.base_url) setBaseUrl(data.base_url);
        if (data.model) {
          const presetModels = PROVIDER_MODELS[data.provider] || [];
          setModel(data.model);
          setIsCustomModel(!presetModels.includes(data.model));
        }
      } catch (e) {}
    })();
  }, [open]);

  React.useEffect(() => {
    if (!open) return;
    (async () => {
      try {
        const res = await fetch(`${SETTINGS_API_BASE}/api/config/web-search-status`);
        const data = await res.json();
        setWebStatus(data);
        setWebEnabled(!!(data.configured || data.provider));
        if (data.provider === 'custom') setWebProvider('custom');
        if (data.base_url) setWebBaseUrl(data.base_url);
        setWebApiKey('');
      } catch (e) {}
    })();
  }, [open]);

  const handleProviderChange = (key) => {
    setProvider(key);
    setApiKey(''); // switching provider must not carry a previously-typed key
    const preset = LLM_PROVIDER_PRESETS[key];
    if (preset) {
      setBaseUrl(preset.base_url);
      setApiFormat(preset.api_format || 'openai-compatible');
      setModel(preset.model);
      setIsCustomModel(false);
    }
    setTestStatus(null);
  };

  const handleSave = async (silent) => {
    setSaving(true);
    try {
      const res = await fetch(`${SETTINGS_API_BASE}/api/config/llm`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider, api_format: apiFormat, api_key: apiKey, base_url: baseUrl, model }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `Save failed: ${res.status}`);
      localStorage.setItem('folio.llmProvider', provider);
      localStorage.setItem('folio.llmApiFormat', apiFormat);
      if (baseUrl) localStorage.setItem('folio.llmBaseUrl', baseUrl);
      if (model) localStorage.setItem('folio.llmModel', model);
      if (!silent) setTestStatus(null);
      setApiKey('');
      setLlmStatus({ configured: !!data.configured, provider, api_format: apiFormat, base_url: baseUrl, model, has_api_key: provider !== 'local' && !!data.configured });
      onConfigSaved?.();
      return true;
    } catch (e) {
      if (!silent) alert('Failed to save: ' + (e.message || 'Unknown error'));
      return false;
    } finally {
      setSaving(false);
    }
  };

  const handleTest = async () => {
    setTestStatus('testing');
    try {
      const saved = await handleSave(true);
      if (!saved) {
        setTestStatus({ ok: false, error: 'Configuration did not save. Fix the settings and try again.' });
        return;
      }
      const res = await fetch(`${SETTINGS_API_BASE}/api/config/test-llm`, { method: 'POST' });
      const data = await res.json();
      setTestStatus(data);
    } catch (e) {
      setTestStatus({ ok: false, error: e.message || 'Connection failed' });
    }
  };

  const handleDisable = async () => {
    setSaving(true);
    try {
      const res = await fetch(`${SETTINGS_API_BASE}/api/config/llm`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ disabled: true }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `Disable failed: ${res.status}`);
      setLlmStatus({ configured: false });
      setApiKey('');
      setTestStatus(null);
      onConfigSaved?.();
    } catch (e) {
      alert('Failed to disable LLM: ' + (e.message || 'Unknown error'));
    } finally {
      setSaving(false);
    }
  };

  const handleSaveWebSearch = async (silent) => {
    setWebSaving(true);
    try {
      const provider = webEnabled ? webProvider : 'disabled';
      const res = await fetch(`${SETTINGS_API_BASE}/api/config/web-search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider, api_key: webApiKey, base_url: webBaseUrl }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `Save failed: ${res.status}`);
      setWebStatus({ configured: !!data.configured, provider: data.provider || '', base_url: webBaseUrl });
      setWebApiKey('');
      setWebTestStatus(null);
    } catch (e) {
      if (!silent) alert('Failed to save web search settings: ' + (e.message || 'Unknown error'));
    } finally {
      setWebSaving(false);
    }
  };

  const handleTestWebSearch = async () => {
    setWebTestStatus('testing');
    try {
      await handleSaveWebSearch(true);
      const res = await fetch(`${SETTINGS_API_BASE}/api/config/test-web-search`, { method: 'POST' });
      const data = await res.json();
      setWebTestStatus(data);
    } catch (e) {
      setWebTestStatus({ ok: false, error: e.message || 'Connection failed' });
    }
  };

  const handleResetSourceScope = () => {
    updateSettings({ sources: { openScope: window.FOLIO_SETTINGS_DEFAULTS.sources.openScope } });
    if (onResetSourceScope) onResetSourceScope();
  };

  const presetModels = PROVIDER_MODELS[provider] || [];
  const themeSummary = themes?.[theme]?.label || theme || 'Theme';
  const providerLabel = provider ? (LLM_PROVIDER_PRESETS[provider]?.label || provider) : '';
  const llmSummary = llmStatus?.configured
    ? [providerLabel || llmStatus.provider || 'Configured', model || llmStatus.model].filter(Boolean).join(' - ')
    : providerLabel
      ? `${providerLabel} not saved`
      : 'Not configured';
  const webSummary = webStatus?.configured ? (webStatus.provider === 'custom' ? 'Custom configured' : 'Tavily configured') : 'Not configured';
  const modalFontScale = settings?.layout?.fontScale || 'normal';
  const studioFlashcardCount = Math.max(10, Math.min(50, Number.parseInt(settings.studio.flashcardCount, 10) || 20));
  const appearanceSummary = `${themeSummary} - ${optionLabel(['compact', 'normal', 'large'], settings.layout.fontScale)} type`;
  const aiSummary = `${llmSummary}; internet ${settings.ai.internetDefault ? 'default on' : 'default off'}`;
  const answerSummary = `${optionLabel(['brief', 'normal', 'detailed'], settings.answer.length)} ${settings.answer.format}`;
  const studioSummary = `${settings.studio.summaryLength} summaries, ${studioFlashcardCount} cards`;
  const sourcesSummary = `${optionLabel([
    { value: 'last-or-all', label: 'Last or all' },
    { value: 'all', label: 'All' },
    { value: 'none', label: 'None' },
  ], settings.sources.openScope)} - ${settings.sources.sort}`;

  const update = (section, key, value) => updateSettings({ [section]: { [key]: value } });

  if (!open) return null;

  return (
    <div className={`fixed inset-0 z-50 flex items-center justify-center fade-in folio-font-${modalFontScale}`} data-font-scale={modalFontScale}>
      <div className="absolute inset-0 backdrop-blur-[2px]" style={{ background: 'rgba(0,0,0,0.62)' }} onClick={onClose} />
      <div className="relative bg-card border border-line rounded-lg w-[92vw] max-w-[640px] max-h-[85vh] flex flex-col" style={{ boxShadow: '0 8px 32px rgba(0,0,0,0.24)' }}>
        <div className="flex items-center justify-between px-6 pt-5 pb-4 border-b border-line2">
          <div className="serif-display text-[20px]">Settings</div>
          <button type="button" onClick={onClose} className="text-muted hover:text-ink p-1 rounded-full" aria-label="Close settings">
            <Icon name="x" size={18} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto scroll-thin px-6 py-5">
          <SettingsSection label="Appearance" summary={appearanceSummary} defaultOpen>
            <SettingsInnerSection label="Theme" summary={themeSummary} defaultOpen>
              <SettingsField>
                {themes && (
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(themes).map(([key, currentTheme]) => (
                      <button
                        key={key}
                        type="button"
                        onClick={() => { onThemeChange(key); localStorage.setItem('folio.theme', key); }}
                        className="rounded-full border transition-all px-3 py-2 sans inline-flex items-center gap-2"
                        style={{
                          borderColor: theme === key ? 'var(--accent)' : 'var(--border-soft)',
                          background: theme === key ? 'var(--surface-selected)' : 'var(--surface-card)',
                          fontSize: 'calc(12px * var(--folio-font-scale, 1))',
                          color: 'var(--text-primary)',
                        }}
                      >
                        <span style={{ width: 12, height: 12, borderRadius: '50%', background: currentTheme.vars['--accent'], display: 'inline-block', flexShrink: 0 }} />
                        {currentTheme.label}
                      </button>
                    ))}
                  </div>
                )}
              </SettingsField>
            </SettingsInnerSection>

            <SettingsSubhead>Layout</SettingsSubhead>
            <SettingsField label="Font scale">
              <SegmentedControl
                value={settings.layout.fontScale}
                onChange={(value) => update('layout', 'fontScale', value)}
                options={[
                  { value: 'compact', label: 'Compact' },
                  { value: 'normal', label: 'Normal' },
                  { value: 'large', label: 'Large' },
                ]}
              />
            </SettingsField>
            <SettingsField label="Reading width">
              <SegmentedControl
                value={settings.layout.readingWidth}
                onChange={(value) => update('layout', 'readingWidth', value)}
                options={[
                  { value: 'narrow', label: 'Narrow' },
                  { value: 'normal', label: 'Normal' },
                  { value: 'wide', label: 'Wide' },
                ]}
              />
            </SettingsField>
          </SettingsSection>

          <SettingsSection label="AI" summary={aiSummary} defaultOpen>
            {llmStatus && !llmStatus.configured && (
              <SettingsNote tone="warn">
                Set up an LLM provider to unlock Studio features and conversational Q&A.
              </SettingsNote>
            )}

            <SettingsSubhead detail="Controls the default composer state for internet research. The Tavily credential remains configured below.">
              Defaults
            </SettingsSubhead>
            <SettingsField>
              <ToggleControl
                checked={settings.ai.internetDefault}
                onChange={(value) => update('ai', 'internetDefault', value)}
                label="Internet research on by default"
                description="Start new chat sessions with web research enabled."
              />
            </SettingsField>

            <SettingsInnerSection label="LLM provider" buttonLabel={`LLM API ${llmSummary}`} summary={llmSummary}>
            <SettingsSubhead detail="Provider settings are saved through the existing backend config flow and local provider keys.">
              Provider
            </SettingsSubhead>
            <SettingsField label="Provider">
              <div className="flex flex-wrap gap-1.5">
                {Object.entries(LLM_PROVIDER_PRESETS).map(([key, preset]) => (
                  <button
                    key={key}
                    type="button"
                    onClick={() => handleProviderChange(key)}
                    className="rounded-full border transition-all px-2.5 py-1.5 sans text-[11px]"
                    style={{
                      borderColor: provider === key ? 'var(--accent)' : 'var(--border-soft)',
                      background: provider === key ? 'var(--surface-selected)' : 'var(--surface-card)',
                      color: 'var(--text-primary)',
                    }}
                  >
                    {preset.label}
                  </button>
                ))}
              </div>
            </SettingsField>
            <SettingsField label="API Format">
              <SegmentedControl
                value={apiFormat}
                onChange={setApiFormat}
                options={[
                  { value: 'openai-compatible', label: 'OpenAI-compatible' },
                  { value: 'anthropic', label: 'Anthropic' },
                  { value: 'google', label: 'Google' },
                ]}
              />
            </SettingsField>
            <SettingsField label="API Key">
              <input
                type="password"
                value={apiKey}
                onChange={e => setApiKey(e.target.value)}
                placeholder={llmStatus?.has_api_key ? 'Configured - leave blank to keep current key' : 'Paste API key...'}
                className="w-full rounded-lg border px-3 py-2 sans text-[13px]"
                style={{ background: 'var(--surface-card)', borderColor: 'var(--border-medium)', color: 'var(--text-primary)' }}
              />
            </SettingsField>
            <SettingsField label="Base URL">
              <input
                type="text"
                value={baseUrl}
                onChange={e => setBaseUrl(e.target.value)}
                placeholder="https://api.openai.com/v1"
                className="w-full rounded-lg border px-3 py-2 sans text-[13px]"
                style={{ background: 'var(--surface-card)', borderColor: 'var(--border-medium)', color: 'var(--text-primary)' }}
              />
              {provider && LLM_PROVIDER_PRESETS[provider]?.base_url && baseUrl === LLM_PROVIDER_PRESETS[provider].base_url && (
                <div className="sans mt-1" style={{ fontSize: 'calc(10px * var(--folio-font-scale, 1))', color: 'var(--text-secondary)' }}>
                  Pre-filled from {LLM_PROVIDER_PRESETS[provider].label} preset.
                </div>
              )}
            </SettingsField>
            <SettingsField label="Model">
              {presetModels.length > 0 && (
                <div className="flex flex-wrap gap-1.5 mb-2">
                  {presetModels.map((modelName) => (
                    <button
                      key={modelName}
                      type="button"
                      onClick={() => { setModel(modelName); setIsCustomModel(false); }}
                      className="rounded-full border px-2.5 py-1 sans text-[11px] transition-colors"
                      style={{
                        borderColor: model === modelName && !isCustomModel ? 'var(--accent)' : 'var(--border-soft)',
                        background: model === modelName && !isCustomModel ? 'var(--surface-selected)' : 'var(--surface-card)',
                        color: 'var(--text-primary)',
                      }}
                    >
                      {modelName}
                    </button>
                  ))}
                  <button
                    type="button"
                    onClick={() => { setIsCustomModel(true); setModel(''); }}
                    className="rounded-full border px-2.5 py-1 sans text-[11px] transition-colors"
                    style={{
                      borderColor: isCustomModel ? 'var(--accent)' : 'var(--border-soft)',
                      background: isCustomModel ? 'var(--surface-selected)' : 'var(--surface-card)',
                      color: isCustomModel ? 'var(--text-primary)' : 'var(--text-muted)',
                    }}
                  >
                    Custom...
                  </button>
                </div>
              )}
              <input
                type="text"
                value={model}
                onChange={e => { setModel(e.target.value); if (e.target.value) setIsCustomModel(true); }}
                placeholder={presetModels.length > 0 ? 'Or type a custom model name...' : 'Model name (e.g. gpt-4o, claude-sonnet-4-6)'}
                className="w-full rounded-lg border px-3 py-2 sans text-[13px]"
                style={{ background: 'var(--surface-card)', borderColor: 'var(--border-medium)', color: 'var(--text-primary)' }}
              />
            </SettingsField>
            {provider === 'local' && (
              <SettingsNote>
                Local LLM via Ollama, LM Studio, or vLLM. Make sure your local server is running and the model is pulled before testing. No API key is required.
              </SettingsNote>
            )}
            {provider === 'custom' && (
              <SettingsNote>
                Choose the API format above, then enter the matching base URL and model name. OpenAI-compatible endpoints should support /chat/completions.
              </SettingsNote>
            )}
            <div className="flex items-center gap-3 flex-wrap">
              <SettingsButton onClick={handleTest} disabled={testStatus === 'testing' || !baseUrl}>
                {testStatus === 'testing' ? 'Testing...' : 'Test Connection'}
              </SettingsButton>
              <SettingsButton onClick={() => handleSave(false)} disabled={saving} variant="primary">
                {saving ? 'Saving...' : 'Save'}
              </SettingsButton>
              {llmStatus?.configured && (
                <SettingsButton onClick={handleDisable} disabled={saving}>
                  Disable LLM
                </SettingsButton>
              )}
            </div>
            {testStatus && testStatus !== 'testing' && (
              <div
                className="mt-3 p-3 rounded-lg sans text-[12px]"
                style={{
                  background: 'var(--surface-card)',
                  border: `1px solid ${testStatus.ok ? 'var(--ok)' : 'var(--warn)'}`,
                  color: testStatus.ok ? 'var(--ok)' : 'var(--warn)',
                }}
              >
                {testStatus.ok
                  ? `Connected to ${testStatus.model || 'LLM'} (${testStatus.latency_ms}ms)`
                  : testStatus.error || 'Connection failed'}
              </div>
            )}
            </SettingsInnerSection>

            <SettingsInnerSection label="Internet Research" summary={webSummary}>
            <SettingsSubhead detail="Internet research is optional and only runs when the chat composer toggle is on. Web citations appear as [W1], [W2].">
              Web search
            </SettingsSubhead>
            <SettingsField>
              <ToggleControl
                checked={webEnabled}
                onChange={(value) => setWebEnabled(value)}
                label="Enable internet research"
                description="Turns the web-search provider on or off for the chat composer."
              />
            </SettingsField>
            {webEnabled && (
              <SettingsField label="Provider">
                <SegmentedControl
                  value={webProvider}
                  onChange={(value) => { setWebProvider(value); setWebTestStatus(null); }}
                  options={[
                    { value: 'tavily', label: 'Tavily' },
                    { value: 'custom', label: 'Custom' },
                  ]}
                />
              </SettingsField>
            )}
            {webEnabled && webProvider === 'custom' && (
              <SettingsField label="Base URL">
                <input
                  type="text"
                  value={webBaseUrl}
                  onChange={e => setWebBaseUrl(e.target.value)}
                  placeholder="https://search.example.com/api/search"
                  className="w-full rounded-lg border px-3 py-2 sans text-[13px]"
                  style={{ background: 'var(--surface-card)', borderColor: 'var(--border-medium)', color: 'var(--text-primary)' }}
                />
              </SettingsField>
            )}
            {webEnabled && (
              <SettingsField label="API Key">
                <input
                  type="password"
                  value={webApiKey}
                  onChange={e => setWebApiKey(e.target.value)}
                  placeholder={webStatus?.configured ? 'Configured - leave blank to keep current key' : 'API key (optional for custom)...'}
                  className="w-full rounded-lg border px-3 py-2 sans text-[13px]"
                  style={{ background: 'var(--surface-card)', borderColor: 'var(--border-medium)', color: 'var(--text-primary)' }}
                />
              </SettingsField>
            )}
            {webEnabled && webProvider === 'custom' && (
              <SettingsNote>
                Custom endpoints should accept <code>?q=&amp;count=</code> and return <code>{'{"results": [{"title", "url", "content"}]}'}</code> (Tavily-compatible shape). The API key is sent as a Bearer token.
              </SettingsNote>
            )}
            <div className="flex items-center gap-3 flex-wrap">
              {webEnabled && (
                <SettingsButton onClick={handleTestWebSearch} disabled={webTestStatus === 'testing'}>
                  {webTestStatus === 'testing' ? 'Testing...' : 'Test Connection'}
                </SettingsButton>
              )}
              <SettingsButton onClick={() => handleSaveWebSearch(false)} disabled={webSaving} variant="primary">
                {webSaving ? 'Saving...' : 'Save'}
              </SettingsButton>
            </div>
            {webTestStatus && webTestStatus !== 'testing' && (
              <div
                className="mt-3 p-3 rounded-lg sans text-[12px]"
                style={{
                  background: 'var(--surface-card)',
                  border: `1px solid ${webTestStatus.ok ? 'var(--ok)' : 'var(--warn)'}`,
                  color: webTestStatus.ok ? 'var(--ok)' : 'var(--warn)',
                }}
              >
                {webTestStatus.ok
                  ? `Connected to ${webTestStatus.provider || 'web search'} (${webTestStatus.latency_ms}ms)`
                  : webTestStatus.error || 'Connection failed'}
              </div>
            )}
            </SettingsInnerSection>
          </SettingsSection>

          <SettingsSection label="Answer" summary={answerSummary}>
            <SettingsField>
              <ToggleControl
                checked={settings.answer.enterToSend}
                onChange={(value) => update('answer', 'enterToSend', value)}
                label="Enter sends messages"
                description="Use Shift+Enter for a new line when enabled."
              />
            </SettingsField>
            <SettingsField label="Follow-up mode">
              <SegmentedControl
                value={settings.answer.followupMode}
                onChange={(value) => update('answer', 'followupMode', value)}
                options={[
                  { value: 'manual', label: 'Manual' },
                  { value: 'automatic', label: 'Automatic' },
                ]}
              />
            </SettingsField>
            <SettingsField label="Language">
              <SelectControl
                value={settings.answer.language}
                onChange={(value) => update('answer', 'language', value)}
                options={[
                  { value: 'auto', label: 'Auto' },
                  { value: 'Chinese', label: 'Chinese' },
                  { value: 'English', label: 'English' },
                ]}
              />
            </SettingsField>
            <SettingsField label="Answer length">
              <SegmentedControl
                value={settings.answer.length}
                onChange={(value) => update('answer', 'length', value)}
                options={[
                  { value: 'brief', label: 'Brief' },
                  { value: 'normal', label: 'Normal' },
                  { value: 'detailed', label: 'Detailed' },
                ]}
              />
            </SettingsField>
            <SettingsField label="Format">
              <SegmentedControl
                value={settings.answer.format}
                onChange={(value) => update('answer', 'format', value)}
                options={[
                  { value: 'paragraph', label: 'Paragraph' },
                  { value: 'bullets', label: 'Bullets' },
                  { value: 'study-notes', label: 'Study notes' },
                ]}
              />
            </SettingsField>
            <SettingsField label="Citations">
              <SegmentedControl
                value={settings.answer.citations}
                onChange={(value) => update('answer', 'citations', value)}
                options={[
                  { value: 'minimal', label: 'Minimal' },
                  { value: 'normal', label: 'Normal' },
                  { value: 'verbose', label: 'Verbose' },
                ]}
              />
            </SettingsField>
            <SettingsField>
              <ToggleControl
                checked={!!settings.answer.autoSummaryTable}
                onChange={(value) => update('answer', 'autoSummaryTable', value)}
                label="Auto summary table"
                description="Use a compact table when an answer compares conflicts, positions, or concepts."
              />
            </SettingsField>
            <SettingsField>
              <ToggleControl
                checked={settings.answer.showAgentTrace}
                onChange={(value) => update('answer', 'showAgentTrace', value)}
                label="Show agent trace"
                description="Expose planning and tool trace details when answers include them."
              />
            </SettingsField>
          </SettingsSection>

          <SettingsSection label="Studio" summary={studioSummary}>
            <SettingsSubhead>Summaries</SettingsSubhead>
            <SettingsField label="Length">
              <SegmentedControl
                value={settings.studio.summaryLength}
                onChange={(value) => update('studio', 'summaryLength', value)}
                options={[
                  { value: 'short', label: 'Short' },
                  { value: 'medium', label: 'Medium' },
                  { value: 'detailed', label: 'Detailed' },
                ]}
              />
            </SettingsField>
            <SettingsField label="Style">
              <SegmentedControl
                value={settings.studio.summaryStyle}
                onChange={(value) => update('studio', 'summaryStyle', value)}
                options={[
                  { value: 'prose', label: 'Prose' },
                  { value: 'bullets', label: 'Bullets' },
                  { value: 'structured', label: 'Structured' },
                ]}
              />
            </SettingsField>

            <SettingsSubhead>Practice</SettingsSubhead>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <SettingsField label="Flashcards">
                <NumberControl
                  value={settings.studio.flashcardCount}
                  onChange={(value) => update('studio', 'flashcardCount', value)}
                  min={10}
                  max={50}
                  step={5}
                />
              </SettingsField>
              <SettingsField label="Quiz questions">
                <NumberControl
                  value={settings.studio.quizCount}
                  onChange={(value) => update('studio', 'quizCount', value)}
                  min={5}
                  max={30}
                  step={5}
                />
              </SettingsField>
            </div>
            <SettingsField label="Quiz difficulty">
              <SegmentedControl
                value={settings.studio.quizDifficulty}
                onChange={(value) => update('studio', 'quizDifficulty', value)}
                options={[
                  { value: 'basic', label: 'Basic' },
                  { value: 'medium', label: 'Medium' },
                  { value: 'hard', label: 'Hard' },
                ]}
              />
            </SettingsField>

            <SettingsField>
              <ToggleControl
                checked={settings.studio.autoExpandGenerated}
                onChange={(value) => update('studio', 'autoExpandGenerated', value)}
                label="Auto-expand generated items"
                description="Open new Studio artifacts when generation completes."
              />
            </SettingsField>
          </SettingsSection>

          <SettingsSection label="Sources" summary={sourcesSummary}>
            <SettingsField label="Default scope">
              <SegmentedControl
                value={settings.sources.openScope}
                onChange={(value) => update('sources', 'openScope', value)}
                options={[
                  { value: 'last-or-all', label: 'Last or all' },
                  { value: 'all', label: 'All' },
                  { value: 'none', label: 'None' },
                ]}
              />
            </SettingsField>
            <SettingsField>
              <ToggleControl
                checked={settings.sources.autoIncludeUploads}
                onChange={(value) => update('sources', 'autoIncludeUploads', value)}
                label="Auto-include uploads"
                description="Newly indexed sources become part of the active source set."
              />
            </SettingsField>
            <SettingsField label="Sort sources">
              <SegmentedControl
                value={settings.sources.sort}
                onChange={(value) => update('sources', 'sort', value)}
                options={[
                  { value: 'recent', label: 'Recent' },
                  { value: 'title', label: 'Title' },
                  { value: 'status', label: 'Status' },
                ]}
              />
            </SettingsField>
            <div className="flex items-center gap-2 flex-wrap">
              <SettingsButton onClick={handleResetSourceScope}>Reset source scope</SettingsButton>
            </div>
          </SettingsSection>
        </div>

        <div className="flex items-center justify-between gap-3 px-6 py-4 border-t border-line2">
          <button
            type="button"
            onClick={resetAllSettings}
            className="sans text-[12px]"
            style={{ color: 'var(--text-secondary)' }}
          >
            Reset preference defaults
          </button>
          <SettingsButton onClick={onClose} variant="primary">Done</SettingsButton>
        </div>
      </div>
    </div>
  );
};

window.SettingsModal = SettingsModal;
