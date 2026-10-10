// Browser-global Folio preferences store.
(function installFolioSettingsStore() {
  if (window.getSettings && window.setSettings && window.subscribeSettings && window.resetSettings) return;

  const STORAGE_KEY = 'folio.settings.v1';
  const DEFAULTS = {
    ai: {
      internetDefault: false,
    },
    _llm: {
      provider: '',
      api_format: '',
      base_url: '',
      model: '',
      has_api_key: false,
      configured: false,
    },
    _webSearch: {
      provider: '',
      configured: false,
    },
    answer: {
      enterToSend: true,
      followupMode: 'manual',
      language: 'auto',
      length: 'normal',
      format: 'paragraph',
      citations: 'normal',
      autoSummaryTable: false,
      showAgentTrace: false,
    },
    studio: {
      summaryLength: 'medium',
      summaryStyle: 'structured',
      flashcardCount: 20,
      quizCount: 10,
      quizDifficulty: 'medium',
      autoExpandGenerated: true,
    },
    sources: {
      openScope: 'last-or-all',
      autoIncludeUploads: true,
      sort: 'recent',
    },
    layout: {
      fontScale: 'normal',
      readingWidth: 'normal',
    },
  };

  const listeners = new Set();
  const isPlainObject = (value) =>
    !!value && typeof value === 'object' && !Array.isArray(value);
  const clone = (value) => JSON.parse(JSON.stringify(value));

  const deepMerge = (base, patch) => {
    const result = clone(base || {});
    if (!isPlainObject(patch)) return result;
    Object.keys(patch).forEach((key) => {
      const next = patch[key];
      if (typeof next === 'undefined') return;
      if (isPlainObject(next)) {
        result[key] = deepMerge(isPlainObject(result[key]) ? result[key] : {}, next);
      } else {
        result[key] = clone(next);
      }
    });
    return result;
  };

  const normalizeAnswerLanguage = (value) => {
    const raw = typeof value === 'string' ? value.trim() : '';
    const key = raw.toLowerCase().replace(/_/g, '-');
    if (!raw || ['auto', 'default', 'system'].includes(key)) return 'auto';
    if (['zh', 'zh-cn', 'zh-hans', 'chinese', 'simplified-chinese'].includes(key)) return 'Chinese';
    if (['en', 'en-us', 'english'].includes(key)) return 'English';
    return raw;
  };

  const sanitizeSettings = (settings) => {
    const next = deepMerge(DEFAULTS, settings);
    if (next.layout) {
      delete next.layout.reducedMotion;
      delete next.layout.highContrast;
    }
    if (next.answer) {
      next.answer.language = normalizeAnswerLanguage(next.answer.language);
    }
    return next;
  };

  const readStored = () => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return clone(DEFAULTS);
      return sanitizeSettings(JSON.parse(raw));
    } catch (e) {
      return clone(DEFAULTS);
    }
  };

  const writeStored = (settings) => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(sanitizeSettings(settings)));
    } catch (e) {}
  };

  let current = readStored();

  const notify = () => {
    const snapshot = clone(current);
    listeners.forEach((listener) => {
      try { listener(snapshot); } catch (e) {}
    });
    try {
      window.dispatchEvent(new CustomEvent('folio:settings-changed', { detail: snapshot }));
    } catch (e) {}
  };

  const getSettings = () => {
    current = readStored();
    return clone(current);
  };

  const setSettings = (patch) => {
    current = sanitizeSettings(deepMerge(current, patch));
    writeStored(current);
    notify();
    return clone(current);
  };

  const subscribeSettings = (listener) => {
    if (typeof listener !== 'function') return () => {};
    listeners.add(listener);
    return () => listeners.delete(listener);
  };

  const resetSettings = () => {
    current = clone(DEFAULTS);
    writeStored(current);
    notify();
    return clone(current);
  };

  window.addEventListener('storage', (event) => {
    if (event.key !== STORAGE_KEY) return;
    current = readStored();
    notify();
  });

  // ── Pull backend LLM / web-search config into settings._llm / ._webSearch ──
  const API_BASE = 'http://127.0.0.1:8000';
  fetch(`${API_BASE}/api/config/llm-status`)
    .then((r) => r.json())
    .then((cfg) => {
      current = deepMerge(current, { _llm: cfg });
      writeStored(current);
      notify();
    })
    .catch(() => {}); // backend not available — silently ignore

  fetch(`${API_BASE}/api/config/web-search-status`)
    .then((r) => r.json())
    .then((cfg) => {
      current = deepMerge(current, { _webSearch: cfg });
      writeStored(current);
      notify();
    })
    .catch(() => {});

  Object.assign(window, {
    FOLIO_SETTINGS_STORAGE_KEY: STORAGE_KEY,
    FOLIO_SETTINGS_DEFAULTS: clone(DEFAULTS),
    FOLIO_SETTINGS: {
      key: STORAGE_KEY,
      defaults: clone(DEFAULTS),
      getSettings,
      setSettings,
      subscribeSettings,
      resetSettings,
    },
    getSettings,
    setSettings,
    subscribeSettings,
    resetSettings,
  });
})();
