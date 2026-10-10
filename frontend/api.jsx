// API client for the RAG Study Companion backend.
// Runs in the browser alongside the JSX demo.
// Exposed as window.FOLIO_API; loaded after data.jsx, before app.jsx.

const API_BASE = 'http://127.0.0.1:8000';
window.FOLIO_API_BASE = API_BASE;

// Per-launch API token, injected by the Electron main process via preload. When
// present, every request carries it so the backend (which enforces the token
// when FOLIO_API_TOKEN is set) rejects foreign web pages. Empty in a bare dev
// browser, where the backend does not enforce it.
const API_TOKEN = (typeof window !== 'undefined' && window.electronAPI && window.electronAPI.apiToken) || '';
const _nativeFetch = window.fetch.bind(window);
window.fetch = (url, options = {}) => {
  if (!API_TOKEN) return _nativeFetch(url, options);
  const headers = new Headers(options.headers || {});
  if (!headers.has('X-Auth-Token')) headers.set('X-Auth-Token', API_TOKEN);
  return _nativeFetch(url, { ...options, headers });
};

// Shared title cleaner — mirrors backend _title_utils.clean_filename
const _cleanTitle = (title, kind) => {
  if (!title) return 'Untitled';
  let s = String(title);
  s = s.replace(/^(Flashcards|Quiz|Summary):\s*/i, '');
  s = s.replace(/_z[_-]?library[_-]?sk[_,.\s]*/gi, '');
  s = s.replace(/_1lib[_-]?sk[_,.\s]*/gi, '');
  s = s.replace(/_?libgen[_-]?[a-z]*[_,.\s]*/gi, '');
  s = s.replace(/\(z[_-]?lib[^)]*\)/gi, '');
  s = s.replace(/\[z[_-]?lib[^\]]*\]/gi, '');
  s = s.replace(/\.(pdf|epub|mobi|djvu|txt|docx?)(?=\s|—|—|,|$)/gi, '');
  s = s.replace(/_/g, ' ');
  s = s.replace(/^[,;\-\s]+/, '').replace(/[,;\-\s]+$/, '');
  s = s.replace(/\s{2,}/g, ' ').trim();
  return s || 'Untitled';
};
window._cleanDisplayTitle = _cleanTitle;

const _safeJsonParse = (raw) => {
  if (!raw || typeof raw !== 'string') return null;
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed : null;
  } catch (e) {
    return null;
  }
};

const _readLocalObject = (key) => {
  try {
    return _safeJsonParse(window.localStorage?.getItem(key));
  } catch (e) {
    return null;
  }
};

const _answerSource = (source) => {
  if (!source || typeof source !== 'object') return null;
  return source.answerPreferences
    || source.answer_preferences
    || source.answer
    || source.preferences?.answer
    || source;
};

const _firstStringPreference = (sources, keys) => {
  for (const source of sources) {
    if (!source || typeof source !== 'object') continue;
    for (const key of keys) {
      const value = source[key];
      if (typeof value === 'string' && value.trim()) {
        return value.trim();
      }
    }
  }
  return '';
};

const _normalizeLanguagePreference = (value) => {
  const raw = typeof value === 'string' ? value.trim() : '';
  const key = raw.toLowerCase().replace(/_/g, '-');
  if (!raw || ['auto', 'default', 'system'].includes(key)) return '';
  if (['zh', 'zh-cn', 'zh-hans', 'chinese', 'simplified-chinese'].includes(key)) return 'Chinese';
  if (['en', 'en-us', 'english'].includes(key)) return 'English';
  return raw;
};

const _normalizeAnswerPreferences = (explicitPreferences = null) => {
  const globalPrefs = typeof window !== 'undefined'
    ? [
        _answerSource(window.FOLIO_ANSWER_PREFERENCES),
        _answerSource(window.FOLIO_PREFERENCES?.answer),
        _answerSource(window.FOLIO_PREFERENCES),
        _answerSource(window.FOLIO_SETTINGS?.answerPreferences),
        _answerSource(window.FOLIO_SETTINGS?.answer),
        _answerSource(window.FOLIO_SETTINGS?.getSettings?.()?.answer),
        _answerSource(_readLocalObject('folio.answerPreferences')),
        _answerSource(_readLocalObject('folio.answer.preferences')),
        _answerSource(_readLocalObject('folio.preferences')?.answer),
        _answerSource(_readLocalObject('folio.settings')?.answer),
        _answerSource(_readLocalObject('folio.settings.v1')?.answer),
      ]
    : [];
  const sources = [_answerSource(explicitPreferences), ...globalPrefs];
  const preferences = {};

  const style = _firstStringPreference(sources, ['style', 'answer_style', 'answerStyle']);
  const length = _firstStringPreference(sources, ['length', 'answer_length', 'answerLength']);
  const detailLevel = _firstStringPreference(sources, ['detail_level', 'detailLevel']);
  const tone = _firstStringPreference(sources, ['tone']);
  const format = _firstStringPreference(sources, ['format']);
  const audience = _firstStringPreference(sources, ['audience']);
  const language = _firstStringPreference(sources, ['language']);
  const citations = _firstStringPreference(sources, ['citations', 'citation_density', 'citationDensity']);
  const customInstructions = _firstStringPreference(sources, [
    'custom_instructions',
    'customInstructions',
    'instructions',
  ]);

  if (style) preferences.style = style;
  if (length) preferences.length = length;
  if (detailLevel) preferences.detail_level = detailLevel;
  if (tone) preferences.tone = tone;
  if (format) preferences.format = format;
  if (audience) preferences.audience = audience;
  const normalizedLanguage = _normalizeLanguagePreference(language);
  if (normalizedLanguage) preferences.language = normalizedLanguage;
  if (citations) preferences.citations = citations;
  if (customInstructions) preferences.custom_instructions = customInstructions;
  const autoSummarySource = sources.find(source => source && typeof source === 'object' && (
    Object.prototype.hasOwnProperty.call(source, 'autoSummaryTable')
    || Object.prototype.hasOwnProperty.call(source, 'auto_summary_table')
  ));
  if (autoSummarySource) {
    preferences.auto_summary_table = !!(autoSummarySource.autoSummaryTable ?? autoSummarySource.auto_summary_table);
  }

  return Object.keys(preferences).length ? preferences : null;
};

const FOLIO_API = {

  // ── Documents ─────────────────────────────────────────────────────────────

  /**
   * List all documents in a notebook.
   * GET /api/documents?notebook_id=default
   * Returns array of { document_id, filename, page_count, chunk_count, status, created_at }
   */
  async listDocuments(notebookId = 'default') {
    const res = await fetch(`${API_BASE}/api/documents?notebook_id=${encodeURIComponent(notebookId)}`);
    if (!res.ok) {
      throw new Error(`Failed to list documents: ${res.status}`);
    }
    return res.json();
  },

  /**
   * Upload a file for ingestion (pdf, epub, pptx, docx, md, txt).
   * POST /api/documents/ingest?notebook_id=default&force_reingest=false
   * Body: multipart/form-data with `file` field.
   * Returns { document_id, notebook_id, filename, page_count, chunk_count, status, ... }
   */
  async uploadDocument(file, notebookId = 'default') {
    const form = new FormData();
    form.append('file', file);

    const res = await fetch(
      `${API_BASE}/api/documents/ingest?notebook_id=${encodeURIComponent(notebookId)}&force_reingest=false`,
      { method: 'POST', body: form }
    );

    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `Upload failed: ${res.status}`);
    }
    return res.json();
  },

  /**
   * Poll an async ingestion job by id.
   * GET /api/jobs/{job_id}
   */
  async getJob(jobId) {
    const res = await fetch(`${API_BASE}/api/jobs/${encodeURIComponent(jobId)}`);
    if (!res.ok) throw new Error(`Get job failed: ${res.status}`);
    return res.json();
  },

  /**
   * List ingestion jobs for a notebook.
   * GET /api/jobs?notebook_id=default
   */
  async listJobs(notebookId = 'default') {
    const res = await fetch(`${API_BASE}/api/jobs?notebook_id=${encodeURIComponent(notebookId)}`);
    if (!res.ok) throw new Error(`List jobs failed: ${res.status}`);
    return res.json();
  },

  /**
   * Delete a document and all its chunks (and dense vectors).
   * DELETE /api/documents/{document_id}
   */
  async deleteDocument(documentId) {
    const res = await fetch(`${API_BASE}/api/documents/${encodeURIComponent(documentId)}`, {
      method: 'DELETE',
    });
    if (!res.ok) {
      throw new Error(`Failed to delete document: ${res.status}`);
    }
    return res.json();
  },

  /**
   * Get document detail including section outline.
   * GET /api/documents/{document_id}
   * Returns { document_id, filename, page_count, chunk_count, status, sections }
   */
  async getDocument(documentId) {
    const res = await fetch(`${API_BASE}/api/documents/${encodeURIComponent(documentId)}`);
    if (!res.ok) {
      throw new Error(`Failed to get document: ${res.status}`);
    }
    return res.json();
  },

  // ── Search ─────────────────────────────────────────────────────────────────

  /**
   * Search across ingested documents using hybrid retrieval.
   * POST /api/search
   * Body: { notebook_id, query, top_k }
   * Returns { notebook_id, query, results: [{ chunk_id, document_id, text, title, section_path, page_start, page_end, dense_score, fts_score, rrf_score, citation }] }
   */
  async search(query, topK = 10, notebookId = 'default', documentIds = null) {
    const body = {
      notebook_id: notebookId,
      query: query,
      top_k: topK,
    };
    if (documentIds && documentIds.length > 0) {
      body.document_ids = documentIds;
    }
    const res = await fetch(`${API_BASE}/api/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      throw new Error(`Search failed: ${res.status}`);
    }
    return res.json();
  },

  // ── Health ─────────────────────────────────────────────────────────────────

  /** Quick health check. */
  async health() {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error(`Backend not healthy: ${res.status}`);
    return res.json();
  },

  // ── Ask (Agentic RAG) ──────────────────────────────────────────────────────

  /**
   * Ask a question — LLM-required agentic QA: planner → tools → generation.
   * POST /api/ask
   * Body: { notebook_id, query, top_k, document_ids?, history?, allow_web?, preferences? }
   * Returns { notebook_id, query, answer, agent_status, planner_task, used_llm, citations, tool_trace }
   */
  async ask(query, notebookId = 'default', documentIds = null, topK = 10, history = [], allowWeb = false, preferences = null) {
    let followUp = false;
    if (allowWeb && typeof allowWeb === 'object') {
      const options = allowWeb;
      allowWeb = !!(options.allowWeb ?? options.allow_web);
      preferences = options.preferences || options.answerPreferences || options.answer_preferences || preferences;
      followUp = !!(options.followUp ?? options.follow_up);
    }
    const answerPreferences = _normalizeAnswerPreferences(preferences);
    const body = {
      notebook_id: notebookId,
      query: query,
      top_k: topK,
      history: history,
      allow_web: !!allowWeb,
    };
    if (documentIds && documentIds.length > 0) {
      body.document_ids = documentIds;
    }
    if (answerPreferences) {
      body.preferences = answerPreferences;
    }
    if (followUp) {
      body.follow_up = true;
    }
    const res = await fetch(`${API_BASE}/api/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      const detail = body.detail;  // backend FastAPI HTTPException detail
      // Build a readable message string
      let msg;
      if (typeof detail === 'object' && detail !== null && detail.message) {
        msg = detail.message;
      } else if (typeof detail === 'string') {
        msg = detail;
      } else {
        msg = `Ask failed: ${res.status}`;
      }
      // Preserve the structured detail so the caller can inspect code/message
      const err = new Error(msg);
      err.detail = detail || { code: `http_${res.status}`, message: msg };
      err.status = res.status;
      throw err;
    }
    return res.json();
  },

  // ── Generate (Studio) ──────────────────────────────────────────────────────

  /**
   * Generate a study artifact (summary, quiz, etc.).
   * POST /api/generate
   * Body: { kind, notebook_id, document_ids, params: { scope, length, style } }
   * Returns { artifact_id, kind, title, markdown, stats, sources, source_document_ids, params }
   */
  async generate(kind, documentIds, params = {}, notebookId = 'default') {
    const res = await fetch(`${API_BASE}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        kind,
        notebook_id: notebookId,
        document_ids: documentIds,
        params,
      }),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      const msg = detail.detail?.message || detail.detail || `Generate failed: ${res.status}`;
      const err = new Error(msg);
      err.detail = detail.detail || { code: `http_${res.status}`, message: msg };
      err.status = res.status;
      throw err;
    }
    return res.json();
  },

  // ── Artifacts (Saved items) ─────────────────────────────────────────────────

  /**
   * List saved artifacts.
   * GET /api/artifacts?notebook_id=default&kind=summary
   */
  async listArtifacts(notebookId = 'default', kind = null) {
    let url = `${API_BASE}/api/artifacts?notebook_id=${encodeURIComponent(notebookId)}`;
    if (kind) url += `&kind=${encodeURIComponent(kind)}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to list artifacts: ${res.status}`);
    const data = await res.json();
    // Normalize titles on all returned artifacts
    if (data.artifacts) {
      data.artifacts = data.artifacts.map(a => ({
        ...a,
        title: _cleanTitle(a.title, a.kind),
      }));
    }
    return data;
  },

  /**
   * Get a single artifact by ID.
   * GET /api/artifacts/{artifact_id}
   */
  async getArtifact(artifactId) {
    const res = await fetch(`${API_BASE}/api/artifacts/${encodeURIComponent(artifactId)}`);
    if (!res.ok) throw new Error(`Failed to get artifact: ${res.status}`);
    const data = await res.json();
    if (data.title) data.title = _cleanTitle(data.title, data.kind);
    return data;
  },

  /**
   * Delete an artifact by ID.
   * DELETE /api/artifacts/{artifact_id}
   */
  async deleteArtifact(artifactId) {
    const res = await fetch(`${API_BASE}/api/artifacts/${encodeURIComponent(artifactId)}`, {
      method: 'DELETE',
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      const detail = body.detail;
      const msg = typeof detail === 'object' && detail?.message
        ? detail.message
        : (typeof detail === 'string' ? detail : `Failed to delete artifact: ${res.status}`);
      const err = new Error(msg);
      err.status = res.status;
      err.detail = detail || { code: `http_${res.status}`, message: msg };
      throw err;
    }
    return res.json();
  },

  // ── Conversations ─────────────────────────────────────────────────────────

  async listReviewItems(notebookId = 'default', filter = 'all') {
    const url = `${API_BASE}/api/review-items?notebook_id=${encodeURIComponent(notebookId)}&filter=${encodeURIComponent(filter || 'all')}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to list review items: ${res.status}`);
    return res.json();
  },

  async saveReviewItem(payload = {}) {
    const res = await fetch(`${API_BASE}/api/review-items`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      const msg = detail.detail?.message || detail.detail || `Save review item failed: ${res.status}`;
      const err = new Error(msg);
      err.detail = detail.detail || { code: `http_${res.status}`, message: msg };
      err.status = res.status;
      throw err;
    }
    return res.json();
  },

  async deleteReviewItem(itemId) {
    const res = await fetch(`${API_BASE}/api/review-items/${encodeURIComponent(itemId)}`, {
      method: 'DELETE',
    });
    if (!res.ok) throw new Error(`Failed to delete review item: ${res.status}`);
    return res.json();
  },

  /**
   * Create a saved artifact, used for user-saved chat notes.
   * POST /api/artifacts
   */
  async createArtifact(kind, title, contentJson = {}, sourceDocumentIds = [], paramsJson = {}, notebookId = 'default') {
    const res = await fetch(`${API_BASE}/api/artifacts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        notebook_id: notebookId,
        kind,
        title,
        content_json: contentJson,
        source_document_ids: sourceDocumentIds,
        params_json: paramsJson,
      }),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `Create artifact failed: ${res.status}`);
    }
    const data = await res.json();
    if (data.title) data.title = _cleanTitle(data.title, data.kind);
    return data;
  },

  /**
   * Save a chat conversation.
   * POST /api/conversations
   */
  async saveConversation(title, messages, sourceDocumentIds = [], notebookId = 'default', conversationId = '') {
    const res = await fetch(`${API_BASE}/api/conversations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        id: conversationId,
        notebook_id: notebookId,
        title,
        messages,
        source_document_ids: sourceDocumentIds,
      }),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `Save failed: ${res.status}`);
    }
    return res.json();
  },

  /**
   * List saved conversations.
   * GET /api/conversations?notebook_id=default
   */
  async listConversations(notebookId = 'default') {
    const res = await fetch(`${API_BASE}/api/conversations?notebook_id=${encodeURIComponent(notebookId)}`);
    if (!res.ok) throw new Error(`List conversations failed: ${res.status}`);
    return res.json();
  },

  // ── Notebooks ────────────────────────────────────────────────────────────

  async listNotebooks() {
    const res = await fetch(`${API_BASE}/api/notebooks`);
    if (!res.ok) throw new Error(`List notebooks failed: ${res.status}`);
    return res.json();
  },

  async createNotebook(name, description = '') {
    const res = await fetch(`${API_BASE}/api/notebooks`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, description }),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `Create notebook failed: ${res.status}`);
    }
    return res.json();
  },

  async getNotebook(notebookId) {
    const res = await fetch(`${API_BASE}/api/notebooks/${encodeURIComponent(notebookId)}`);
    if (!res.ok) {
      if (res.status === 404) return null;
      throw new Error(`Get notebook failed: ${res.status}`);
    }
    return res.json();
  },

  async updateNotebook(notebookId, data) {
    const res = await fetch(`${API_BASE}/api/notebooks/${encodeURIComponent(notebookId)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `Update notebook failed: ${res.status}`);
    }
    return res.json();
  },

  async deleteNotebook(notebookId) {
    const res = await fetch(`${API_BASE}/api/notebooks/${encodeURIComponent(notebookId)}`, {
      method: 'DELETE',
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `Delete notebook failed: ${res.status}`);
    }
    return true;
  },

  async touchNotebook(notebookId) {
    const res = await fetch(`${API_BASE}/api/notebooks/${encodeURIComponent(notebookId)}/opened`, {
      method: 'PUT',
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `Mark opened failed: ${res.status}`);
    }
    return res.json();
  },

  async embeddingStatus() {
    const res = await fetch(`${API_BASE}/api/config/embedding-status`);
    if (!res.ok) throw new Error(`Embedding status failed: ${res.status}`);
    return res.json();
  },
};

window.FOLIO_API = FOLIO_API;
