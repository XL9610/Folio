// Notebook chat state: the notebook's sessions, the active session's messages
// and draft, a localStorage cache of the active session, and backend autosave.
// A session is one saved conversation row; a new one is unsaved until it has a
// message.
const { useState, useEffect, useRef } = React;

// Title = first question, truncated. Titles saved before sessions existed
// ("Chat - 5/20/2026") carried only a date, so they fall through to it too.
const sessionTitle = (session) => {
  const title = (session?.title || '').trim();
  if (title && !/^Chat - /.test(title)) return title;
  const first = (session?.messages || []).find(m => m.role === 'user')?.text || '';
  const line = first.replace(/\s+/g, ' ').trim();
  if (!line) return 'New chat';
  return line.length > 60 ? line.slice(0, 57) + '...' : line;
};

const chatSig = (messages) => messages.map(m => m.id || '').join('|');

const useChat = (notebookId, includedIds, restoreSources) => {
  // Active-session cache (localStorage, last 50 messages)
  const CHAT_STORAGE_KEY = 'folio.chat.' + notebookId + '.v1';
  const CHAT_SCROLL_KEY = 'folio.chatScroll.' + notebookId + '.v1';
  const SESSION_KEY = 'folio.session.' + notebookId + '.v1';

  const loadStoredChat = () => {
    try {
      const raw = localStorage.getItem(CHAT_STORAGE_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      // A streaming placeholder left by a closed window is not a real message.
      return Array.isArray(parsed) ? parsed.filter(m => !m._streaming) : [];
    } catch (e) {
      console.warn('Failed to load stored chat', e);
      return [];
    }
  };
  const loadStoredSessionId = () => {
    try { return localStorage.getItem(SESSION_KEY) || ''; } catch (e) { return ''; }
  };

  const [chat, setChat] = useState(() => loadStoredChat());
  const [chatDraft, setChatDraft] = useState('');
  const [sessions, setSessions] = useState([]);
  const [conversationId, setConversationId] = useState(() => loadStoredSessionId());
  const [lastSavedSig, setLastSavedSig] = useState('');

  // Latest values for the async session actions below.
  const live = useRef({});
  live.current = { chat, sessions, conversationId, lastSavedSig, includedIds, restoreSources };

  // Persist the active session's chat and id to localStorage on every change
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

  useEffect(() => {
    try {
      if (conversationId) localStorage.setItem(SESSION_KEY, conversationId);
      else localStorage.removeItem(SESSION_KEY);
    } catch (e) {}
  }, [conversationId]);

  const clearChat = () => {
    setChat([]);
    try { localStorage.removeItem(CHAT_STORAGE_KEY); } catch (e) {}
  };

  // Load the notebook's sessions on entry and reopen the one last active.
  useEffect(() => {
    if (!notebookId) return;
    (async () => {
      try {
        const data = await window.FOLIO_API.listConversations(notebookId);
        const list = data?.conversations || data || [];
        setSessions(list);
        const stored = loadStoredSessionId();
        // No stored id: first run with sessions, where the cached chat is the
        // most recent conversation. A stored id that is not listed is a new
        // session not saved yet; leave it alone.
        const active = list.find(s => s.id === stored) || (stored ? null : list[0]);
        if (!active) return;
        const msgs = active.messages || [];
        const savedIds = new Set(msgs.map(m => m.id));
        setConversationId(active.id);
        // The local cache holds only the last 50 messages, so it must never
        // replace the saved history: keep that, plus anything not saved yet.
        setChat(prev => [...msgs, ...prev.filter(m => !savedIds.has(m.id))]);
        setLastSavedSig(chatSig(msgs));
      } catch (e) { console.warn('Failed to load conversations:', e); }
    })();
  }, [notebookId]);

  // Save the active session if it has unsaved messages. Saves upsert by id
  // (the backend assigns one when a notebook's very first chat has none).
  const saveNow = async () => {
    const { chat, sessions, conversationId, lastSavedSig, includedIds } = live.current;
    const messages = chat.filter(m => !m._streaming);
    const sig = chatSig(messages);
    if (messages.length === 0 || sig === lastSavedSig) return;
    const known = sessions.find(s => s.id === conversationId);
    const saved = await window.FOLIO_API.saveConversation(
      sessionTitle({ title: known?.title, messages }),
      messages, // the whole history: a truncated save would overwrite the rest
      includedIds,
      notebookId,
      conversationId
    );
    if (!saved || !saved.id) return;
    setLastSavedSig(sig);
    setSessions(prev => [saved, ...prev.filter(s => s.id !== saved.id)]);
    // Only adopt the id if the user has not moved to another session meanwhile.
    if (live.current.conversationId === conversationId) setConversationId(saved.id);
  };

  // Autosave shortly after the chat settles (a finished answer, a delete).
  useEffect(() => {
    const timer = setTimeout(() => {
      saveNow().catch(e => console.warn('Auto-save failed:', e));
    }, 2000);
    return () => clearTimeout(timer);
  }, [chat, notebookId, lastSavedSig]);

  // Save chat on beforeunload
  useEffect(() => {
    const handler = () => {
      const messages = chat.filter(m => !m._streaming);
      if (messages.length > 0) {
        const known = sessions.find(s => s.id === conversationId);
        const payload = JSON.stringify({
          id: conversationId,
          notebook_id: notebookId,
          title: sessionTitle({ title: known?.title, messages }),
          messages,
          source_document_ids: includedIds,
        });
        // sendBeacon can't carry X-Auth-Token; use the token-injecting fetch
        // wrapper with keepalive so the save still lands during unload.
        fetch(`${window.FOLIO_API_BASE}/api/conversations`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: payload,
          keepalive: true,
        });
      }
    };
    window.addEventListener('beforeunload', handler);
    return () => window.removeEventListener('beforeunload', handler);
  }, [chat, sessions, notebookId, includedIds, conversationId]);

  // Show a session's messages and restore the sources it was using.
  const openSession = (session) => {
    const msgs = session?.messages || [];
    // A new chat gets its id up front, so a reload can tell it apart from the
    // saved sessions and its first save cannot land twice.
    setConversationId(session?.id || 's' + Date.now());
    setChat(msgs);
    setLastSavedSig(chatSig(msgs));
    setChatDraft('');
    if (session) live.current.restoreSources?.(session.source_document_ids || []);
  };

  const saveQuietly = async () => {
    try { await saveNow(); } catch (e) { console.warn('Save before switching failed:', e); }
  };

  const newSession = async () => {
    await saveQuietly();
    openSession(null);
  };

  const switchSession = async (id) => {
    if (id === live.current.conversationId) return;
    await saveQuietly();
    const target = live.current.sessions.find(s => s.id === id);
    if (target) openSession(target);
  };

  const renameSession = async (id, title) => {
    const name = (title || '').trim();
    const { chat, sessions, conversationId, includedIds } = live.current;
    const target = sessions.find(s => s.id === id);
    if (!name || !target) return;
    const isActive = id === conversationId;
    const saved = await window.FOLIO_API.saveConversation(
      name,
      isActive ? chat.filter(m => !m._streaming) : target.messages,
      isActive ? includedIds : target.source_document_ids,
      notebookId,
      id
    );
    setSessions(prev => prev.map(s => (s.id === id ? saved : s)));
  };

  const deleteSession = async (id) => {
    await window.FOLIO_API.deleteConversation(id);
    const rest = live.current.sessions.filter(s => s.id !== id);
    setSessions(rest);
    if (id === live.current.conversationId) openSession(rest[0] || null);
  };

  // What the sidebar lists: saved sessions, with the active one on top while
  // it is still unsaved (a new chat, or one whose first save is pending).
  const listed = sessions.map(s => ({
    id: s.id,
    title: sessionTitle(s.id === conversationId ? { title: s.title, messages: chat } : s),
  }));
  const sessionList = sessions.some(s => s.id === conversationId)
    ? listed
    : [{ id: conversationId, title: sessionTitle({ messages: chat }), unsaved: true }, ...listed];

  return {
    chat, setChat, chatDraft, setChatDraft, clearChat, CHAT_SCROLL_KEY,
    sessionList, activeSessionId: conversationId,
    newSession, switchSession, renameSession, deleteSession,
  };
};

window.useChat = useChat;
window.folioSessionTitle = sessionTitle;
