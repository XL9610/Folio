// Notebook chat state: messages, draft, localStorage cache, and backend
// autosave of the notebook's conversation. Moved out of app.jsx unchanged.
const { useState, useEffect } = React;

const useChat = (notebookId, includedIds) => {
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
  }, [chat, notebookId, includedIds, conversationId]);

  return { chat, setChat, chatDraft, setChatDraft, clearChat, CHAT_SCROLL_KEY };
};

window.useChat = useChat;
