const { test, expect } = require('@playwright/test');

test('chat sessions: reopen, switch, new, autosave, rename, delete', async ({ page }) => {
  await page.goto('/index.html');
  await page.evaluate(() => {
    window.__saves = [];
    window.__deleted = [];
    window.__restored = null;
    const stored = [
      // a title saved before sessions existed: shown as its first question
      { id: 's2', title: 'Chat - 1/1/2026', source_document_ids: ['d2'],
        messages: [{ id: 'b1', role: 'user', text: 'Second chat question' }] },
      { id: 's1', title: 'Renamed one', source_document_ids: ['d1'],
        messages: [{ id: 'a1', role: 'user', text: 'First chat question' }] },
    ];
    window.FOLIO_API.listConversations = async () => stored;
    window.FOLIO_API.saveConversation = async (title, messages, source_document_ids, notebookId, id) => {
      window.__saves.push({ title, id, count: messages.length });
      return { id, title, messages, source_document_ids };
    };
    window.FOLIO_API.deleteConversation = async (id) => { window.__deleted.push(id); };

    const Harness = () => {
      const c = window.useChat('nb-sessions-test', ['d9'], ids => { window.__restored = ids; });
      return React.createElement('div', null,
        React.createElement(window.SessionList, {
          list: c.sessionList, activeId: c.activeSessionId, busy: false,
          onNew: c.newSession, onSwitch: c.switchSession, onRename: c.renameSession, onDelete: c.deleteSession,
        }),
        React.createElement('div', { id: 'msgs' }, c.chat.map(m => m.text).join(' / ')),
        React.createElement('button', {
          onClick: () => c.setChat(prev => [...prev, { id: 'n1', role: 'user', text: 'A brand new question' }]),
        }, 'add message'));
    };
    const host = document.createElement('div');
    host.id = 'sessions-root';
    document.body.appendChild(host);
    ReactDOM.createRoot(host).render(React.createElement(Harness));
  });
  const root = page.locator('#sessions-root');
  const row = (name) => root.getByRole('button', { name, exact: true });

  // opens the most recent session; both are listed
  await expect(row('Second chat question')).toHaveAttribute('aria-current', 'true');
  await expect(root.locator('#msgs')).toHaveText('Second chat question');

  // switch: other history, and that session's sources come back
  await row('Renamed one').click();
  await expect(root.locator('#msgs')).toHaveText('First chat question');
  expect(await page.evaluate(() => window.__restored)).toEqual(['d1']);

  // new chat: empty, titled by its first question, autosaved without an LLM title
  await row('+ new chat').click();
  await expect(row('New chat')).toHaveAttribute('aria-current', 'true');
  await expect(root.locator('#msgs')).toHaveText('');
  await row('add message').click();
  await expect(row('A brand new question')).toBeVisible();
  await expect.poll(() => page.evaluate(() => window.__saves)).toEqual([
    { title: 'A brand new question', id: expect.stringMatching(/^s\d+$/), count: 1 },
  ]);
  const newId = (await page.evaluate(() => window.__saves))[0].id;

  // rename another session
  await row('Renamed one').hover();
  await root.locator('div.group', { hasText: 'Renamed one' }).getByRole('button', { name: 'Rename chat' }).click();
  await root.getByLabel('Chat name').fill('Wagner notes');
  await root.getByLabel('Chat name').press('Enter');
  await expect(row('Wagner notes')).toBeVisible();
  expect((await page.evaluate(() => window.__saves)).at(-1)).toEqual({ title: 'Wagner notes', id: 's1', count: 1 });

  // delete the active session: falls back to the next one
  await row('A brand new question').hover();
  const active = root.locator('div.group', { hasText: 'A brand new question' });
  await active.getByRole('button', { name: 'Del' }).click();
  await active.getByRole('button', { name: 'Sure?' }).click();
  await expect.poll(() => page.evaluate(() => window.__deleted)).toEqual([newId]);
  await expect(row('A brand new question')).toHaveCount(0);
  await expect(root.locator('#msgs')).toHaveText('Second chat question');
});

test('a short local cache never replaces the saved history', async ({ page }) => {
  await page.goto('/index.html');
  await page.evaluate(() => {
    // the cache keeps only recent messages; here it also holds one unsaved message
    localStorage.setItem('folio.session.nb-cache-test.v1', 's1');
    localStorage.setItem('folio.chat.nb-cache-test.v1', JSON.stringify([
      { id: 'm2', role: 'assistant', text: 'two' }, { id: 'm3', role: 'user', text: 'three' },
    ]));
    window.FOLIO_API.listConversations = async () => [
      { id: 'other', title: 'Other', messages: [{ id: 'x', role: 'user', text: 'other' }], source_document_ids: [] },
      { id: 's1', title: 'Mine', source_document_ids: [],
        messages: [{ id: 'm1', role: 'user', text: 'one' }, { id: 'm2', role: 'assistant', text: 'two' }] },
    ];
    window.__saved = null;
    window.FOLIO_API.saveConversation = async (title, messages, docs, nb, id) => {
      window.__saved = { id, texts: messages.map(m => m.text) };
      return { id, title, messages, source_document_ids: docs };
    };
    const Harness = () => {
      const c = window.useChat('nb-cache-test', [], () => {});
      return React.createElement('div', { id: 'cache-msgs' }, c.chat.map(m => m.text).join(' / '));
    };
    const host = document.createElement('div');
    document.body.appendChild(host);
    ReactDOM.createRoot(host).render(React.createElement(Harness));
  });
  await expect(page.locator('#cache-msgs')).toHaveText('one / two / three');
  await expect.poll(() => page.evaluate(() => window.__saved)).toEqual({ id: 's1', texts: ['one', 'two', 'three'] });
});
