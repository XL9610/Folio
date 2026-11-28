const { test, expect } = require('@playwright/test');

const sse = (events) => events.map(e => `data: ${JSON.stringify(e)}\n\n`).join('');

test('streamed ask delivers step and token events, then the final result', async ({ page }) => {
  await page.route('**/api/ask/stream', route => route.fulfill({
    status: 200,
    contentType: 'text/event-stream',
    body: sse([
      { type: 'step', reasoning: '', label: 'Searching for "aura"' },
      { type: 'step', reasoning: 'Need the essay itself.', label: 'Reading Chapter 8' },
      { type: 'token', text: 'The aura ' },
      { type: 'token', text: 'declines.' },
      { type: 'done', result: { answer: 'The aura declines. [S1]', agent_status: 'completed', citations: [] } },
    ]),
  }));
  await page.goto('/index.html');

  const out = await page.evaluate(async () => {
    const events = [];
    const result = await window.FOLIO_API.ask('q', 'nb', ['d1'], 10, [], { onEvent: e => events.push(e) });
    return { events, result };
  });
  expect(out.events.map(e => e.type)).toEqual(['step', 'step', 'token', 'token']);
  expect(out.events[1].reasoning).toBe('Need the essay itself.');
  expect(out.result.answer).toBe('The aura declines. [S1]');
});

test('stream error event rejects the ask', async ({ page }) => {
  await page.route('**/api/ask/stream', route => route.fulfill({
    status: 200, contentType: 'text/event-stream',
    body: sse([{ type: 'error', message: 'LLM API call failed: boom' }]),
  }));
  await page.goto('/index.html');
  const message = await page.evaluate(() =>
    window.FOLIO_API.ask('q', 'nb', ['d1'], 10, [], { onEvent: () => {} }).then(() => 'resolved', e => e.message));
  expect(message).toBe('LLM API call failed: boom');
});

test('ReAct steps render collapsed to one line and expand to the reasoning', async ({ page }) => {
  await page.goto('/index.html');
  await page.evaluate(() => {
    const host = document.createElement('div');
    host.id = 'steps-root';
    document.body.appendChild(host);
    const chat = [
      { id: 'u1', role: 'user', text: 'What is the aura?' },
      { id: 'a1', role: 'assistant', text: 'The aura declines.', citations: [],
        _steps: [{ label: 'Searching for "aura"', reasoning: '' },
                 { label: 'Reading Chapter 8', reasoning: 'Need the essay itself.' }] },
    ];
    ReactDOM.createRoot(host).render(React.createElement(window.ChatCenter, {
      chat, setChat: () => {}, thinking: false, setThinking: () => {}, onSend: () => {},
      includedCount: 1, totalSources: 1, prompts: [], sourceDocumentIds: ['d1'],
      draftInput: '', setDraftInput: () => {}, llmConfigured: true,
    }));
  });
  const summary = page.locator('#steps-root summary', { hasText: 'Looked in 2 places' });
  await expect(summary).toBeVisible();
  await expect(page.locator('#steps-root').getByText('Need the essay itself.')).toBeHidden();
  await summary.click();
  await expect(page.locator('#steps-root').getByText('Need the essay itself.')).toBeVisible();
});

test('a streaming message that starts empty renders its text once tokens arrive', async ({ page }) => {
  await page.goto('/index.html');
  await page.evaluate(() => {
    const host = document.createElement('div');
    host.id = 'stream-root';
    document.body.appendChild(host);
    const root = ReactDOM.createRoot(host);
    const render = (text) => root.render(React.createElement(window.ChatCenter, {
      chat: [{ id: 'u1', role: 'user', text: 'q' },
             { id: 'a1', role: 'assistant', text, citations: [], _streaming: true,
               _steps: [{ label: 'Searching', reasoning: '' }] }],
      setChat: () => {}, thinking: true, setThinking: () => {}, onSend: () => {},
      includedCount: 1, totalSources: 1, prompts: [], sourceDocumentIds: ['d1'],
      draftInput: '', setDraftInput: () => {}, llmConfigured: true,
    }));
    render('');
    setTimeout(() => render('The aura declines.'), 100);
  });
  await expect(page.locator('#stream-root').getByText('The aura declines.')).toBeVisible();
});
