const { test, expect } = require('@playwright/test');

test('a multi-file upload surfaces every queued source immediately', async ({ page }) => {
  await page.goto('/index.html');

  await page.evaluate(() => {
    const host = document.createElement('div');
    host.id = 'upload-test-root';
    document.body.appendChild(host);

    window.__uploadStarts = [];
    window.FOLIO_API.uploadDocument = async (file) => ({
      job_id: `job-${file.name}`,
      filename: file.name,
    });
    window.FOLIO_API.getJob = async () => ({
      status: 'running',
      stage: 'extracting',
    });

    const root = ReactDOM.createRoot(host);
    root.render(React.createElement(window.UploadModal, {
      open: true,
      onClose: () => {},
      onComplete: () => {},
      onUploadStart: (id, name) => window.__uploadStarts.push({ id, name }),
      notebookId: 'holdout-notebook',
    }));
  });

  await page.locator('#upload-test-root input[type="file"]').setInputFiles([
    { name: 'first.pdf', mimeType: 'application/pdf', buffer: Buffer.from('%PDF-first') },
    { name: 'second.epub', mimeType: 'application/epub+zip', buffer: Buffer.from('epub-second') },
  ]);
  await expect(page.getByText('2 files selected')).toBeVisible();

  await page.getByRole('button', { name: 'Ingest 2 files' }).click();

  await expect.poll(() => page.evaluate(() => window.__uploadStarts)).toEqual([
    expect.objectContaining({ name: 'first.pdf' }),
    expect.objectContaining({ name: 'second.epub' }),
  ]);
});
