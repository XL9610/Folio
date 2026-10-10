// Frameless-window title bar.
//
// Only rendered inside Electron (window.electronAPI is set by the preload); in
// the browser it returns null so the dev build is unchanged. The strip is a
// drag region; on Linux (frame:false) it also draws min/max/close controls
// wired to the main process via IPC. Windows/macOS keep native controls via
// titleBarOverlay / traffic lights, so no buttons are drawn there.

const TitleBar = () => {
  const api = window.electronAPI;
  if (!api) return null;

  const showControls = api.platform === 'linux';
  const barStyle = {
    height: 40,
    flexShrink: 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '0 14px 0 16px',
    background: 'var(--surface-main)',
    borderBottom: '1px solid var(--border-soft)',
    WebkitAppRegion: 'drag',
    userSelect: 'none',
  };
  const btnStyle = {
    WebkitAppRegion: 'no-drag',
    border: 'none',
    background: 'transparent',
    cursor: 'pointer',
    color: 'var(--text-secondary)',
    width: 34,
    height: 28,
    borderRadius: 6,
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: 14,
    lineHeight: 1,
  };

  return (
    <div className="titlebar" style={barStyle}>
      <span style={{ fontSize: 'calc(12px * var(--folio-font-scale, 1))', color: 'var(--text-secondary)', letterSpacing: '0.03em' }}>
        Folio — Study Companion
      </span>
      {showControls && (
        <div style={{ display: 'flex', gap: 2 }}>
          <button type="button" style={btnStyle} onClick={() => api.minimize()} title="Minimize">–</button>
          <button type="button" style={btnStyle} onClick={() => api.maximize()} title="Maximize">□</button>
          <button type="button" style={btnStyle} onClick={() => api.close()} title="Close">×</button>
        </div>
      )}
    </div>
  );
};

window.TitleBar = TitleBar;
