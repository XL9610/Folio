// tweaks-panel.jsx
//
// Provides `useTweaks` — the hook that owns lightweight app-level tweak values
// (theme, showSidebar, showStudio). The old floating "Tweaks" debug panel and
// its form controls were removed: they were inert prototype tooling that only
// opened on a __activate_edit_mode postMessage nothing in the shipped app sends.

function useTweaks(defaults) {
  const [values, setValues] = React.useState(defaults);
  // Accepts setTweak('key', value) or setTweak({ key: value, ... }) so a
  // useState-style call doesn't write a "[object Object]" key.
  const setTweak = React.useCallback((keyOrEdits, val) => {
    const edits = typeof keyOrEdits === 'object' && keyOrEdits !== null
      ? keyOrEdits : { [keyOrEdits]: val };
    setValues((prev) => ({ ...prev, ...edits }));
    // Same-window signal so in-page listeners can react to tweak changes.
    window.dispatchEvent(new CustomEvent('tweakchange', { detail: edits }));
  }, []);
  return [values, setTweak];
}

window.useTweaks = useTweaks;
