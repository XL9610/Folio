// Shared primitives ...buttons, chips, badges
const cx = (...a) => a.filter(Boolean).join(' ');

const Button = ({ children, variant = 'primary', size = 'md', onClick, className = '', icon, iconRight, disabled, type = 'button' }) => {
  const variants = {
    primary: 'bg-ink text-cream hover:opacity-90 disabled:opacity-50',
    indigo: 'bg-indigo-500 text-cream hover:bg-indigo-600 disabled:opacity-50',
    secondary: 'bg-card text-ink border border-line hover:border-ink/40 hover:bg-cream',
    ghost: 'text-ink2 hover:text-ink hover:bg-line2/50',
    quiet: 'text-muted hover:text-ink',
    outline: 'border border-line text-ink hover:border-indigo-500 hover:text-indigo-500',
  };
  const sizes = {
    sm: 'h-8 px-3 text-[13px] gap-1.5 rounded-full',
    md: 'h-9 px-3.5 text-[13px] gap-2 rounded-full',
    lg: 'h-11 px-5 text-[14px] gap-2 rounded-full',
  };
  return (
    <button type={type} onClick={onClick} disabled={disabled}
      className={cx('inline-flex items-center justify-center font-normal tracking-normal transition-colors whitespace-nowrap', variants[variant], sizes[size], className)}>
      {icon}
      {children}
      {iconRight}
    </button>
  );
};

const Card = ({ children, className = '', as: As = 'div', ...rest }) => (
  <As className={cx('bg-card border border-line rounded-lg shadow-card', className)} {...rest}>{children}</As>
);

// Shared animated spinner ...use wherever generation/processing is happening
const InlineSpinner = ({ size = 14, color }) => (
  <span
    className="inline-block animate-spin"
    style={{
      width: size,
      height: size,
      borderWidth: 2,
      borderStyle: 'solid',
      borderLeftColor: color || 'var(--accent)',
      borderRightColor: color || 'var(--accent)',
      borderBottomColor: color || 'var(--accent)',
      borderTopColor: 'transparent',
      borderRadius: '50%',
      verticalAlign: 'middle',
      flexShrink: 0,
    }}
    role="status"
    aria-label="loading"
  />
);

const ConfirmDeleteButton = ({ onDelete, className, label = 'Del' }) => {
  const [confirming, setConfirming] = React.useState(false);
  const [deleting, setDeleting] = React.useState(false);
  const timerRef = React.useRef(null);

  React.useEffect(() => () => { if (timerRef.current) clearTimeout(timerRef.current); }, []);

  const handleClick = (e) => {
    e.stopPropagation();
    if (confirming) {
      setDeleting(true);
      const result = onDelete?.();
      if (result && typeof result.then === 'function') {
        result.finally(() => { setDeleting(false); setConfirming(false); });
      } else {
        setDeleting(false); setConfirming(false);
      }
    } else {
      setConfirming(true);
      timerRef.current = setTimeout(() => setConfirming(false), 4000);
    }
  };

  const handleCancel = (e) => { e.stopPropagation(); setConfirming(false); };

  return (
    <span className={cx('inline-flex items-center', className)}>
      {!confirming ? (
        <button onClick={handleClick} className="sans" style={{
          fontSize: 'calc(10px * var(--folio-font-scale, 1))', letterSpacing: '0.08em', textTransform: 'uppercase',
          color: 'var(--text-muted)',
        }} title="Delete">{label}</button>
      ) : (
        <span className="inline-flex items-center gap-1">
          <button onClick={handleClick} disabled={deleting} className="sans" style={{
            fontSize: 'calc(10px * var(--folio-font-scale, 1))', letterSpacing: '0.08em', textTransform: 'uppercase',
            color: deleting ? 'var(--text-muted)' : '#B6552A',
            opacity: deleting ? 0.6 : 1,
          }}>{deleting ? '...' : 'Sure?'}</button>
          <button onClick={handleCancel} className="sans" style={{
            fontSize: 'calc(14px * var(--folio-font-scale, 1))', color: 'var(--text-muted)', lineHeight: 1,
          }} aria-label="Cancel delete" title="Cancel delete">
            <Icon name="x" size={12} />
          </button>
        </span>
      )}
    </span>
  );
};

Object.assign(window, { cx, Button, Card, InlineSpinner, ConfirmDeleteButton });
