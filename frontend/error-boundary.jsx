const { Component } = React;

class ErrorBoundary extends Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null };
    }
    static getDerivedStateFromError(error) {
        return { hasError: true, error };
    }
    render() {
        if (this.state.hasError) {
            return this.props.fallback || (
                <div style={{
                    padding: 40, textAlign: 'center', display: 'flex',
                    flexDirection: 'column', alignItems: 'center',
                    justifyContent: 'center', height: '100%',
                }}>
                    <div style={{ fontSize: 'calc(18px * var(--folio-font-scale, 1))', fontWeight: 600,
                        color: 'var(--text-primary)', marginBottom: 8 }}>
                        Something went wrong
                    </div>
                    <div style={{ fontSize: 'calc(13px * var(--folio-font-scale, 1))', color: 'var(--text-muted)',
                        marginBottom: 16, maxWidth: 400 }}>
                        {this.state.error?.message || 'An unexpected error occurred.'}
                    </div>
                    <button onClick={() => this.setState({ hasError: false, error: null })}
                        style={{ padding: '8px 20px', borderRadius: 20,
                            background: 'var(--accent)', color: 'var(--surface-main)',
                            border: 'none', cursor: 'pointer', fontSize: 'calc(13px * var(--folio-font-scale, 1))' }}>
                        Try again
                    </button>
                </div>
            );
        }
        return this.props.children;
    }
}

window.ErrorBoundary = ErrorBoundary;
