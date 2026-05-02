// src/components/common/Spinner.jsx

const styles = `
  @keyframes spin { to { transform: rotate(360deg); } }
  .spinner-ring {
    display: inline-block; border-radius: 50%; border-style: solid;
    border-color: var(--border) var(--border) var(--border) var(--amber-400);
    animation: spin 0.75s linear infinite; flex-shrink: 0;
  }
  .spinner-page {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; gap: 14px; width: 100%; min-height: 240px;
    color: var(--text-muted); font-family: var(--font-mono); font-size: 0.8125rem;
  }
  .spinner-overlay {
    position: fixed; inset: 0; background: rgba(14,15,17,0.75);
    backdrop-filter: blur(4px); display: flex; align-items: center;
    justify-content: center; z-index: 9999; flex-direction: column;
    gap: 16px; color: var(--text-muted); font-family: var(--font-mono); font-size: 0.875rem;
  }
  .spinner-inline {
    display: inline-flex; align-items: center; gap: 8px;
    color: var(--text-muted); font-size: 0.875rem;
  }
`

const SIZES = { xs:{size:14,border:2}, sm:{size:18,border:2}, md:{size:28,border:3}, lg:{size:40,border:3}, xl:{size:56,border:4} }

function Ring({ size = "md" }) {
  const { size: px, border } = SIZES[size] || SIZES.md
  return (
    <span className="spinner-ring"
      style={{ width: px, height: px, borderWidth: border }}
      aria-hidden="true" />
  )
}

export function PageSpinner({ label = "Loading\u2026" }) {
  return (
    <>
      <style>{styles}</style>
      <div className="spinner-page" role="status" aria-label={label}>
        <Ring size="lg" /><span>{label}</span>
      </div>
    </>
  )
}

export function OverlaySpinner({ label = "Loading\u2026" }) {
  return (
    <>
      <style>{styles}</style>
      <div className="spinner-overlay" role="status" aria-label={label}>
        <Ring size="xl" /><span>{label}</span>
      </div>
    </>
  )
}

export function InlineSpinner({ label, size = "sm" }) {
  return (
    <>
      <style>{styles}</style>
      <span className="spinner-inline" role="status">
        <Ring size={size} />
        {label && <span>{label}</span>}
      </span>
    </>
  )
}

export default function Spinner({ variant = "page", size = "lg", label }) {
  if (variant === "overlay") return <OverlaySpinner label={label} />
  if (variant === "inline")  return <InlineSpinner  label={label} size={size} />
  return <PageSpinner label={label} />
}