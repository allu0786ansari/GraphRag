// src/components/common/Navbar.jsx
import { NavLink } from 'react-router-dom'
import { useIndexingContext } from '../../context/IndexingContext.jsx'

const NAV_LINKS = [
  { to: '/query',      label: 'Query'      },
  { to: '/graph',      label: 'Graph'      },
  { to: '/evaluation', label: 'Evaluation' },
  { to: '/indexing',   label: 'Indexing'   },
]

function IndexStatusDot({ status, artifactsReady }) {
  let color, label
  if (artifactsReady) {
    color = 'var(--success)'; label = 'indexed'
  } else if (status === 'running' || status === 'queued') {
    color = 'var(--warning)'; label = 'indexing\u2026'
  } else if (status === 'failed') {
    color = 'var(--error)'; label = 'failed'
  } else {
    color = 'var(--text-muted)'; label = 'not indexed'
  }

  const isPulsing = status === 'running' || status === 'queued'

  return (
    <NavLink to="/indexing" className="index-status-pill" title="Indexing status">
      <span
        className={`status-dot${isPulsing ? ' pulsing' : ''}`}
        style={{ background: color, boxShadow: artifactsReady ? `0 0 6px ${color}` : 'none' }}
      />
      <span>{label}</span>
    </NavLink>
  )
}

export default function Navbar() {
  const { status, artifactsReady } = useIndexingContext()

  return (
    <>
      <style>{`
        @keyframes pulse-dot {
          0%,100% { opacity:1; transform:scale(1); }
          50%      { opacity:0.4; transform:scale(0.85); }
        }
        .navbar {
          background: var(--bg-surface);
          border-bottom: 1px solid var(--border);
          position: sticky; top: 0; z-index: 100;
        }
        .navbar-inner {
          max-width: 1400px; margin: 0 auto;
          padding: 0 24px; height: 52px;
          display: flex; align-items: center;
          justify-content: space-between; gap: 24px;
        }
        .navbar-brand {
          display: flex; align-items: center; gap: 10px;
          text-decoration: none; flex-shrink: 0;
        }
        .navbar-logo {
          width: 28px; height: 28px;
          background: linear-gradient(135deg, var(--amber-400) 0%, var(--amber-600) 100%);
          border-radius: 6px; display: flex; align-items: center;
          justify-content: center; font-family: var(--font-mono);
          font-size: 0.75rem; color: #1a1000; letter-spacing: -0.04em;
        }
        .navbar-title {
          font-family: var(--font-display); font-size: 0.9375rem;
          font-weight: 400; color: var(--text-primary);
          letter-spacing: -0.01em; white-space: nowrap;
        }
        .navbar-title em { color: var(--text-muted); font-style: italic; }
        .navbar-nav {
          display: flex; align-items: center; gap: 2px;
          flex: 1; justify-content: center;
        }
        .nav-link {
          padding: 6px 14px; border-radius: 6px; font-size: 0.875rem;
          color: var(--text-secondary); text-decoration: none;
          transition: all var(--transition); border: 1px solid transparent;
        }
        .nav-link:hover { color: var(--text-primary); background: var(--bg-elevated); }
        .nav-link.active {
          color: var(--amber-400); background: rgba(240,165,0,0.08);
          border-color: rgba(240,165,0,0.2);
        }
        .index-status-pill {
          display: flex; align-items: center; gap: 6px;
          font-family: var(--font-mono); font-size: 0.75rem;
          color: var(--text-muted); padding: 4px 10px;
          background: var(--bg-elevated); border: 1px solid var(--border);
          border-radius: 100px; text-decoration: none;
          transition: all var(--transition); flex-shrink: 0;
        }
        .index-status-pill:hover { border-color: var(--border-strong); color: var(--text-secondary); }
        .status-dot {
          width: 7px; height: 7px; border-radius: 50%; display: inline-block;
        }
        .status-dot.pulsing { animation: pulse-dot 1.6s ease-in-out infinite; }
        @media (max-width: 768px) {
          .navbar-title { display: none; }
          .nav-link { padding: 6px 10px; font-size: 0.8125rem; }
        }
      `}</style>

      <nav className="navbar" aria-label="Main navigation">
        <div className="navbar-inner">
          <NavLink to="/query" className="navbar-brand">
            <div className="navbar-logo">GR</div>
            <span className="navbar-title">
              GraphRAG <em>vs</em> VectorRAG
            </span>
          </NavLink>

          <div className="navbar-nav">
            {NAV_LINKS.map(({ to, label }) => (
              <NavLink
                key={to}
                to={to}
                className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
              >
                {label}
              </NavLink>
            ))}
          </div>

          <IndexStatusDot status={status} artifactsReady={artifactsReady} />
        </div>
      </nav>
    </>
  )
}