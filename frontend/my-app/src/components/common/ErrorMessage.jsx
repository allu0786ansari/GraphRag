// src/components/common/ErrorMessage.jsx
import { X, AlertTriangle, Info, AlertCircle } from "lucide-react"

const CONFIG = {
  error:   { icon: AlertCircle,   color: "var(--error)",   bg: "var(--error-bg)",   border: "rgba(217,79,79,0.3)"   },
  warning: { icon: AlertTriangle, color: "var(--warning)", bg: "var(--warning-bg)", border: "rgba(212,137,0,0.3)"  },
  info:    { icon: Info,          color: "var(--info)",    bg: "var(--info-bg)",    border: "rgba(59,130,246,0.3)"  },
}

export default function ErrorMessage({
  message,
  variant = "error",
  onDismiss,
  title,
  className = "",
}) {
  if (!message) return null

  const { icon: Icon, color, bg, border } = CONFIG[variant] || CONFIG.error

  return (
    <>
      <style>{`
        .error-msg {
          display: flex; align-items: flex-start; gap: 10px;
          padding: 12px 14px; border-radius: var(--radius-md);
          border: 1px solid; font-size: 0.875rem; line-height: 1.5;
          animation: fade-in-down 0.2s ease;
        }
        @keyframes fade-in-down {
          from { opacity: 0; transform: translateY(-6px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        .error-msg-icon { flex-shrink: 0; margin-top: 1px; }
        .error-msg-body { flex: 1; min-width: 0; }
        .error-msg-title { font-weight: 500; margin-bottom: 2px; }
        .error-msg-text  { word-break: break-word; }
        .error-msg-dismiss {
          flex-shrink: 0; background: none; border: none; cursor: pointer;
          padding: 2px; border-radius: 4px; opacity: 0.6;
          transition: opacity var(--transition); color: inherit;
          display: flex; align-items: center; margin-top: -1px;
        }
        .error-msg-dismiss:hover { opacity: 1; }
      `}</style>

      <div
        className={`error-msg ${className}`}
        role={variant === "error" ? "alert" : "status"}
        style={{ background: bg, borderColor: border, color }}
      >
        <Icon className="error-msg-icon" size={16} strokeWidth={2} />

        <div className="error-msg-body">
          {title && <div className="error-msg-title">{title}</div>}
          <div className="error-msg-text">{message}</div>
        </div>

        {onDismiss && (
          <button
            className="error-msg-dismiss"
            onClick={onDismiss}
            aria-label="Dismiss"
            style={{ color }}
          >
            <X size={14} />
          </button>
        )}
      </div>
    </>
  )
}