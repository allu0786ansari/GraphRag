// src/components/common/Toast.jsx
import { createContext, useContext, useCallback, useReducer, useEffect, useRef } from "react"
import { CheckCircle, AlertCircle, Info, X } from "lucide-react"

// ── Context ────────────────────────────────────────────────────────────────────
const ToastContext = createContext(null)

const ICONS = {
  success: CheckCircle,
  error:   AlertCircle,
  info:    Info,
}

const COLORS = {
  success: { color: "var(--success)", bg: "var(--success-bg)", border: "rgba(42,157,92,0.3)" },
  error:   { color: "var(--error)",   bg: "var(--error-bg)",   border: "rgba(217,79,79,0.3)" },
  info:    { color: "var(--info)",    bg: "var(--info-bg)",    border: "rgba(59,130,246,0.3)" },
}

let _toastId = 0

function toastReducer(state, action) {
  switch (action.type) {
    case "ADD":    return [...state, action.payload]
    case "REMOVE": return state.filter(t => t.id !== action.id)
    default:       return state
  }
}

// ── Single toast item ─────────────────────────────────────────────────────────
function ToastItem({ toast, onRemove }) {
  const Icon = ICONS[toast.variant] || Info
  const { color, bg, border } = COLORS[toast.variant] || COLORS.info
  const timerRef = useRef(null)

  useEffect(() => {
    timerRef.current = setTimeout(() => onRemove(toast.id), toast.duration || 4000)
    return () => clearTimeout(timerRef.current)
  }, [toast.id, toast.duration, onRemove])

  return (
    <div
      className="toast-item"
      role="alert"
      style={{ background: bg, borderColor: border, color }}
    >
      <Icon size={15} strokeWidth={2} style={{ flexShrink: 0, marginTop: 1 }} />
      <span className="toast-message">{toast.message}</span>
      <button
        className="toast-close"
        onClick={() => onRemove(toast.id)}
        aria-label="Dismiss notification"
        style={{ color }}
      >
        <X size={13} />
      </button>
    </div>
  )
}

// ── Provider ──────────────────────────────────────────────────────────────────
export function ToastProvider({ children }) {
  const [toasts, dispatch] = useReducer(toastReducer, [])

  const addToast = useCallback(({ message, variant = "info", duration = 4000 }) => {
    const id = ++_toastId
    dispatch({ type: "ADD", payload: { id, message, variant, duration } })
  }, [])

  const removeToast = useCallback((id) => {
    dispatch({ type: "REMOVE", id })
  }, [])

  const toast = useCallback({
    success: (message, opts) => addToast({ message, variant: "success", ...opts }),
    error:   (message, opts) => addToast({ message, variant: "error",   ...opts }),
    info:    (message, opts) => addToast({ message, variant: "info",    ...opts }),
  }, [addToast])

  // Also expose addToast directly
  toast.show = addToast

  return (
    <ToastContext.Provider value={toast}>
      <style>{`
        .toast-portal {
          position: fixed; top: 16px; right: 16px; z-index: 10000;
          display: flex; flex-direction: column; gap: 8px;
          max-width: 360px; width: calc(100vw - 32px);
          pointer-events: none;
        }
        .toast-item {
          display: flex; align-items: flex-start; gap: 10px;
          padding: 11px 14px; border-radius: var(--radius-md);
          border: 1px solid; font-size: 0.875rem; line-height: 1.4;
          box-shadow: var(--shadow-lg); pointer-events: all;
          animation: toast-in 0.25s cubic-bezier(0.34,1.56,0.64,1);
        }
        @keyframes toast-in {
          from { opacity: 0; transform: translateX(20px) scale(0.96); }
          to   { opacity: 1; transform: translateX(0) scale(1); }
        }
        .toast-message { flex: 1; word-break: break-word; }
        .toast-close {
          flex-shrink: 0; background: none; border: none; cursor: pointer;
          padding: 1px; border-radius: 3px; opacity: 0.6;
          transition: opacity var(--transition); display: flex; align-items: center;
        }
        .toast-close:hover { opacity: 1; }
      `}</style>

      {children}

      <div className="toast-portal" aria-live="polite" aria-atomic="false">
        {toasts.map(t => (
          <ToastItem key={t.id} toast={t} onRemove={removeToast} />
        ))}
      </div>
    </ToastContext.Provider>
  )
}

// ── Hook ──────────────────────────────────────────────────────────────────────
export function useToast() {
  const ctx = useContext(ToastContext)
  if (!ctx) throw new Error("useToast must be used inside <ToastProvider>")
  return ctx
}

export default ToastProvider