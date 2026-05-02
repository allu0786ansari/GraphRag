// src/utils/formatters.js

export function formatNumber(n) {
  if (n == null) return '—'
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M'
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K'
  return n.toLocaleString()
}

export function formatTokens(n) {
  if (n == null) return '—'
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(2) + 'M tokens'
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K tokens'
  return n + ' tokens'
}

export function formatCost(usd) {
  if (usd == null) return '—'
  if (usd === 0) return '$0.0000 (free)'
  if (usd < 0.0001) return '<$0.0001'
  return '$' + usd.toFixed(4)
}

export function formatLatency(ms) {
  if (ms == null) return '—'
  if (ms >= 60_000) return (ms / 60_000).toFixed(1) + 'min'
  if (ms >= 1_000) return (ms / 1_000).toFixed(1) + 's'
  return Math.round(ms) + 'ms'
}

export function formatDate(iso) {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  } catch {
    return iso
  }
}

export function formatRelativeTime(iso) {
  if (!iso) return '—'
  const diff = Date.now() - new Date(iso).getTime()
  const seconds = Math.floor(diff / 1000)
  if (seconds < 60) return 'just now'
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  return `${days}d ago`
}

export function truncateText(text, maxChars = 100) {
  if (!text) return ''
  if (text.length <= maxChars) return text
  return text.slice(0, maxChars).trimEnd() + '…'
}

export function formatWinRate(rate) {
  if (rate == null) return '—'
  return (rate * 100).toFixed(0) + '%'
}

export function formatDuration(seconds) {
  if (seconds == null) return '—'
  if (seconds < 60) return seconds.toFixed(1) + 's'
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds % 60)
  return `${m}m ${s}s`
}