import { useEffect, useState } from 'react'
import './Toast.css'

// A transient, non-blocking notification - unlike Alert, it floats over the
// page instead of pushing layout, and dismisses itself. Hovering pauses the
// timer so it can't vanish out from under someone still reading it.
export default function Toast({ message, variant = 'error', duration = 5000, onClose }) {
  const [closing, setClosing] = useState(false)
  const [paused, setPaused] = useState(false)

  useEffect(() => {
    if (paused) return
    const timer = setTimeout(() => setClosing(true), duration)
    return () => clearTimeout(timer)
  }, [paused, duration])

  useEffect(() => {
    if (!closing) return
    const timer = setTimeout(onClose, 200) // matches the exit animation in Toast.css
    return () => clearTimeout(timer)
  }, [closing, onClose])

  return (
    <div
      className={`toast toast-${variant} ${closing ? 'toast-closing' : ''}`}
      role="alert"
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
    >
      <svg className="toast-icon" viewBox="0 0 24 24" width="18" height="18" fill="none" aria-hidden="true">
        <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2" />
        <path d="M12 8v5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
        <circle cx="12" cy="16" r="1" fill="currentColor" />
      </svg>
      <p className="toast-message">{message}</p>
      <button type="button" className="toast-close" onClick={() => setClosing(true)} aria-label="Dismiss">
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" aria-hidden="true">
          <path d="M6 6L18 18M18 6L6 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
        </svg>
      </button>
    </div>
  )
}
