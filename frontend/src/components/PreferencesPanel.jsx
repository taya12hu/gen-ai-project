import { useEffect, useState } from 'react'
import { X, Trash2 } from 'lucide-react'
import { ApiError, deletePreference, listPreferences } from '../lib/api'
import { friendlyErrorMessage } from '../lib/errorMessages'
import { labelFor } from '../lib/preferenceLabels'
import Spinner from './Spinner'
import './PreferencesPanel.css'

// A settings-style dialog for the durable, cross-session preference facts
// DineMind has picked up from chat (see app.conversation.preferences on the
// backend) - the only place a user can actually see and remove one, since
// they're captured automatically rather than through an explicit form.
export default function PreferencesPanel({ token, onClose, onAuthError }) {
  const [preferences, setPreferences] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [forgettingKey, setForgettingKey] = useState(null)

  useEffect(() => {
    function onKeyDown(e) {
      if (e.key === 'Escape') onClose()
    }
    document.body.style.overflow = 'hidden'
    window.addEventListener('keydown', onKeyDown)
    return () => {
      document.body.style.overflow = ''
      window.removeEventListener('keydown', onKeyDown)
    }
  }, [onClose])

  useEffect(() => {
    listPreferences(token)
      .then(setPreferences)
      .catch((err) => {
        if (err instanceof ApiError && err.status === 401 && onAuthError?.(err)) return
        setError(friendlyErrorMessage(err))
      })
      .finally(() => setLoading(false))
  }, [token, onAuthError])

  async function handleForget(key) {
    setError(null)
    setForgettingKey(key)
    try {
      await deletePreference(key, token)
      setPreferences((prev) => prev.filter((p) => p.key !== key))
    } catch (err) {
      if (err instanceof ApiError && err.status === 401 && onAuthError?.(err)) return
      // Already gone server-side - reflect that locally instead of leaving
      // a stale row the user can't clear.
      if (err instanceof ApiError && err.errorCode === 'preference_not_found') {
        setPreferences((prev) => prev.filter((p) => p.key !== key))
      } else {
        setError(friendlyErrorMessage(err))
      }
    } finally {
      setForgettingKey(null)
    }
  }

  return (
    <div className="prefs-backdrop" onClick={onClose}>
      <div
        className="prefs-panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="prefs-title"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="prefs-header">
          <h2 id="prefs-title">Remembered preferences</h2>
          <button type="button" className="prefs-close" onClick={onClose} aria-label="Close">
            <X size={18} />
          </button>
        </div>
        <p className="prefs-subtitle">
          Lasting facts DineMind has picked up from your conversations and may use to shape future
          suggestions. Remove anything that shouldn't stick around.
        </p>

        {error && <p className="prefs-error">{error}</p>}

        {loading ? (
          <div className="prefs-loading">
            <Spinner size={18} label="Loading preferences" />
          </div>
        ) : preferences.length === 0 ? (
          <p className="prefs-empty">
            Nothing remembered yet - it'll show up here the next time you mention a lasting preference.
          </p>
        ) : (
          <ul className="prefs-list">
            {preferences.map((p) => (
              <li key={p.key} className="prefs-item">
                <div className="prefs-item-text">
                  <span className="prefs-item-key">{labelFor(p.key)}</span>
                  <span className="prefs-item-value">{p.value}</span>
                </div>
                <button
                  type="button"
                  className="prefs-forget"
                  onClick={() => handleForget(p.key)}
                  disabled={forgettingKey === p.key}
                  aria-label={`Forget ${labelFor(p.key)} preference`}
                  title="Forget"
                >
                  {forgettingKey === p.key ? <Spinner size={14} /> : <Trash2 size={14} />}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}
