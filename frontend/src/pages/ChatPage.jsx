import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { LogOut, Menu, X } from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import {
  ApiError,
  createConversation,
  fetchConversationMessages,
  listConversations,
  sendChatMessageStream,
} from '../lib/api'
import { friendlyErrorMessage } from '../lib/errorMessages'
import Button from '../components/Button'
import Toast from '../components/Toast'
import Spinner from '../components/Spinner'
import EmptyState from '../components/EmptyState'
import ChatMessageBubble from '../components/ChatMessageBubble'
import BrandLogo from '../components/BrandLogo'
import './ChatPage.css'

// First letter of whatever identifies the user, for the sidebar avatar.
function avatarInitial(user) {
  const label = user?.display_name?.trim() || user?.email || ''
  return label.charAt(0).toUpperCase() || '?'
}

function formatConversationLabel(createdAt) {
  return new Date(createdAt).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  })
}

function startOfDay(date) {
  const d = new Date(date)
  d.setHours(0, 0, 0, 0)
  return d.getTime()
}

const DAY_MS = 24 * 60 * 60 * 1000
const TITLE_MAX_LENGTH = 60

// Mirrors the backend's _derive_title (app/conversation/store.py) so a
// brand-new conversation's sidebar entry matches what the server will
// persist, without waiting on a round trip to find out.
function deriveTitle(text) {
  const collapsed = text.trim().replace(/\s+/g, ' ')
  if (collapsed.length <= TITLE_MAX_LENGTH) return collapsed
  return collapsed.slice(0, TITLE_MAX_LENGTH).trimEnd() + '…'
}

// "Today" / "Yesterday", then each of the last few calendar days by exact
// date ("Aug 9"), then everything older collapses into "Older" - same
// grouping shape as Claude's own chat history sidebar.
function conversationGroupLabel(createdAt, todayStart) {
  const dayStart = startOfDay(createdAt)
  const diffDays = Math.round((todayStart - dayStart) / DAY_MS)
  if (diffDays <= 0) return 'Today'
  if (diffDays === 1) return 'Yesterday'
  if (diffDays <= 6) return new Date(dayStart).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
  return 'Older'
}

// Buckets conversations by conversationGroupLabel, preserving the list's
// existing most-recent-first order within (and across) groups.
function groupConversationsByDate(conversations) {
  const todayStart = startOfDay(new Date())
  const groups = new Map()

  for (const c of conversations) {
    const label = conversationGroupLabel(c.created_at, todayStart)
    if (!groups.has(label)) groups.set(label, [])
    groups.get(label).push(c)
  }

  return groups
}

export default function ChatPage() {
  const { token, user, logout } = useAuth()
  const navigate = useNavigate()

  const [conversations, setConversations] = useState([])
  const [conversationId, setConversationId] = useState(null)
  const [messages, setMessages] = useState([])
  const [searchQuery, setSearchQuery] = useState('')
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [awaitingFirstToken, setAwaitingFirstToken] = useState(false)
  const [historyLoading, setHistoryLoading] = useState(true)
  const [error, setError] = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const scrollRef = useRef(null)

  // Mobile only (CSS ignores this state above the drawer breakpoint) - keeps
  // the drawer closable with Escape and stops the page scrolling behind it.
  useEffect(() => {
    if (!sidebarOpen) return
    document.body.style.overflow = 'hidden'
    function onKeyDown(e) {
      if (e.key === 'Escape') setSidebarOpen(false)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => {
      document.body.style.overflow = ''
      window.removeEventListener('keydown', onKeyDown)
    }
  }, [sidebarOpen])

  function handleLogout() {
    logout()
    navigate('/login')
  }

  function handleAuthError(err) {
    if (err instanceof ApiError && err.status === 401) {
      logout()
      navigate('/login?reason=session_expired')
      return true
    }
    return false
  }

  useEffect(() => {
    listConversations(token)
      .then(async (convos) => {
        setConversations(convos)
        if (convos.length > 0) {
          setConversationId(convos[0].id)
          const msgs = await fetchConversationMessages(convos[0].id, token)
          setMessages(msgs)
        }
      })
      .catch((err) => {
        if (!handleAuthError(err)) setError(friendlyErrorMessage(err))
      })
      .finally(() => setHistoryLoading(false))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages, loading])

  async function selectConversation(id) {
    setError(null)
    setConversationId(id)
    try {
      const msgs = await fetchConversationMessages(id, token)
      setMessages(msgs)
    } catch (err) {
      if (handleAuthError(err)) return
      if (err instanceof ApiError && err.errorCode === 'conversation_not_found') {
        setConversations((prev) => prev.filter((c) => c.id !== id))
        startNewChat()
      }
      setError(friendlyErrorMessage(err))
    }
  }

  function startNewChat() {
    setConversationId(null)
    setMessages([])
    setError(null)
  }

  async function handleSend(e) {
    e.preventDefault()
    const text = input.trim()
    if (!text || loading) return

    setInput('')
    setError(null)
    setMessages((prev) => [...prev, { id: `local-${Date.now()}`, role: 'user', content: text }])
    setLoading(true)
    setAwaitingFirstToken(true)

    const replyId = `reply-${Date.now()}`
    let streamedText = ''
    let bubbleStarted = false

    try {
      let convId = conversationId
      if (!convId) {
        const conversation = await createConversation(token)
        convId = conversation.id
        setConversationId(convId)
        // Title comes back null (the server only sets it once this message
        // is persisted) - fill it in optimistically so the sidebar entry
        // doesn't briefly show a raw timestamp before flipping to a title.
        setConversations((prev) => [{ ...conversation, title: deriveTitle(text) }, ...prev])
      }

      await sendChatMessageStream(convId, text, token, {
        onToken: (chunk) => {
          streamedText += chunk
          if (!bubbleStarted) {
            // First chunk actually arrived - swap the "Thinking…" spinner
            // for a live, growing reply bubble instead of waiting for the
            // whole thing to finish.
            bubbleStarted = true
            setAwaitingFirstToken(false)
            setMessages((prev) => [
              ...prev,
              { id: replyId, role: 'assistant', content: streamedText, matched_restaurants: [] },
            ])
          } else {
            setMessages((prev) => prev.map((m) => (m.id === replyId ? { ...m, content: streamedText } : m)))
          }
        },
        onDone: (data) => {
          setMessages((prev) =>
            prev.map((m) => (m.id === replyId ? { ...m, matched_restaurants: data.matched_restaurants } : m))
          )
        },
        onError: (err) => {
          throw err
        },
      })
    } catch (err) {
      if (handleAuthError(err)) {
        // The user bubble for this turn stays in local state until the
        // redirect happens - not persisted server-side, but that's fine,
        // logging back in starts a fresh view anyway.
      } else {
        if (err instanceof ApiError && err.errorCode === 'conversation_not_found') {
          setConversations((prev) => prev.filter((c) => c.id !== conversationId))
          setConversationId(null)
        }
        setError(friendlyErrorMessage(err))
      }
    } finally {
      setLoading(false)
      setAwaitingFirstToken(false)
    }
  }

  const filteredConversations = searchQuery.trim()
    ? conversations.filter((c) =>
        (c.title || formatConversationLabel(c.created_at)).toLowerCase().includes(searchQuery.trim().toLowerCase())
      )
    : conversations
  const conversationGroups = groupConversationsByDate(filteredConversations)

  function closeSidebarAnd(fn) {
    return (...args) => {
      fn(...args)
      setSidebarOpen(false)
    }
  }

  return (
    <div className="chat-page">
      {sidebarOpen && <div className="chat-sidebar-backdrop" onClick={() => setSidebarOpen(false)} />}

      <aside className={`chat-sidebar ${sidebarOpen ? 'chat-sidebar-open' : ''}`}>
        <div className="chat-sidebar-header">
          <span className="chat-sidebar-brand">
            <BrandLogo size={24} />
            DineMind
          </span>
          <button
            type="button"
            className="chat-sidebar-close"
            onClick={() => setSidebarOpen(false)}
            aria-label="Close menu"
          >
            <X size={18} />
          </button>
        </div>

        <div className="chat-sidebar-scroll">
          <Button className="btn-block" onClick={closeSidebarAnd(startNewChat)}>
            + New chat
          </Button>
          <div className="chat-search-wrap">
            <svg className="chat-search-icon" viewBox="0 0 24 24" width="16" height="16" fill="none" aria-hidden="true">
              <circle cx="11" cy="11" r="7" stroke="currentColor" strokeWidth="2" />
              <path d="M20 20L16.5 16.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            </svg>
            <input
              type="text"
              className="chat-search-input"
              placeholder="Search chats…"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              aria-label="Search conversations"
            />
          </div>
          {conversationGroups.size === 0 && searchQuery.trim() && (
            <p className="chat-search-empty">No chats match "{searchQuery.trim()}"</p>
          )}
          {[...conversationGroups.entries()].map(([label, group]) => (
            <div className="conversation-group" key={label}>
              <p className="conversation-group-label">{label}</p>
              <ul className="conversation-list">
                {group.map((c) => (
                  <li key={c.id}>
                    <button
                      type="button"
                      className={`conversation-item ${c.id === conversationId ? 'conversation-item-active' : ''}`}
                      onClick={closeSidebarAnd(() => selectConversation(c.id))}
                      title={c.title || formatConversationLabel(c.created_at)}
                    >
                      {c.title || formatConversationLabel(c.created_at)}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="chat-sidebar-footer">
          <span className="chat-avatar" aria-hidden="true">
            {avatarInitial(user)}
          </span>
          <span className="chat-sidebar-user">
            <span className="chat-sidebar-name">{user?.display_name || user?.email}</span>
            {user?.display_name && <span className="chat-sidebar-email">{user.email}</span>}
          </span>
          <button type="button" className="chat-sidebar-logout" onClick={handleLogout} aria-label="Log out" title="Log out">
            <LogOut size={18} />
          </button>
        </div>
      </aside>

      <div className="chat-main">
        <div className="chat-mobile-header">
          <button
            type="button"
            className="chat-sidebar-toggle"
            onClick={() => setSidebarOpen(true)}
            aria-label="Open menu"
          >
            <Menu size={20} />
          </button>
          <span className="chat-mobile-brand">
            <BrandLogo size={20} />
            DineMind
          </span>
        </div>

        {error && (
          // Longer than Toast's 5s default - this covers things like the AI
          // assistant being down, which the user needs time to actually
          // register rather than have vanish while they're still reading.
          <Toast message={error} duration={8000} onClose={() => setError(null)} />
        )}

        <div className="chat-messages" ref={scrollRef}>
          {historyLoading ? (
            <div className="chat-history-loading">
              <Spinner size={20} label="Loading conversation" />
            </div>
          ) : messages.length === 0 ? (
            <EmptyState
              title="Ask me anything about restaurants"
              description={
                'Try "Suggest a good vegetarian restaurant under Rs1000" or "Where should I go for a quiet date?"'
              }
            />
          ) : (
            messages.map((m) => <ChatMessageBubble key={m.id} message={m} />)
          )}

          {loading && awaitingFirstToken && (
            <div className="chat-typing">
              <Spinner size={16} label="Thinking" />
              <span>Thinking…</span>
            </div>
          )}
        </div>

        <form className="chat-input-form" onSubmit={handleSend}>
          <div className="chat-input-wrap">
            <input
              type="text"
              className="chat-input"
              placeholder="Ask about restaurants…"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              aria-label="Chat message"
            />
            <button
              type="submit"
              className="chat-send-btn"
              disabled={!input.trim() || loading}
              aria-label="Send message"
            >
              {loading ? (
                <Spinner size={16} />
              ) : (
                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" aria-hidden="true">
                  <path
                    d="M12 19V5M12 5L6 11M12 5L18 11"
                    stroke="currentColor"
                    strokeWidth="2.4"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
