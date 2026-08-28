const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

export class ApiError extends Error {
  constructor(message, status, errorCode = null) {
    super(message)
    this.status = status
    this.errorCode = errorCode
  }
}

// The backend sends error detail in one of three shapes: a plain string
// (older/simple endpoints, e.g. "Email already registered"), a structured
// {error_code, message} object (chat endpoints - see _classify_chat_error
// in api.py), or a FastAPI validation error array. This normalizes all
// three into a single (message, errorCode) pair.
function parseErrorDetail(detail) {
  if (detail && typeof detail === 'object' && !Array.isArray(detail)) {
    return { message: detail.message || 'Request failed', errorCode: detail.error_code || null }
  }
  if (Array.isArray(detail)) {
    return { message: detail[0]?.msg || 'Request failed', errorCode: null }
  }
  return { message: detail || 'Request failed', errorCode: null }
}

async function request(path, { method = 'GET', body, token } = {}) {
  const headers = { 'Content-Type': 'application/json' }
  if (token) headers.Authorization = `Bearer ${token}`

  const res = await fetch(`${API_BASE_URL}${path}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  })

  let data = null
  try {
    data = await res.json()
  } catch {
    // no JSON body (e.g. network error before a response was formed)
  }

  if (!res.ok) {
    const { message, errorCode } = parseErrorDetail(data?.detail)
    throw new ApiError(message, res.status, errorCode)
  }

  return data
}

export function register({ email, password, displayName }) {
  return request('/auth/register', {
    method: 'POST',
    body: { email, password, display_name: displayName || null },
  })
}

export function login({ email, password }) {
  return request('/auth/login', { method: 'POST', body: { email, password } })
}

export function fetchMe(token) {
  return request('/auth/me', { token })
}

export function forgotPassword(email) {
  return request('/auth/forgot-password', { method: 'POST', body: { email } })
}

export function resetPassword(token, newPassword) {
  return request('/auth/reset-password', {
    method: 'POST',
    body: { token, new_password: newPassword },
  })
}

export function createConversation(token) {
  return request('/chat/conversations', { method: 'POST', token })
}

export function listConversations(token) {
  return request('/chat/conversations', { token })
}

export function fetchConversationMessages(conversationId, token) {
  return request(`/chat/conversations/${conversationId}/messages`, { token })
}

export function sendChatMessage(conversationId, message, token) {
  return request(`/chat/conversations/${conversationId}/messages`, {
    method: 'POST',
    token,
    body: { message },
  })
}

export function listPreferences(token) {
  return request('/preferences', { token })
}

export function deletePreference(key, token) {
  return request(`/preferences/${encodeURIComponent(key)}`, { method: 'DELETE', token })
}

// SSE events look like "event: <type>\ndata: <json>\n\n" - each blank line
// ends one event. fetch()'s reader hands back raw byte chunks that don't
// line up with event boundaries, so this buffers text until it sees a full
// "\n\n"-terminated block before parsing it.
function parseSseEvent(rawEvent) {
  let type = 'message'
  const dataLines = []
  for (const line of rawEvent.split('\n')) {
    if (line.startsWith('event:')) type = line.slice(6).trim()
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
  }
  if (dataLines.length === 0) return null
  try {
    return { type, data: JSON.parse(dataLines.join('\n')) }
  } catch {
    return null
  }
}

const STREAM_MAX_RETRIES = 3
const STREAM_RETRY_BASE_MS = 800
const STREAM_RETRY_MAX_MS = 6000

// Full jitter (AWS's term for it): each retry waits somewhere between 0 and
// an exponentially growing ceiling, so a burst of clients that all dropped
// at the same moment (e.g. a blip on the server) don't all reconnect in
// lockstep and hammer it a second time.
function retryDelay(attempt) {
  const ceiling = Math.min(STREAM_RETRY_BASE_MS * 2 ** attempt, STREAM_RETRY_MAX_MS)
  return Math.random() * ceiling
}

function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, ms)
    signal?.addEventListener(
      'abort',
      () => {
        clearTimeout(timer)
        reject(new DOMException('Aborted', 'AbortError'))
      },
      { once: true }
    )
  })
}

// Runs one connect-and-read attempt of the SSE stream. Resolves to:
//   'done'          - the server sent a "done" event; the turn completed normally
//   ApiError        - a non-retryable rejection: either a bad HTTP status, or
//                      a server-sent "error" event. The backend has already
//                      tried its own Groq->Gemini fallback by the time it
//                      emits either of those, so retrying the same request
//                      here would just reproduce the same failure.
//   'disconnected'  - the connection was cut (network drop, or the reader
//                      simply ended) before the server told us the turn
//                      either finished or failed. This is the retryable case.
// AbortError propagates instead of resolving, so the caller can tell a
// deliberate cancel apart from a real failure.
async function runStreamAttempt(conversationId, message, token, { onToken, onDone, signal }) {
  try {
    const res = await fetch(`${API_BASE_URL}/chat/conversations/${conversationId}/messages/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ message }),
      signal,
    })

    if (!res.ok || !res.body) {
      let data = null
      try {
        data = await res.json()
      } catch {
        // no JSON body
      }
      const { message: msg, errorCode } = parseErrorDetail(data?.detail)
      return new ApiError(msg, res.status, errorCode)
    }

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) return 'disconnected'
      buffer += decoder.decode(value, { stream: true })

      let boundary
      while ((boundary = buffer.indexOf('\n\n')) !== -1) {
        const rawEvent = buffer.slice(0, boundary)
        buffer = buffer.slice(boundary + 2)

        const event = parseSseEvent(rawEvent)
        if (!event) continue
        if (event.type === 'token') onToken?.(event.data.text)
        else if (event.type === 'done') {
          onDone?.(event.data)
          return 'done'
        } else if (event.type === 'error') {
          return new ApiError(event.data.message, 500, event.data.error_code)
        }
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') throw err
    return 'disconnected' // fetch/reader network failure mid-connect or mid-read
  }
}

// Streams a chat reply token by token instead of waiting for the full
// reply. Can't use the browser's built-in EventSource here - it only
// supports GET requests and can't send an Authorization header - so this
// reads the response body manually and parses the same SSE format by hand.
//
// Retries a dropped connection with backoff instead of surfacing it as a
// hard failure. This is safe to do blindly (no idempotency key, no "did the
// last attempt actually land" check) because the backend only persists the
// user's message and the assistant's reply once the reply is fully
// generated (see finalize_chat_turn in app/chat/service.py) - a connection
// dropped mid-stream means nothing was written server-side, so re-sending
// is a clean retry, not a duplicate. onReconnecting fires before each retry
// so the caller can discard whatever partial text it displayed from the
// failed attempt rather than appending a second, disjoint generation onto it.
//
// Pass an AbortSignal to let the caller cancel outright (e.g. the user
// switches conversations mid-stream) - that propagates as AbortError and is
// never retried.
export async function sendChatMessageStream(
  conversationId,
  message,
  token,
  { onToken, onDone, onError, onReconnecting, signal } = {}
) {
  for (let attempt = 0; attempt <= STREAM_MAX_RETRIES; attempt++) {
    const outcome = await runStreamAttempt(conversationId, message, token, { onToken, onDone, signal })

    if (outcome === 'done') return
    if (outcome instanceof ApiError) {
      onError?.(outcome)
      return
    }

    if (attempt === STREAM_MAX_RETRIES) {
      onError?.(new ApiError('Lost connection to the server. Please try again.', 0, 'stream_disconnected'))
      return
    }
    onReconnecting?.(attempt + 1)
    await sleep(retryDelay(attempt), signal)
  }
}

export function fetchConversationFilters(conversationId, token) {
  return request(`/chat/conversations/${conversationId}/filters`, { token })
}

// Returns the filters that remain, so the caller updates from an
// authoritative list rather than guessing what removal left behind.
export function clearConversationFilter(conversationId, dimension, token) {
  return request(`/chat/conversations/${conversationId}/filters/${encodeURIComponent(dimension)}`, {
    method: 'DELETE',
    token,
  })
}
