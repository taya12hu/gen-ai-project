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

// Streams a chat reply token by token instead of waiting for the full
// reply. Can't use the browser's built-in EventSource here - it only
// supports GET requests and can't send an Authorization header - so this
// reads the response body manually and parses the same SSE format by hand.
export async function sendChatMessageStream(conversationId, message, token, { onToken, onDone, onError }) {
  const res = await fetch(`${API_BASE_URL}/chat/conversations/${conversationId}/messages/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ message }),
  })

  if (!res.ok || !res.body) {
    let data = null
    try {
      data = await res.json()
    } catch {
      // no JSON body
    }
    const { message, errorCode } = parseErrorDetail(data?.detail)
    throw new ApiError(message, res.status, errorCode)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let boundary
    while ((boundary = buffer.indexOf('\n\n')) !== -1) {
      const rawEvent = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)

      const event = parseSseEvent(rawEvent)
      if (!event) continue
      if (event.type === 'token') onToken?.(event.data.text)
      else if (event.type === 'done') onDone?.(event.data)
      else if (event.type === 'error') onError?.(new ApiError(event.data.message, 500, event.data.error_code))
    }
  }
}
