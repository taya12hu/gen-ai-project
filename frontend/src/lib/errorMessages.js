import { ApiError } from './api'

// Maps the backend's stable error codes (see app/api/main.py's
// _classify_chat_error) to what a user should actually read. The frontend
// owns this wording on purpose - reword any of these without touching or
// redeploying the backend at all.
const ERROR_MESSAGES = {
  llm_unavailable: 'Our AI assistant is temporarily unavailable. Please try again in a few minutes.',
  database_unavailable: "We're having trouble connecting right now. Please try again in a moment.",
  conversation_not_found: "This conversation couldn't be found - starting a new one.",
  preference_not_found: 'That preference was already removed.',
}

export const SESSION_EXPIRED_MESSAGE = 'Your session has expired. Please log in again.'

// Turns any caught error into copy that makes sense to a user - never the
// raw backend/provider exception text. Order of precedence: a recognized
// error_code > the backend's own message (already reasonable for cases like
// "Email already registered" or "Invalid email or password") > a generic
// fallback for anything unclassified.
//
// Deliberately does NOT special-case status 401 here: a 401 means different
// things depending on which endpoint it came from ("your credentials were
// wrong" on /auth/login vs "your token expired" on a protected route), and
// this function has no way to tell those apart. Callers that hit protected
// routes (see ChatPage's handleAuthError) must detect 401 themselves - via
// SESSION_EXPIRED_MESSAGE - before falling back to this for everything else.
export function friendlyErrorMessage(err) {
  if (err instanceof ApiError) {
    if (err.errorCode && ERROR_MESSAGES[err.errorCode]) return ERROR_MESSAGES[err.errorCode]
    return err.message || 'Something went wrong. Please try again.'
  }
  // Not an ApiError - fetch() itself threw, meaning the request never even
  // reached the server (offline, DNS failure, backend process down, etc).
  return "Can't reach the server. Check your connection and try again."
}
