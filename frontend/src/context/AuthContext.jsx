import { createContext, useContext, useEffect, useState } from 'react'
import * as api from '../lib/api'

const AuthContext = createContext(null)

const STORAGE_KEY = 'auth'

function loadStored() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : null
  } catch {
    return null
  }
}

export function AuthProvider({ children }) {
  const [auth, setAuth] = useState(loadStored)
  const [ready, setReady] = useState(false)

  useEffect(() => {
    // Validate the stored token is still good (not expired/revoked) on load.
    if (!auth?.token) {
      setReady(true)
      return
    }
    api
      .fetchMe(auth.token)
      .then((user) => setAuth({ token: auth.token, user }))
      .catch(() => setAuth(null))
      .finally(() => setReady(true))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (auth) localStorage.setItem(STORAGE_KEY, JSON.stringify(auth))
    else localStorage.removeItem(STORAGE_KEY)
  }, [auth])

  async function login(email, password) {
    const data = await api.login({ email, password })
    setAuth({ token: data.access_token, user: data.user })
  }

  async function register(email, password, displayName) {
    const data = await api.register({ email, password, displayName })
    setAuth({ token: data.access_token, user: data.user })
  }

  function logout() {
    setAuth(null)
  }

  const value = {
    user: auth?.user ?? null,
    token: auth?.token ?? null,
    isAuthenticated: Boolean(auth?.token),
    ready,
    login,
    register,
    logout,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within an AuthProvider')
  return ctx
}
