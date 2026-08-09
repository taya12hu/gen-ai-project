const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

export class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.status = status
  }
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
    const message = Array.isArray(data?.detail)
      ? data.detail[0]?.msg || 'Request failed'
      : data?.detail || 'Request failed'
    throw new ApiError(message, res.status)
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

export function fetchOptions() {
  return request('/options')
}

export function fetchRecommendation({ place, cuisines, maxPrice, minRating }, token) {
  return request('/recommend', {
    method: 'POST',
    token,
    body: {
      place,
      cuisines,
      max_price: maxPrice === '' || maxPrice == null ? null : Number(maxPrice),
      min_rating: minRating === '' || minRating == null ? null : Number(minRating),
    },
  })
}
