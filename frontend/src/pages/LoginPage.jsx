import { useState } from 'react'
import { Link, Navigate, useNavigate, useSearchParams } from 'react-router-dom'
import { MessagesSquare, Target } from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import { friendlyErrorMessage, SESSION_EXPIRED_MESSAGE } from '../lib/errorMessages'
import Card from '../components/Card'
import { TextField } from '../components/Field'
import Button from '../components/Button'
import Alert from '../components/Alert'
import BrandLogo from '../components/BrandLogo'
import './AuthPages.css'

export default function LoginPage() {
  const { login, isAuthenticated } = useAuth()
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()

  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const sessionExpired = searchParams.get('reason') === 'session_expired'

  if (isAuthenticated) return <Navigate to="/" replace />

  const features = [
    { icon: Target, label: 'AI Recommendations', desc: 'Smart suggestions just for you' },
    { icon: MessagesSquare, label: 'Real Reviews', desc: 'Insights from real diners' },
  ]

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      await login(email, password)
      navigate('/')
    } catch (err) {
      setError(friendlyErrorMessage(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="auth-page">
      {/* Left Column: Hero & Brand */}
      <div className="auth-hero">
        <p className="auth-hero-eyebrow">
          <span className="auth-hero-eyebrow-item">
            <span className="auth-hero-eyebrow-dot" aria-hidden="true" />
            Smart Recommendations
          </span>
          <span className="auth-hero-eyebrow-item">
            <span className="auth-hero-eyebrow-dot" aria-hidden="true" />
            Real Reviews
          </span>
          <span className="auth-hero-eyebrow-item">
            <span className="auth-hero-eyebrow-dot" aria-hidden="true" />
            Perfect Choices
          </span>
        </p>

        <div className="auth-hero-brand">
          <div className="auth-hero-logo-wrapper">
            <BrandLogo size={80} />
            <h2 className="auth-hero-wordmark">
              Dine<span>Mind</span>
            </h2>
          </div>
          
          <div className="auth-hero-content">
            <p className="auth-hero-description">
              Your AI companion to discover the best restaurants that match your taste, mood and budget.
            </p>
            <div className="auth-hero-features">
              {features.map((f) => (
                <div className="auth-hero-feature" key={f.label}>
                  <span className="auth-hero-feature-icon" aria-hidden="true">
                    <f.icon size={18} strokeWidth={2} />
                  </span>
                  <div>
                    <p className="auth-hero-feature-label">{f.label}</p>
                    <p className="auth-hero-feature-desc">{f.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <p className="auth-hero-quote">
          Good food. Great choices.
          <span>Right around you.</span>
        </p>
      </div>

      {/* Right Column: Authentication Form */}
      <div className="auth-form-wrapper">
        <Card className="auth-card">
          <div className="auth-card-header">
            <h1>Welcome! 👋</h1>
            <p className="auth-subtitle">Login to continue discovering amazing food.</p>
          </div>

          {error ? (
            <Alert>{error}</Alert>
          ) : (
            sessionExpired && <Alert variant="info">{SESSION_EXPIRED_MESSAGE}</Alert>
          )}

          <form className="auth-form" onSubmit={handleSubmit}>
            <TextField
              label="Email"
              type="email"
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
            <TextField
              label="Password"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
            <Button type="submit" className="btn-block" loading={loading}>
              Log in
            </Button>
          </form>

          <p className="auth-switch">
            <Link to="/forgot-password">Forgot password?</Link>
          </p>
          <p className="auth-switch">
            Don't have an account? <Link to="/register">Register</Link>
          </p>
        </Card>
      </div>
    </div>
  )
}