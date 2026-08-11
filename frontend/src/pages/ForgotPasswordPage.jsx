import { useState } from 'react'
import { Link } from 'react-router-dom'
import { forgotPassword } from '../lib/api'
import { friendlyErrorMessage } from '../lib/errorMessages'
import Card from '../components/Card'
import { TextField } from '../components/Field'
import Button from '../components/Button'
import Alert from '../components/Alert'
import './AuthPages.css'

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [sent, setSent] = useState(false)

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      await forgotPassword(email)
      setSent(true)
    } catch (err) {
      setError(friendlyErrorMessage(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="auth-page">
      <Card className="auth-card">
        <div className="auth-card-header">
          <h1>Reset your password</h1>
          <p className="auth-subtitle">We'll email you a link to reset it.</p>
        </div>

        {error && <Alert>{error}</Alert>}

        {sent ? (
          <Alert variant="success">
            If an account with that email exists, a password reset link has been sent. Check your inbox.
          </Alert>
        ) : (
          <form className="auth-form" onSubmit={handleSubmit}>
            <TextField
              label="Email"
              type="email"
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
            <Button type="submit" className="btn-block" loading={loading}>
              Send reset link
            </Button>
          </form>
        )}

        <p className="auth-switch">
          Remembered it? <Link to="/login">Log in</Link>
        </p>
      </Card>
    </div>
  )
}
