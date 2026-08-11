import { useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { resetPassword } from '../lib/api'
import { friendlyErrorMessage } from '../lib/errorMessages'
import Card from '../components/Card'
import { TextField } from '../components/Field'
import Button from '../components/Button'
import Alert from '../components/Alert'
import './AuthPages.css'

export default function ResetPasswordPage() {
  const [searchParams] = useSearchParams()
  const token = searchParams.get('token')

  const [newPassword, setNewPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [done, setDone] = useState(false)

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      await resetPassword(token, newPassword)
      setDone(true)
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
          <h1>Set a new password</h1>
          <p className="auth-subtitle">Choose a new password for your account.</p>
        </div>

        {!token && <Alert>This reset link is missing its token. Request a new one below.</Alert>}
        {error && <Alert>{error}</Alert>}

        {done ? (
          <Alert variant="success">
            Password updated. You can now log in with your new password.{' '}
            <Link to="/login">Log in</Link>
          </Alert>
        ) : (
          token && (
            <form className="auth-form" onSubmit={handleSubmit}>
              <TextField
                label="New password"
                type="password"
                autoComplete="new-password"
                minLength={8}
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                required
              />
              <Button type="submit" className="btn-block" loading={loading}>
                Update password
              </Button>
            </form>
          )
        )}

        <p className="auth-switch">
          <Link to="/forgot-password">Request a new reset link</Link>
        </p>
      </Card>
    </div>
  )
}
