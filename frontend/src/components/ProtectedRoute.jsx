import { Navigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import Spinner from './Spinner'

export default function ProtectedRoute({ children }) {
  const { isAuthenticated, ready } = useAuth()

  if (!ready) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', padding: '48px' }}>
        <Spinner size={28} label="Checking session" />
      </div>
    )
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />
  }

  return children
}
