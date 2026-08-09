import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import Button from './Button'
import './Navbar.css'

export default function Navbar() {
  const { isAuthenticated, user, logout } = useAuth()
  const navigate = useNavigate()

  function handleLogout() {
    logout()
    navigate('/login')
  }

  return (
    <nav className="navbar" aria-label="Main navigation">
      <Link to="/" className="brand">
        Restaurant Recommender
      </Link>

      {isAuthenticated ? (
        <div className="navbar-right">
          <span className="navbar-greeting">Hi, {user?.display_name || user?.email}</span>
          <Button variant="secondary" onClick={handleLogout}>
            Log out
          </Button>
        </div>
      ) : (
        <div className="navbar-right">
          <Link to="/login" className="nav-link">
            Log in
          </Link>
          <Link to="/register" className="nav-link">
            Register
          </Link>
        </div>
      )}
    </nav>
  )
}
