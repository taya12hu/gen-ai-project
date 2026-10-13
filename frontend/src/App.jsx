import { BrowserRouter, Route, Routes, useLocation } from 'react-router-dom'
import { Analytics } from '@vercel/analytics/react'
import { AuthProvider } from './context/AuthContext'
import Navbar from './components/Navbar'
import ProtectedRoute from './components/ProtectedRoute'
import LoginPage from './pages/LoginPage'
import RegisterPage from './pages/RegisterPage'
import ForgotPasswordPage from './pages/ForgotPasswordPage'
import ResetPasswordPage from './pages/ResetPasswordPage'
import ChatPage from './pages/ChatPage'

function AppLayout() {
  const location = useLocation()
  // Chat page ('/') builds its own header + identity into the sidebar,
  // Claude-style, so the generic top navbar would just be a duplicate.
  const hideNavbar = location.pathname === '/login' || location.pathname === '/'

  return (
    <>
      {!hideNavbar && <Navbar />}
      <main>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
          <Route path="/forgot-password" element={<ForgotPasswordPage />} />
          <Route path="/reset-password" element={<ResetPasswordPage />} />
          <Route
            path="/"
            element={
              <ProtectedRoute>
                <ChatPage />
              </ProtectedRoute>
            }
          />
        </Routes>
      </main>
    </>
  )
}

export default function App() {
  return (
    <>
      <BrowserRouter>
        <AuthProvider>
          <AppLayout />
        </AuthProvider>
      </BrowserRouter>
      {/* Vercel Web Analytics. The `/react` entry point, not `/next` - the
          snippet in Vercel's own dashboard assumes Next.js, and importing it
          here would try to resolve `next/navigation` and fail the Vite build.
          Sits outside the router because it takes no route props: every route
          in this app is static, so the injected script's own history tracking
          records navigations correctly. Passing a `route` prop is what turns
          auto-tracking off, and is only needed for dynamic route patterns
          (e.g. /users/[id]) that should be grouped rather than recorded as
          distinct URLs. */}
      <Analytics />
    </>
  )
}
