import './Alert.css'

export default function Alert({ variant = 'error', children }) {
  return (
    <div className={`alert alert-${variant}`} role="alert">
      {children}
    </div>
  )
}
