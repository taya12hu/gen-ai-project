import Spinner from './Spinner'
import './Button.css'

export default function Button({
  children,
  variant = 'primary',
  loading = false,
  disabled = false,
  type = 'button',
  className = '',
  ...rest
}) {
  return (
    <button
      type={type}
      className={`btn btn-${variant} ${className}`.trim()}
      disabled={disabled || loading}
      aria-busy={loading}
      {...rest}
    >
      {loading && <Spinner size={16} />}
      {children}
    </button>
  )
}
