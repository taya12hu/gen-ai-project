import { useId } from 'react'
import './Field.css'

export function TextField({ label, error, ...rest }) {
  const id = useId()
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <input id={id} className={error ? 'has-error' : ''} aria-invalid={Boolean(error)} {...rest} />
      {error && (
        <span className="field-error" role="alert">
          {error}
        </span>
      )}
    </div>
  )
}

export function SelectField({ label, error, children, ...rest }) {
  const id = useId()
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <select id={id} className={error ? 'has-error' : ''} aria-invalid={Boolean(error)} {...rest}>
        {children}
      </select>
      {error && (
        <span className="field-error" role="alert">
          {error}
        </span>
      )}
    </div>
  )
}
