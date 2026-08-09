import './EmptyState.css'

export default function EmptyState({ title, description }) {
  return (
    <div className="empty-state">
      <h3>{title}</h3>
      {description && <p>{description}</p>}
    </div>
  )
}
