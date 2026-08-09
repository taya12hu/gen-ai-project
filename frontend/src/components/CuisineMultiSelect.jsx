import { useId, useState } from 'react'
import './CuisineMultiSelect.css'

export default function CuisineMultiSelect({ options, selected, onChange, error }) {
  const [filter, setFilter] = useState('')
  const id = useId()

  const filtered = options.filter((o) => o.toLowerCase().includes(filter.toLowerCase()))

  function toggle(cuisine) {
    if (selected.includes(cuisine)) {
      onChange(selected.filter((c) => c !== cuisine))
    } else {
      onChange([...selected, cuisine])
    }
  }

  return (
    <div className="field">
      <label htmlFor={id}>Cuisines</label>
      <div className={`multiselect ${error ? 'has-error' : ''}`}>
        <input
          id={id}
          type="text"
          placeholder="Search cuisines…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
        <div className="multiselect-options" role="group" aria-label="Cuisine options">
          {filtered.length === 0 && <p className="multiselect-empty">No cuisines match "{filter}"</p>}
          {filtered.map((c) => (
            <label key={c} className="multiselect-option">
              <input type="checkbox" checked={selected.includes(c)} onChange={() => toggle(c)} />
              {c}
            </label>
          ))}
        </div>
      </div>
      {selected.length > 0 && (
        <p className="multiselect-selected">Selected: {selected.join(', ')}</p>
      )}
      {error && (
        <span className="field-error" role="alert">
          {error}
        </span>
      )}
    </div>
  )
}
