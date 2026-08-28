import { X } from 'lucide-react'
import './FilterChips.css'

// The structured constraints currently shaping the search, shown above the
// input so they're visible rather than implicit.
//
// These persist across turns: say "under Rs 800" once and it keeps applying
// until removed. That's what makes "what about somewhere cheaper?" work
// without repeating yourself, and it's also why the chips have to exist - an
// invisible constraint that silently narrows every later search is the kind
// of thing a user can only discover by being confused.
//
// Labels come from the backend (SearchState.as_chips) rather than being
// formatted here, so a chip and the assistant's reply describe the same
// constraint in the same words.
export default function FilterChips({ filters, onRemove, removing }) {
  if (!filters || filters.length === 0) return null

  return (
    // The group's aria-label carries what the removed visible caption used to
    // say - the chips are self-evident sighted, but a screen reader needs to
    // know these are active constraints rather than suggestions.
    <div className="filter-chips" role="group" aria-label="Active search filters">
      {filters.map((f) => (
        <span key={f.dimension} className="filter-chip">
          <span className="filter-chip-text">{f.label}</span>
          <button
            type="button"
            className="filter-chip-remove"
            onClick={() => onRemove(f.dimension)}
            disabled={removing === f.dimension}
            aria-label={`Remove filter: ${f.label}`}
            title={`Remove ${f.label}`}
          >
            <X size={11} />
          </button>
        </span>
      ))}
    </div>
  )
}
