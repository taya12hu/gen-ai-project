const PREFERENCE_LABELS = {
  dietary: 'Dietary',
  ambience: 'Ambience',
  occasion: 'Occasion',
  vibe: 'Vibe',
}

// Human-readable label for a stored preference key - shared between
// PreferencesPanel (the management view) and ChatPage (the "just
// remembered this" notice) so both describe the same fact the same way.
export function labelFor(key) {
  return PREFERENCE_LABELS[key] || key.charAt(0).toUpperCase() + key.slice(1)
}
