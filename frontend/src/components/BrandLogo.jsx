import { Flame, MapPin } from 'lucide-react'
import './BrandLogo.css'

export default function BrandLogo({ size = 32 }) {
  const flameSize = Math.round(size * 0.4)
  const flameTop = Math.round(size * 0.2)

  return (
    <span className="brand-logo-mark" style={{ width: size, height: size }} aria-hidden="true">
      <MapPin size={size} strokeWidth={1.5} fill="var(--accent)" color="var(--accent)" />
      <Flame
        size={flameSize}
        strokeWidth={0}
        fill="var(--bg)"
        color="var(--bg)"
        className="brand-logo-flame"
        style={{ top: flameTop }}
      />
    </span>
  )
}
