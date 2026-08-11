import { useState } from 'react'
import './ReviewSnippet.css'

const TRUNCATE_LENGTH = 110

function truncateAtWord(text, maxLength) {
  if (text.length <= maxLength) return text
  const cut = text.slice(0, maxLength)
  const lastSpace = cut.lastIndexOf(' ')
  return (lastSpace > 40 ? cut.slice(0, lastSpace) : cut).trimEnd() + '…'
}

export default function ReviewSnippet({ text, rating }) {
  const [expanded, setExpanded] = useState(false)
  const isLong = text.length > TRUNCATE_LENGTH
  const display = expanded || !isLong ? text : truncateAtWord(text, TRUNCATE_LENGTH)

  return (
    <li className="review-snippet">
      <p className="review-snippet-text">
        {rating != null && <span className="review-snippet-rating">{rating}★</span>}
        <span className="review-snippet-quote">"{display}"</span>
        {isLong && (
          <button
            type="button"
            className="review-snippet-toggle"
            onClick={() => setExpanded((prev) => !prev)}
          >
            {expanded ? 'Show less' : 'Read more'}
          </button>
        )}
      </p>
    </li>
  )
}
