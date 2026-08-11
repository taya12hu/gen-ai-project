import Card from './Card'
import ReviewSnippet from './ReviewSnippet'
import './RestaurantCard.css'

export default function RestaurantCard({ restaurant }) {
  const snippets = restaurant.review_snippets || []

  return (
    <Card className="restaurant-card">
      <div className="restaurant-card-header">
        <div className="restaurant-card-icon" aria-hidden="true">
          🍽️
        </div>
        <div className="restaurant-card-heading">
          <h3>{restaurant.name}</h3>
          <p className="restaurant-meta">
            {restaurant.place} · {restaurant.cuisines.join(', ')}
          </p>
        </div>
        <span className="rating-badge">{restaurant.rating}★</span>
      </div>
      <p className="restaurant-meta">
        Rs {restaurant.price.toFixed(0)} for two · {restaurant.votes} votes
        {restaurant.rest_type ? ` · ${restaurant.rest_type}` : ''}
      </p>
      {snippets.length > 0 && (
        <div className="restaurant-reviews-section">
          <p className="restaurant-reviews-label">What people say</p>
          <ul className="restaurant-review-snippets">
            {snippets.map((s, i) => (
              <ReviewSnippet key={i} text={s.text} rating={s.rating} />
            ))}
          </ul>
        </div>
      )}
    </Card>
  )
}
