import Card from './Card'
import './RestaurantCard.css'

export default function RestaurantCard({ restaurant }) {
  return (
    <Card className="restaurant-card">
      <div className="restaurant-card-header">
        <h3>{restaurant.name}</h3>
        <span className="rating-badge">{restaurant.rating}★</span>
      </div>
      <p className="restaurant-meta">
        {restaurant.place} · {restaurant.cuisines.join(', ')}
      </p>
      <p className="restaurant-meta">
        Rs {restaurant.price.toFixed(0)} for two · {restaurant.votes} votes
        {restaurant.rest_type ? ` · ${restaurant.rest_type}` : ''}
      </p>
    </Card>
  )
}
