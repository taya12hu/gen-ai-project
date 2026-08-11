import RestaurantCard from './RestaurantCard'
import './ChatMessageBubble.css'

export default function ChatMessageBubble({ message }) {
  const isUser = message.role === 'user'
  const restaurants = message.matched_restaurants || []

  return (
    <div className={`chat-bubble-row ${isUser ? 'chat-bubble-row-user' : ''}`}>
      <div className={`chat-bubble ${isUser ? 'chat-bubble-user' : 'chat-bubble-assistant'}`}>
        <p>{message.content}</p>
      </div>
      {!isUser && restaurants.length > 0 && (
        <div className="chat-bubble-restaurants">
          {restaurants.map((r) => (
            <RestaurantCard key={r.id} restaurant={r} />
          ))}
        </div>
      )}
    </div>
  )
}
