import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import RestaurantCard from './RestaurantCard'
import './ChatMessageBubble.css'

export default function ChatMessageBubble({ message }) {
  const isUser = message.role === 'user'
  const restaurants = message.matched_restaurants || []

  return (
    <div className={`chat-bubble-row ${isUser ? 'chat-bubble-row-user' : ''}`}>
      <div className={`chat-bubble ${isUser ? 'chat-bubble-user' : 'chat-bubble-assistant'}`}>
        {isUser ? (
          <p>{message.content}</p>
        ) : (
          <div className="chat-bubble-markdown">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
          </div>
        )}
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
