import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { ApiError, fetchOptions, fetchRecommendation } from '../lib/api'
import Card from '../components/Card'
import { SelectField, TextField } from '../components/Field'
import CuisineMultiSelect from '../components/CuisineMultiSelect'
import Button from '../components/Button'
import Alert from '../components/Alert'
import EmptyState from '../components/EmptyState'
import RestaurantCard from '../components/RestaurantCard'
import Spinner from '../components/Spinner'
import './RecommendPage.css'

export default function RecommendPage() {
  const { token, logout } = useAuth()
  const navigate = useNavigate()

  const [places, setPlaces] = useState([])
  const [cuisineOptions, setCuisineOptions] = useState([])
  const [optionsError, setOptionsError] = useState(null)
  const [optionsLoading, setOptionsLoading] = useState(true)

  const [place, setPlace] = useState('')
  const [cuisines, setCuisines] = useState([])
  const [maxPrice, setMaxPrice] = useState('')
  const [minRating, setMinRating] = useState('')

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)

  useEffect(() => {
    fetchOptions()
      .then((data) => {
        setPlaces(data.places)
        setCuisineOptions(data.cuisines)
      })
      .catch((err) => setOptionsError(err.message))
      .finally(() => setOptionsLoading(false))
  }, [])

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const data = await fetchRecommendation({ place, cuisines, maxPrice, minRating }, token)
      setResult(data)
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        logout()
        navigate('/login')
        return
      }
      setError(err instanceof ApiError ? err.message : 'Something went wrong. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="recommend-page">
      <div className="recommend-intro">
        <h1>Find a restaurant</h1>
        <p>Tell us what you're after, and we'll find the best fit.</p>
      </div>

      {optionsError && <Alert>Could not reach the backend: {optionsError}</Alert>}

      <Card>
        {optionsLoading ? (
          <div className="options-loading">
            <Spinner size={22} label="Loading options" />
            <span>Loading places and cuisines…</span>
          </div>
        ) : (
          <form className="preference-form" onSubmit={handleSubmit}>
            <SelectField label="Place" value={place} onChange={(e) => setPlace(e.target.value)} required>
              <option value="" disabled>
                Select a place
              </option>
              {places.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </SelectField>

            <div className="preference-cuisines">
              <CuisineMultiSelect options={cuisineOptions} selected={cuisines} onChange={setCuisines} />
            </div>

            <TextField
              label="Max price for two (Rs) — optional"
              type="number"
              min="1"
              step="50"
              placeholder="Any budget"
              value={maxPrice}
              onChange={(e) => setMaxPrice(e.target.value)}
            />

            <TextField
              label="Minimum rating — optional"
              type="number"
              min="0"
              max="5"
              step="0.1"
              placeholder="Any rating"
              value={minRating}
              onChange={(e) => setMinRating(e.target.value)}
            />

            <Button
              type="submit"
              className="btn-block preference-submit"
              loading={loading}
              disabled={!place || cuisines.length === 0}
            >
              {loading ? 'Finding restaurants…' : 'Get recommendation'}
            </Button>
          </form>
        )}
      </Card>

      {error && <Alert>{error}</Alert>}

      {!result && !loading && !error && (
        <EmptyState
          title="Ready when you are"
          description="Fill in your preferences above and we'll find a restaurant that fits."
        />
      )}

      {result && (
        <section className="results">
          {!result.found_any && (
            <EmptyState
              title="No restaurants matched"
              description="Try widening your budget, lowering the minimum rating, or picking a different place or cuisine."
            />
          )}

          {result.relaxed && (
            <Alert variant="info">
              Showing the closest matches — no restaurant fully met both your budget and minimum
              rating.
            </Alert>
          )}

          {result.matched_restaurants.length > 0 && (
            <div className="restaurant-grid">
              {result.matched_restaurants.map((r) => (
                <RestaurantCard key={r.id} restaurant={r} />
              ))}
            </div>
          )}

          {result.explanation && <p className="explanation">{result.explanation}</p>}
        </section>
      )}
    </div>
  )
}
