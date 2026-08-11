-- Phase 14 - Review Ingestion & Embedding.
-- Individual customer reviews, re-derived from the raw reviews_list column
-- dropped in Phase 2 cleaning, matched back to restaurants.id via the same
-- (name, place) key Phase 2 dedupes on. Each review gets a local
-- sentence-transformers embedding (all-MiniLM-L6-v2, 384-dim) so qualitative
-- queries ("quiet", "good ambience") can be answered with semantic search
-- instead of exact keyword matching.

create extension if not exists vector;

create table if not exists restaurant_reviews (
  id bigint generated always as identity primary key,
  restaurant_id bigint not null references restaurants (id) on delete cascade,
  review_rating numeric(2, 1),
  review_text text not null,
  embedding vector(384),
  created_at timestamptz not null default now()
);

create index if not exists restaurant_reviews_restaurant_id_idx
  on restaurant_reviews (restaurant_id);
