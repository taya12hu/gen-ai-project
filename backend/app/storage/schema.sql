-- Phase 3 - Storage & Indexing.
-- Table for the cleaned restaurant data produced by Phase 2.
-- cuisines is stored as text[] (not a comma-separated string) so cuisine
-- preference matching can use a GIN index instead of ILIKE scans.

create table if not exists restaurants (
  id bigint generated always as identity primary key,
  name text not null,
  place text not null,
  city text not null,
  cuisines text[] not null,
  price numeric(10, 2) not null,
  rating numeric(2, 1) not null,
  rest_type text,
  votes integer not null default 0,
  created_at timestamptz not null default now(),
  unique (name, place)
);

-- B-tree indexes for equality/range filters (price, rating, place).
create index if not exists restaurants_place_idx on restaurants (place);
create index if not exists restaurants_price_idx on restaurants (price);
create index if not exists restaurants_rating_idx on restaurants (rating);

-- GIN index for cuisine containment queries (cuisines @> array['Chinese']).
create index if not exists restaurants_cuisines_idx on restaurants using gin (cuisines);
