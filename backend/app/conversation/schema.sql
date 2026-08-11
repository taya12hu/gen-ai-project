-- Phase 15 - Conversation & Personalization Storage.
-- Three tables: conversations/messages give the chat multi-turn memory
-- (needed to resolve "its ambience" back to a restaurant named earlier),
-- user_preferences gives it cross-session memory (needed so "I'm
-- vegetarian" stated once keeps mattering in later conversations).

create table if not exists conversations (
  id bigint generated always as identity primary key,
  user_id bigint not null references users (id) on delete cascade,
  -- Set once, from the first user message (see add_message in store.py) - not
  -- editable after, so a sidebar can show a stable label without re-deriving
  -- it from message content on every load.
  title text,
  created_at timestamptz not null default now()
);

-- Re-running this file against a database created before `title` existed
-- needs an explicit add, since `create table if not exists` is a no-op once
-- the table's already there.
alter table conversations add column if not exists title text;

create index if not exists conversations_user_id_idx on conversations (user_id);

create table if not exists messages (
  id bigint generated always as identity primary key,
  conversation_id bigint not null references conversations (id) on delete cascade,
  role text not null check (role in ('user', 'assistant')),
  content text not null,
  -- Restaurant ids the response actually grounded its reply in (assistant
  -- messages only). Lets query understanding resolve a follow-up like
  -- "what about its ambience?" against a concrete restaurant instead of
  -- re-inferring it from prose.
  mentioned_restaurant_ids bigint[],
  -- The specific restaurant_reviews.id rows shown as evidence alongside
  -- those restaurants (assistant messages only). Lets a reloaded
  -- conversation replay the exact snippets shown live instead of
  -- re-selecting a fresh "top rated" set - a structured-only query that
  -- attached no reviews at all must stay reviewless on reload too.
  mentioned_review_ids bigint[],
  created_at timestamptz not null default now()
);

-- Re-running this file against a database created before `mentioned_review_ids`
-- existed needs an explicit add, same reasoning as the `title` backfill above.
alter table messages add column if not exists mentioned_review_ids bigint[];

create index if not exists messages_conversation_id_idx on messages (conversation_id, created_at);

-- Durable, cross-session preference facts (e.g. key='dietary' value='vegetarian',
-- key='ambience' value='quiet'). Free-text key/value rather than fixed columns
-- because the set of preferences a user might state in natural language isn't
-- known up front. One row per (user, key); a restated preference overwrites
-- the old value rather than accumulating duplicates.
create table if not exists user_preferences (
  id bigint generated always as identity primary key,
  user_id bigint not null references users (id) on delete cascade,
  key text not null,
  value text not null,
  updated_at timestamptz not null default now(),
  unique (user_id, key)
);
