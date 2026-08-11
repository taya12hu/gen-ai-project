-- Phase 9 - Authentication & Authorization.
-- Extends the storage layer from Phase 3 with a users table. Passwords are
-- never stored in plaintext - only the bcrypt hash.

create table if not exists users (
  id bigint generated always as identity primary key,
  email text not null unique,
  hashed_password text not null,
  display_name text,
  created_at timestamptz not null default now()
);

-- One-time password reset tokens (see password_reset.py). Only the SHA-256
-- hash of the token is stored - a database leak alone can't be used to
-- reset anyone's password, since the raw token was only ever sent by email.
create table if not exists password_reset_tokens (
  id bigint generated always as identity primary key,
  user_id bigint not null references users (id) on delete cascade,
  token_hash text not null unique,
  expires_at timestamptz not null,
  used_at timestamptz,
  created_at timestamptz not null default now()
);

create index if not exists password_reset_tokens_user_id_idx on password_reset_tokens (user_id);
