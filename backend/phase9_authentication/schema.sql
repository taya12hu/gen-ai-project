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
